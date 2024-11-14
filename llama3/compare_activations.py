import torch
import time
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM,AutoConfig
from llama3_8B import Llama, Transformer, Tokenizer  # Import my implementation
import random
import numpy as np
import copy

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

def print_gpu_memory():
    print("\nGPU Memory Summary:")
    print("-" * 40)
    
    # Convert bytes to GB
    def bytes_to_gb(x): 
        return x / 1024**3
    
    # Current memory
    allocated = bytes_to_gb(torch.cuda.memory_allocated('cuda:0'))
    reserved = bytes_to_gb(torch.cuda.memory_reserved('cuda:0'))
    
    # Peak memory
    max_allocated = bytes_to_gb(torch.cuda.max_memory_allocated('cuda:0'))
    max_reserved = bytes_to_gb(torch.cuda.max_memory_reserved('cuda:0'))
    
    print(f"Current Memory Usage:")
    print(f"  Allocated: {allocated:.2f} GB")
    print(f"  Reserved:  {reserved:.2f} GB")
    print(f"\nPeak Memory Usage:")
    print(f"  Max Allocated: {max_allocated:.2f} GB")
    print(f"  Max Reserved:  {max_reserved:.2f} GB")


print(f"GPU before loading model:")
print_gpu_memory()

# Initialize Hugging Face implementation
print("\nTrying to load Hugging Face model")
hf_path = "/home/huang717/.llama/checkpoints/Llama-3-8B-HF"

hf_config_1 = AutoConfig.from_pretrained(hf_path)
hf_config_2 = AutoConfig.from_pretrained(hf_path)
hf_config_1._attn_implementation = 'sdpa'
hf_config_2._attn_implementation = 'eager'

# Initialize my implementation
ckpt_dir = "/home/huang717/.llama/checkpoints/Llama-3-8B/"
tokenizer_path = "/home/huang717/.llama/checkpoints/Llama-3-8B/tokenizer.model"
max_seq_len = 128
max_batch_size = 1

my_llama_1 = Llama.build(
    ckpt_dir=ckpt_dir,
    tokenizer_path=tokenizer_path,
    max_seq_len=max_seq_len,
    max_batch_size=max_batch_size,
    model_parallel_size=1,
    sdpa=True
)
model_1 = my_llama_1.model.cuda()
model_1.return_activation = True  # Enable returning activation
my_tokenizer = my_llama_1.tokenizer

my_llama_2 = Llama.build(
    ckpt_dir=ckpt_dir,
    tokenizer_path=tokenizer_path,
    max_seq_len=max_seq_len,
    max_batch_size=max_batch_size,
    model_parallel_size=1,
    sdpa=False
)
model_2 = my_llama_2.model.cuda()
model_2.return_activation = True

# Initialize HF implementations
# model_1 = AutoModelForCausalLM.from_pretrained(hf_path, config = hf_config_1,torch_dtype=torch.bfloat16).to(device)
# model_2 = AutoModelForCausalLM.from_pretrained(hf_path, config = hf_config_2,torch_dtype=torch.bfloat16).to(device)

device_1 = next(model_1.parameters()).device
print(f"Model_1 is on: {device_1}")
device_2 = next(model_2.parameters()).device
print(f"Model_2 is on: {device_2}") 


model_1.eval()
model_2.eval()

hf_tokenizer = AutoTokenizer.from_pretrained(hf_path)


print(f"GPU after loading model:")
print_gpu_memory()



def get_all_layers_activations(model, input_ids):
    """Helper function to get all layers' activations."""
    with torch.no_grad():
        if isinstance(model, LlamaForCausalLM):
            # For Hugging Face model
            outputs = model(input_ids, output_hidden_states=True)
            return outputs.hidden_states
        elif isinstance(model, Transformer):
            # For my implementation
            _, activations = model(input_ids, start_pos=0)
            return activations
        else:
            raise ValueError("Unsupported model type")

def cosine_sim(A, B):
    dot_product = torch.dot(A, B)
    magnitude_A = torch.norm(A)
    magnitude_B = torch.norm(B)
    
    cosine_similarity = dot_product / (magnitude_A * magnitude_B)
    return cosine_similarity
    
prompt = "The theory of universal approximation states that"
input_ids = hf_tokenizer.encode(prompt, return_tensors="pt").to(device)
# input_ids = torch.tensor([my_tokenizer.encode(prompt, bos=True, eos=False)]).cuda()
print(f"Input tokens length: {input_ids.shape}")

# Get activations for all layers
print("\nTrying to get activations for all layers")

start_time = time.time()
with torch.no_grad():
    activations_list_1 = get_all_layers_activations(model_1, input_ids)
    activations_list_2 = get_all_layers_activations(model_2, input_ids)
end_time = time.time()

print(f"Shapes of activations: {[a.shape for a in activations_list_1]}")

mse_list = np.zeros(len(activations_list_1))
mae_list = np.zeros(len(activations_list_1))
cos_sims = np.zeros(len(activations_list_1))

for i, (my_activations,hf_activations) in enumerate(zip(activations_list_1, activations_list_2)):
    if my_activations.shape == hf_activations.shape:
        mse = torch.mean((my_activations - hf_activations) ** 2)
        mse_list[i] = mse
        
        mae = torch.mean(torch.abs(my_activations - hf_activations))
        mae_list[i] = mae
        
        cosine_similarity = torch.nn.functional.cosine_similarity(my_activations.flatten(), hf_activations.flatten(), dim=0)
        # cosine_similarity = cosine_sim(my_activations.flatten(), hf_activations.flatten())
        if cosine_similarity != 1:
            print(f"After Block_{i}, two tensors equal? -- {torch.equal(my_activations,hf_activations)}")
            # # For each token
            # for j in range(input_ids.shape[-1]):
            #     cos_sim_per_token = cosine_sim(my_activations.squeeze()[j], hf_activations.squeeze()[j])
            #     print(cos_sim_per_token)
            #     if cos_sim_per_token != 1:
            #         print(f"Two tensors equal? -- {torch.equal(my_activations.squeeze()[j],hf_activations.squeeze()[j])}")
        cos_sims[i] = cosine_similarity
    else:
        print(f"Activation shapes do not match at layer {i}. Cannot compute similarity.")

print("   MSE           MAE         COSINE")
np.set_printoptions(precision=10, suppress=True)
print(np.vstack((mse_list,mae_list,cos_sims)).T)

print(f"It took {end_time-start_time:.2f} seconds to compare activations")

# # Get outputs
# print("Trying to generate")
# start_time = time.time()
# with torch.no_grad():
#     prompt = [prompt]
#     my_outputs, hf_outputs = generate(my_llama,my_tokenizer, hf_model, hf_tokenizer, prompt)
#     print(f"My output:\n{prompt[0] + my_outputs[0]['generation']}")
#     print(f"HF output:\n{hf_outputs}")
# end_time = time.time()
# print(f"It took {end_time-start_time:.2f} seconds to generate and compare outputs")
