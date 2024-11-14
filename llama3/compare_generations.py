import torch
import time
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM,AutoConfig
from llama3_8B import Llama, Transformer, Tokenizer  # Import my implementation
import random
import numpy as np
import copy

# Enforce deterministic behavior
# seed = 0
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)

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


@torch.no_grad()
def generate(model, tokenizer,prompt):
    temperature = 1e-8
    top_p = 1e-8
    max_seq_len = 128
    max_gen_len = 64
    max_batch_size = 4
    input_ids = None

    with torch.no_grad():
        if isinstance(model, LlamaForCausalLM):
            # For Hugging Face model
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            hf_inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True, padding = True).to(device)
            input_ids = hf_inputs.input_ids
            attention_mask = hf_inputs["attention_mask"]

            hf_result = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_gen_len,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                # eos_token_id=terminators,
                pad_token_id=tokenizer.eos_token_id
            )
            hf_result = tokenizer.decode(hf_result[0], skip_special_tokens=True)
            return hf_result
        elif isinstance(model, Transformer):
            # For my implementation
            my_result = model.text_completion(
                        prompt,
                        max_gen_len=max_gen_len,
                        temperature=temperature,
                        top_p=top_p,
                        )
            return my_result
        else:
            raise ValueError("Unsupported model type")


# Initialize Hugging Face implementation
print("\nTrying to load Hugging Face model")
hf_path = "/home/huang717/.llama/checkpoints/Llama-3-8B-HF"
# hf_model = LlamaForCausalLM.from_pretrained(
#     hf_path,
#     torch_dtype=torch.bfloat16,
#     attn_implementation = 'sdpa').cuda()
# hf_config = AutoConfig.from_pretrained(hf_path)
# hf_config.attn_implementation = 'sdpa'

# hf_model = AutoModelForCausalLM.from_pretrained(hf_path,
#                                                 torch_dtype=torch.bfloat16,
#                                                 attn_implementation = 'sdpa',
#                                                 device_map=device
#                                                 )
hf_config_1 = AutoConfig.from_pretrained(hf_path)
hf_config_2 = AutoConfig.from_pretrained(hf_path)
hf_config_1._attn_implementation = 'eager'
hf_config_2._attn_implementation = 'sdpa'

# Initialize my implementation
ckpt_dir = "/home/huang717/.llama/checkpoints/Llama-3-8B/"
tokenizer_path = "/home/huang717/.llama/checkpoints/Llama-3-8B/tokenizer.model"
max_seq_len = 128
max_batch_size = 1
# my_llama = Llama.build(
#     ckpt_dir=ckpt_dir,
#     tokenizer_path=tokenizer_path,
#     max_seq_len=max_seq_len,
#     max_batch_size=max_batch_size,
#     model_parallel_size=1
# )
# my_model = my_llama.model.cuda()
# my_tokenizer = my_llama.tokenizer
# my_model.return_activation = True  # Enable returning activation

model_1 = AutoModelForCausalLM.from_pretrained(hf_path, config = hf_config_1).to(device)
model_2 = AutoModelForCausalLM.from_pretrained(hf_path, config = hf_config_2).to(device)
hf_tokenizer = AutoTokenizer.from_pretrained(hf_path)


device_1 = next(model_1.parameters()).device
print(f"Model_1 is on: {device_1}")
device_2 = next(model_2.parameters()).device
print(f"Model_2 is on: {device_2}") 


print(f"GPU after loading model:")
print_gpu_memory()


# Get outputs
prompt = "The theory of Universal Approximation state that"

print("Trying to generate")
start_time = time.time()
with torch.no_grad():
    prompt = [prompt]
    model_1_outputs = generate(model_1,hf_tokenizer, prompt)
    model_2_outputs = generate(model_2,hf_tokenizer, prompt)
    print(f"\n1st output:\n{model_1_outputs}")
    print(f"2nd output:\n{model_2_outputs}")
end_time = time.time()
print(f"\nIt took {end_time-start_time:.2f} seconds to generate and compare outputs")