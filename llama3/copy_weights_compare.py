import torch
import time
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM,AutoConfig
from llama3_8B import Llama, Transformer, Tokenizer  # Import my implementation
import random
import numpy as np

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')



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

hf_config = AutoConfig.from_pretrained(hf_path)
hf_config._attn_implementation = 'eager'

hf_model = AutoModelForCausalLM.from_pretrained(hf_path, config = hf_config).cuda()
# new_hf_model = AutoModelForCausalLM.from_pretrained(hf_path, config = hf_config).cuda()

hf_tokenizer = AutoTokenizer.from_pretrained(hf_path)

# Initialize my implementation
ckpt_dir = "/home/huang717/.llama/checkpoints/Llama-3-8B/"
tokenizer_path = "/home/huang717/.llama/checkpoints/Llama-3-8B/tokenizer.model"
max_seq_len = 128
max_batch_size = 1
my_llama = Llama.build(
    ckpt_dir=ckpt_dir,
    tokenizer_path=tokenizer_path,
    max_seq_len=max_seq_len,
    max_batch_size=max_batch_size,
    model_parallel_size=1,
    sdpa=False
)
my_model = my_llama.model.to(device)
my_tokenizer = my_llama.tokenizer

# Set both models to evaluation mode
hf_model.eval()
my_model.eval()


print("Hugging Face Llama:")
print(hf_model)
"""
Hugging Face Llama:
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(128256, 4096)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaSdpaAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (v_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=4096, out_features=14336, bias=False)
          (up_proj): Linear(in_features=4096, out_features=14336, bias=False)
          (down_proj): Linear(in_features=14336, out_features=4096, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((4096,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=4096, out_features=128256, bias=False)
)
"""

print("\n\nMy Llama:")
print(my_model)
"""
My Llama:
Transformer(
  (tok_embeddings): Embedding(128256, 4096)
  (layers): ModuleList(
    (0-31): 32 x TransformerBlock(
      (attention): Attention(
        (wq): Linear(in_features=4096, out_features=4096, bias=False)
        (wk): Linear(in_features=4096, out_features=1024, bias=False)
        (wv): Linear(in_features=4096, out_features=1024, bias=False)
        (wo): Linear(in_features=4096, out_features=4096, bias=False)
      )
      (feed_forward): FeedForward(
        (w1): Linear(in_features=4096, out_features=14336, bias=False)
        (w3): Linear(in_features=4096, out_features=14336, bias=False)
        (w2): Linear(in_features=14336, out_features=4096, bias=False)
      )
      (attention_norm): RMSNorm()
      (ffn_norm): RMSNorm()
    )
  )
  (norm): RMSNorm()
  (output): Linear(in_features=4096, out_features=128256, bias=False)
  (rotary_emb): LlamaRotaryEmbedding()
)
"""

def compare_matrix_rows(matrix1: torch.Tensor, 
                       matrix2: torch.Tensor) -> bool:
    """
    Compare if two PyTorch tensors contain the same rows regardless of order.
    
    Args:
        matrix1: First matrix as PyTorch tensor
        matrix2: Second matrix as PyTorch tensor
        
    Returns:
        bool: True if matrices contain same rows, False otherwise
        
    Example:
        >>> m1 = torch.tensor([[1, 2], [3, 4]])
        >>> m2 = torch.tensor([[3, 4], [1, 2]])
        >>> compare_matrix_rows(m1, m2)
        True
    """
    # Check if shapes match
    if matrix1.shape != matrix2.shape:
        return False
    
    # Sort both matrices along rows for comparison
    # We use lexicographical sorting
    sorted1, _ = torch.sort(matrix1.view(-1, matrix1.shape[-1]), dim=0)
    sorted2, _ = torch.sort(matrix2.view(-1, matrix2.shape[-1]), dim=0)
    
    # Compare sorted tensors
    return torch.equal(sorted1, sorted2)

def compare_weights_matrices_rows(hf_model, custom_model):
    # Copy layer weights
    for i, (custom_layer, hf_layer) in enumerate(zip(custom_model.layers, hf_model.model.layers)):
        print(f"At Layer {i}:")
        print(f"\tWq have same rows: {compare_matrix_rows(custom_layer.attention.wq.weight.data,hf_layer.self_attn.q_proj.weight.data)}")
        print(f"\tWk have same rows: {compare_matrix_rows(custom_layer.attention.wk.weight.data,hf_layer.self_attn.k_proj.weight.data)}")
        print(f"\tWv have same rows: {compare_matrix_rows(custom_layer.attention.wv.weight.data,hf_layer.self_attn.v_proj.weight.data)}")
        print(f"\tWo have same rows: {compare_matrix_rows(custom_layer.attention.wo.weight.data,hf_layer.self_attn.o_proj.weight.data)}")

def copy_weights_to_custom_llama(hf_model, custom_model):
    """
    Copy weights from Hugging Face Llama model to custom Llama implementation.
    
    Args:
        hf_model: Hugging Face's LlamaForCausalLM
        custom_model: Custom Transformer implementation
    """
    # # Copy embedding weights
    custom_model.tok_embeddings.weight.data.copy_(hf_model.model.embed_tokens.weight.data)
    
    # Copy final norm and output weights
    custom_model.norm.weight.data.copy_(hf_model.model.norm.weight.data)
    custom_model.output.weight.data.copy_(hf_model.lm_head.weight.data)
    
    # Copy layer weights
    for custom_layer, hf_layer in zip(custom_model.layers, hf_model.model.layers):
        # Copy attention weights
        # HF: q_proj, k_proj, v_proj, o_proj
        # Custom: wq, wk, wv, wo
        custom_layer.attention.wq.weight.data.copy_(hf_layer.self_attn.q_proj.weight.data)
        custom_layer.attention.wk.weight.data.copy_(hf_layer.self_attn.k_proj.weight.data)
        custom_layer.attention.wv.weight.data.copy_(hf_layer.self_attn.v_proj.weight.data)
        custom_layer.attention.wo.weight.data.copy_(hf_layer.self_attn.o_proj.weight.data)
        
        # Copy feed forward weights
        # HF: gate_proj (w1), up_proj (w3), down_proj (w2)
        # Custom: w1, w3, w2
        custom_layer.feed_forward.w1.weight.data.copy_(hf_layer.mlp.gate_proj.weight.data)
        custom_layer.feed_forward.w3.weight.data.copy_(hf_layer.mlp.up_proj.weight.data)
        custom_layer.feed_forward.w2.weight.data.copy_(hf_layer.mlp.down_proj.weight.data)
        
        # Copy normalization weights
        custom_layer.attention_norm.weight.data.copy_(hf_layer.input_layernorm.weight.data)
        custom_layer.ffn_norm.weight.data.copy_(hf_layer.post_attention_layernorm.weight.data)
  
    # return hf_model, custom_model

def copy_weights_to_huggingface_llama(custom_model, hf_model):
    """
    Copy weights from custom Llama implementation to Hugging Face's LlamaForCausalLM.
    
    Args:
        custom_model: Custom Transformer implementation
        hf_model: Hugging Face's LlamaForCausalLM
    """
    # Copy embedding weights
    hf_model.model.embed_tokens.weight.data.copy_(custom_model.tok_embeddings.weight.data)
    
    # Copy final norm and output weights
    hf_model.model.norm.weight.data.copy_(custom_model.norm.weight.data)
    hf_model.lm_head.weight.data.copy_(custom_model.output.weight.data)
    
    # Copy layer weights
    for custom_layer, hf_layer in zip(custom_model.layers, hf_model.model.layers):
        # Copy attention weights
        # Custom: wq, wk, wv, wo
        # HF: q_proj, k_proj, v_proj, o_proj
        hf_layer.self_attn.q_proj.weight.data.copy_(custom_layer.attention.wq.weight.data)
        hf_layer.self_attn.k_proj.weight.data.copy_(custom_layer.attention.wk.weight.data)
        hf_layer.self_attn.v_proj.weight.data.copy_(custom_layer.attention.wv.weight.data)
        hf_layer.self_attn.o_proj.weight.data.copy_(custom_layer.attention.wo.weight.data)
        
        # Copy feed forward weights
        # Custom: w1, w3, w2
        # HF: gate_proj (w1), up_proj (w3), down_proj (w2)
        hf_layer.mlp.gate_proj.weight.data.copy_(custom_layer.feed_forward.w1.weight.data)
        hf_layer.mlp.up_proj.weight.data.copy_(custom_layer.feed_forward.w3.weight.data)
        hf_layer.mlp.down_proj.weight.data.copy_(custom_layer.feed_forward.w2.weight.data)
        
        # Copy normalization weights
        hf_layer.input_layernorm.weight.data.copy_(custom_layer.attention_norm.weight.data)
        hf_layer.post_attention_layernorm.weight.data.copy_(custom_layer.ffn_norm.weight.data)


def compare_tensors(t1, t2, name):
    if torch.allclose(t1, t2, atol=1e-5):
        print(f"✓ {name} weights match")
    else:
        print(f"✗ {name} weights don't match")
        print(f"Max difference: {(t1 - t2).abs().max().item()}")

def get_all_layers_activations(model, input_ids):
    """Helper function to get all layers' activations."""
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
    
def generate(my_model, my_tokenizer, hf_model, hf_tokenizer,prompt):
    temperature = 0.6
    top_p = 0.9
    max_seq_len = 128
    max_gen_len = 64
    max_batch_size = 4
    # Output from my llama
    my_result = my_model.text_completion(
                prompt,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
                )
    
    # Output from HF llama
     # Hugging Face model generation
    # terminators = [
    #     hf_tokenizer.eos_token_id,
    #     hf_tokenizer.convert_tokens_to_ids("<|eot_id|>")
    # ]
    if hf_tokenizer.pad_token is None:
            hf_tokenizer.pad_token = hf_tokenizer.eos_token
    hf_inputs = hf_tokenizer(prompt, return_tensors="pt", add_special_tokens=True, padding = True)
    hf_input_ids = hf_inputs.input_ids
    attention_mask = hf_inputs["attention_mask"]

    hf_result = hf_model.generate(
        hf_input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_gen_len,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        # eos_token_id=terminators,
        pad_token_id=hf_tokenizer.eos_token_id
    )
    hf_result = hf_tokenizer.decode(hf_result[0], skip_special_tokens=True)
    return my_result, hf_result

# Compare weight matrices' rows
compare_weights_matrices_rows(hf_model, my_model)

################# Copy HF to Meta
# copy_weights_to_custom_llama(hf_model, my_model)

################# Copy Meta to HF
# copy_weights_to_huggingface_llama(my_model, hf_model)

# Check embeddings
compare_tensors(
    hf_model.model.embed_tokens.weight,
    my_model.tok_embeddings.weight,
    "Embeddings"
)

# Check first layer attention weights
compare_tensors(
    hf_model.model.layers[0].self_attn.q_proj.weight,
    my_model.layers[0].attention.wq.weight,
    "First layer Q projection"
)

# Check last layer feed forward weights
compare_tensors(
    hf_model.model.layers[-1].mlp.down_proj.weight,
    my_model.layers[-1].feed_forward.w2.weight,
    "Last layer feed forward down projection"
)

# Check output weights
compare_tensors(
    hf_model.lm_head.weight,
    my_model.output.weight,
    "Output weights"
)


# Compare activations
prompt = "The theory of relativity states that"
my_model.return_activation = True  # Enable returning activation

my_input_ids = torch.tensor([my_tokenizer.encode(prompt, bos=True, eos=False)]).cuda()
hf_input_ids = hf_tokenizer.encode(prompt, return_tensors="pt").cuda()
print(f"\nCheck the input tokens the same -- {my_input_ids==hf_input_ids}")

# Get activations for all layers
print("\nTrying to get activations for all layers")

start_time = time.time()
with torch.no_grad():
    my_activations_list = get_all_layers_activations(my_model, hf_input_ids) ###### Replace these two lines with models you want to compare
    hf_activations_list = get_all_layers_activations(hf_model, hf_input_ids)
end_time = time.time()

# Compare activations
print(f"\nMy model activations shape: {len(my_activations_list)}")
print(f"\nHugging Face model activations shape: {len(hf_activations_list)}")

mse_list = np.zeros(len(my_activations_list))
mae_list = np.zeros(len(my_activations_list))
cos_sims = np.zeros(len(my_activations_list))

for i, (my_activations,hf_activations) in enumerate(zip(my_activations_list, hf_activations_list)):
    if my_activations.shape == hf_activations.shape:
        mse = torch.mean((my_activations - hf_activations) ** 2)
        mse_list[i] = mse
        
        mae = torch.mean(torch.abs(my_activations - hf_activations))
        mae_list[i] = mae
        
        cosine_similarity = torch.nn.functional.cosine_similarity(my_activations.flatten(), hf_activations.flatten(), dim=0)
        cos_sims[i] = cosine_similarity
    else:
        print(f"Activation shapes do not match at layer {i}. Cannot compute similarity.")


print("MSE, MAE, COSINE")
np.set_printoptions(precision=5, suppress=True)
print(np.vstack((mse_list,mae_list,cos_sims)).T)

print(f"It took {end_time-start_time:.2f} seconds to compare activations")

# Get outputs
print("Trying to generate")
start_time = time.time()
with torch.no_grad():
    prompt = [prompt]
    my_outputs, hf_outputs = generate(my_llama,my_tokenizer, hf_model, hf_tokenizer, prompt)
    # my_outputs, new_hf_outputs = generate(my_llama,my_tokenizer, new_hf_model, hf_tokenizer, prompt)
    print(f"My output:\n{prompt[0] + my_outputs[0]['generation']}")
    print(f"HF output:\n{hf_outputs}")
    # print(f"New HF output:\n{new_hf_outputs}")
end_time = time.time()
print(f"It took {end_time-start_time:.2f} seconds to generate and compare outputs")