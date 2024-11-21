import torch
import os
from transformers import pipeline
from huggingface_hub import login

os.environ['HF_TOKEN'] = "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"
huggingface_token = os.environ.get('HF_TOKEN')
login(token=huggingface_token)

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

llama_31 = "meta-llama/Llama-3.1-8B-Instruct" # <-- llama 3.1
llama_32 = "meta-llama/Llama-3.2-3B-Instruct" # <-- llama 3.2

# prompt = [
#     {"role": "system", "content": "You are a helpful assistant, that responds as a pirate."},
#     {"role": "user", "content": "What's Deep Learning?"},
# ]
prompt = [
    {"role": "system", "content": "You are a helpful assistant, that responds as a pirate."},
    {"role": "user", "content": "How do I finetune a llama model?"},
]

generator = pipeline(model=llama_32, device=device, torch_dtype=torch.bfloat16)
generation = generator(
    prompt,
    do_sample=False,
    temperature=1.0,
    top_p=1,
    max_new_tokens=50
)

print(f"Generation: {generation[0]['generated_text']}")

# Generation:
# [
#   {'role': 'system', 'content': 'You are a helpful assistant, that responds as a pirate.'},
#   {'role': 'user', 'content': "What's Deep Learning?"},
#   {'role': 'assistant', 'content': "Yer lookin' fer a treasure trove o'
#             knowledge on Deep Learnin', eh? Alright then, listen close and
#             I'll tell ye about it.\n\nDeep Learnin' be a type o' machine
#             learnin' that uses neural networks"}
# ]