
from transformers import AutoTokenizer

def download_tokenizer(model_path = "meta-llama/Llama-2-7b-chat-hf", token = ""):
    print("instantiating pretrained tokenizer")
    AutoTokenizer.from_pretrained(model_path)

download_tokenizer()