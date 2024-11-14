from transformers import LlamaTokenizer as HFTokenizer
from transformers import AutoTokenizer

model_path = "/home/huang717/DRAGN/IRM/injectable-alignment-model/src/local_tokenizer"

tokenizer = AutoTokenizer.from_pretrained(model_path)

# Sample text to tokenize
text = "Hello! This is a test string with some numbers 123 and special characters @#$."

# 1. Encode the string
encoded = tokenizer.encode(text)

# 2. Print the token IDs
print("Encoded token IDs:")
print(encoded)
print(f"Length: {len(encoded)}")

# 3. Decode back to string
decoded = tokenizer.decode(encoded)

# Print the results
print("\nOriginal text:")
print(text)
print("\nDecoded text:")
print(decoded)

# Optional: Print token-by-token breakdown
print("\nToken-by-token breakdown:")
tokens = tokenizer.tokenize(text)
for i, token in enumerate(tokens):
    print(f"Token {i}: '{token}'")