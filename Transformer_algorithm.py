from transformers import TFMarianMTModel, MarianTokenizer

# Carregar o modelo e o tokenizador TensorFlow
model_name = 'Helsinki-NLP/opus-mt-en-es'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = TFMarianMTModel.from_pretrained(model_name)

# Texto para tradução
text = "This is an example of using a Transformer model for translation."

# Tokenização do texto
tokens = tokenizer(text, return_tensors='tf', truncation=True, max_length=512)

# Tradução
translated = model.generate(**tokens)

# Decodificar a saída
translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

print(f"Original: {text}")
print(f"Traduzido: {translated_text}")
