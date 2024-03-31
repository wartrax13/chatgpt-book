from transformers import pipeline

# Carregar o pipeline de análise de sentimentos
sentiment_pipeline = pipeline('sentiment-analysis')

# Texto para análise de sentimentos
text = "My heart is crying for not having you by my side."

# Realizar análise de sentimentos
result = sentiment_pipeline(text)

print(f"Texto: {text}")
print(f"Sentimento: {result[0]['label']}, Score: {result[0]['score']:.2f}")
