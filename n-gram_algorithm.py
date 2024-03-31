import nltk
from nltk.util import bigrams
from collections import Counter

# Se receber um erro, é porque precisa chamar ao menos uma vez a linha comentada abaixo.
# nltk.download('punkt')

# Exemplo de texto
text = "Este é um exemplo simples de um modelo n-gram. N-gram modelos são úteis em processamento de linguagem natural."

# Tokenização do texto
tokens = nltk.word_tokenize(text)

# Criação de bigrams
bigram_list = list(bigrams(tokens))

# Contagem da frequência de cada bigram
bigram_counts = Counter(bigram_list)

# Exibir os bigrams mais comuns
print(bigram_counts.most_common(5))
