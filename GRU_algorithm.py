import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense

# Dados de exemplo: "hello"
chars = 'helo'
data = [chars.index('h'), chars.index('e'), chars.index('l'), chars.index('l'), chars.index('o')]
X = np.array(data[:-1]).reshape((1, 4, 1))
y = np.array(data[1:]).reshape((1, 4))

# Normalizando os dados
X = X / float(len(chars))

# Modelo GRU
model = Sequential()
model.add(GRU(10, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(len(chars), activation='softmax'))

# Compilação do modelo
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Treinamento do modelo
model.fit(X, y, epochs=1000, verbose=0)

# Predição
predicted = model.predict(X, verbose=0)
output = [chars[np.argmax(pred)] for pred in predicted[0]]

print("Sequência prevista:", ''.join(output))
