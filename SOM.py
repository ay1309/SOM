import numpy as np
import matplotlib.pyplot as plt

animales = ['paloma', 'gallina', 'pato', 'ganso', 'buho', 'halcon', 'aguila', 'zorro', 
                'perro', 'lobo', 'gato', 'tigre', 'leon', 'caballo', 'zebra', 'vaca']

caracteristicas = np.array([
    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1],  
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0],  
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],  
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],  
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],  
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],  
    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],  
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],  
    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]   
]).T

def normalizar(data):
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val)

caracteristicasNormalizadas = normalizar(caracteristicas)

class SOM:
    def __init__(self, x, y, input_len, learning_rate=0.5, radius=None, iterations=10000):
        self.x = x
        self.y = y
        self.input_len = input_len
        self.tazaAprendizaje= learning_rate
        self.iterations = iterations
        self.radius = radius if radius else max(x, y) / 2
        self.pesos = np.random.random((x, y, input_len))

    def aprendizaje(self, data):
        for i in range(self.iterations):
            rand_i = np.random.randint(len(data))
            sample = data[rand_i]
            bmu = self.busqueda(sample)
            self.actualizarPesos(bmu, sample, i)

    def busqueda(self, sample):
        dist = np.linalg.norm(self.pesos - sample, axis=2)
        busquedaResultado = np.unravel_index(np.argmin(dist), dist.shape)
        return busquedaResultado

    def actualizarPesos(self, busquedaResultado, sample, iteration):
        tazaAprendizaje= self.tazaAprendizaje* (1 - iteration / self.iterations)
        radio = self.radius * (1 - iteration / self.iterations)
        for i in range(self.x):
            for j in range(self.y):
                distanciaBmu = np.linalg.norm(np.array([i, j]) - np.array(busquedaResultado))
                if distanciaBmu <= radio:
                    influence = np.exp(-distanciaBmu / (2 * (radio ** 2)))
                    self.pesos[i, j] += influence * tazaAprendizaje* (sample - self.pesos[i, j])

som = SOM(4, 4, caracteristicasNormalizadas.shape[1], iterations=10000)
som.aprendizaje(caracteristicasNormalizadas)

plt.figure(figsize=(10, 10))
for i in range(4):
    for j in range(4):
        plt.text(j, i, animales[np.argmin(np.linalg.norm(som.pesos[i, j] - caracteristicasNormalizadas, axis=1))],
                ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))

plt.xlim(-0.5, 3.5)
plt.ylim(-0.5, 3.5)
plt.gca().invert_yaxis()
plt.show()

