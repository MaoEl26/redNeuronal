import numpy as np

# Descripcion: Con este programa se predice la nota de un estudiante y por ende si aprueba o no el curso; Aprueba con una nota mayor o igual a 70.
# Entradas: Array de arrays, en cada array se anota el porcentaje para cada rubro evaluado de la materia [35, 20, 15, 5, 10, 15] seria un 100
# Salida: Un array con la nota predecida con base en el entrenamiento. Se debe entrenar el programa aproximadamente 20000 veces para obtener una mejor aproximacion.

# X = (Examen 1, Examen 2, Proyecto 1, Proyecto 2, Tareas, Trabajo en clase), y = Nota final (Con base en estas se dice si pasa o no)
xAll = np.array(([30, 12, 9, 5, 8, 11], #75
                 [30, 5, 10, 3, 10, 10],  #68
                 [30, 5, 1, 5, 1, 5],     #47
                 [35, 20, 5, 5, 1, 10],   #76
                 [20, 20, 1, 5, 10, 11],  #67
                 [24, 14, 10, 5, 10, 7],  #70
                 [35, 17, 10, 5, 10, 8],  #85
                 [35, 20, 15, 5, 10, 10]), dtype=float) # 100                
                 
# **Si se agregan mas notas, agregar la nota que da en el array "y", y modificar las cantidades de "X" y "xPredicted".

# Salidas (Si se agregan mas notas en xAll, se pone el resultado aqui, menos el de la ultima porque esa es la que vamos a predecir).
y = np.array(([75], [68], [47],[76],[67],[70],[85]), dtype=float)

# Ajuste

Ltimes = xAll[-1]                                                       # Ontener la lista de notas que queremos.
times = int(sum(Ltimes)*1000)                                           # Esto es para calcular las veces que se va a entrenar 
xAll = xAll/np.amax(xAll, axis=0)                                       # Aqui se ajustan los datos de entrada
y = y/100                                                               # La nota más alta es 100

# Se separan los datos (Solo se que es necesario)
X = np.split(xAll, [7])[0]                                              # Datos de entrenamientp
xPredicted = np.split(xAll, [7])[1]                                     # Datos de prueba


class Neural_Network(object):
    def __init__(self):
        #Parametros
        self.inputSize = 6                                              #Largo del vector de entrada
        self.outputSize = 1                                             #Largo del vector de salida
        self.hiddenSize = 3                                             #Largo de la red (Por niveles: entradas, medio y salida) 

        #Pesos
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)      # peso de la matriz de la entrada a la capa media.
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)     # peso de la matriz de la capa media a la de salida

    def forward(self, X):
        #forward propagation por la NN
        self.z = np.dot(X, self.W1)                                     # Producto de la matriz de entrda X y el primer ser de pesos W1
        self.z2 = self.sigmoid(self.z)                                  # Funcion de activacion
        self.z3 = np.dot(self.z2, self.W2)                              # producto de la capa edia (z2) y el segundo set de pesos W2
        o = self.sigmoid(self.z3)                                       # Funcion de activacion final
        return o

    # Funcion de activacion
    def sigmoid(self, s):
        return 1/(1+np.exp(-s))

    # Funcion sigmoide
    def sigmoidPrime(self, s):
        return s * (1 - s)

    # backward propagate por la NN
    def backward(self, X, y, o):
        self.o_error = y - o                                            # error en el dato de salida
        self.o_delta = self.o_error*self.sigmoidPrime(o)                # Aplicar derivada de sigmoide al error
    
        self.z2_error = self.o_delta.dot(self.W2.T)                     # error z2: cantidad de error que la capa media aporto al error final
        self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2)        # Aplicar derivada de sigmoide al error z2

        self.W1 += X.T.dot(self.z2_delta)                               # Ajuste de peso del primer set (entrada --> media)
        self.W2 += self.z2.T.dot(self.o_delta)                          # Ajuste de peso del segundo set (media --> salida)

    def train(self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)

    def predict(self):
        print ("Basado en los entrenamientos los datos son: ");
        print ("Porcentajes del estudiante (a escala): \n" + str(xPredicted));
        print ("Nota: \n" + str(self.forward(xPredicted)));
        NT = self.forward(xPredicted)
        if (NT > 0.69):
          print ("¡El estudiante aprueba el curso!")
        else:
          print ("¡El estudiante no aprueba el curso!")

NN = Neural_Network()          
for i in range(times):                                                  # Entrena la NN n veces
  print ("# " + str(i) + "\n")
  print ("Notas de entrada (A escala): \n" + str(X))
  print ("Nota actual: \n" + str(y))
  print ("Nota predecida: \n" + str(NN.forward(X)))
  print ("Perdida: \n" + str(np.mean(np.square(y - NN.forward(X)))))    # Suma media de perdida al cuadrado
  print ("\n")
  NN.train(X, y)

NN.predict()

