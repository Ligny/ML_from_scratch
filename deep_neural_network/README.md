# Deep Neural Network (DNN) Implementation

Cette implémentation d'un réseau neuronal profond (DNN) en Python utilise Numpy pour les calculs de base. Voici une explication détaillée de chaque composant du DNN.

## 1. Initialisation et Architecture

Nous commençons par initialiser notre DNN avec une architecture définie et une fonction d'activation.

```python
class DeepNeuralNetwork():
    def __init__(self, sizes, activation='sigmoid'):
        self.sizes = sizes
        
        # Choose activation function
        if activation == 'relu':
            self.activation = self.relu
        elif activation == 'sigmoid':
            self.activation = self.sigmoid
        else:
            raise ValueError("Activation function is currently not support, please use 'relu' or 'sigmoid' instead.")
        
        # Save all weights
        self.params = self.initialize()
        # Save all intermediate values, i.e. activations
        self.cache = {}
```

## 2. Activation des Neurones

Les fonctions d'activation comme Sigmoid et ReLU transforment les entrées non linéaires pour chaque neurone.

```python
def relu(self, x, derivative=False):
        if derivative:
            x = np.where(x < 0, 0, x)
            x = np.where(x >= 0, 1, x)
            return x
        return np.maximum(0, x)

    def sigmoid(self, x, derivative=False):
        if derivative:
            return (np.exp(-x))/((np.exp(-x)+1)**2)
        return 1/(1 + np.exp(-x))

    def softmax(self, x):
        exps = np.exp(x - x.max())
        return exps / np.sum(exps, axis=0)
```

## 3. Initialisation des Poids et Biais

Les poids et les biais sont initialisés aléatoirement pour chaque couche.

```python
    def initialize(self):
        input_layer = self.sizes[0]
        hidden_layer = self.sizes[1]
        output_layer = self.sizes[2]
        
        params = {
            "W1": np.random.randn(hidden_layer, input_layer) * np.sqrt(1./input_layer),
            "b1": np.zeros((hidden_layer, 1)) * np.sqrt(1./input_layer),
            "W2": np.random.randn(output_layer, hidden_layer) * np.sqrt(1./hidden_layer),
            "b2": np.zeros((output_layer, 1)) * np.sqrt(1./hidden_layer)
        }
        return params
```

## 4. Propagation Avant (Feedforward)

Les données sont transmises à travers le réseau couche par couche.

```python
    def feed_forward(self, x):
        self.cache["X"] = x
        self.cache["Z1"] = np.matmul(self.params["W1"], self.cache["X"].T) + self.params["b1"]
        self.cache["A1"] = self.activation(self.cache["Z1"])
        self.cache["Z2"] = np.matmul(self.params["W2"], self.cache["A1"]) + self.params["b2"]
        self.cache["A2"] = self.softmax(self.cache["Z2"])
        return self.cache["A2"]
```

## 5. Rétropropagation (Backpropagation)

L'algorithme de rétropropagation calcule les gradients nécessaires pour mettre à jour les poids.

```python
    def back_propagate(self, y, output):
        current_batch_size = y.shape[0]
        
        dZ2 = output - y.T
        dW2 = (1./current_batch_size) * np.matmul(dZ2, self.cache["A1"].T)
        db2 = (1./current_batch_size) * np.sum(dZ2, axis=1, keepdims=True)

        dA1 = np.matmul(self.params["W2"].T, dZ2)
        dZ1 = dA1 * self.activation(self.cache["Z1"], derivative=True)
        dW1 = (1./current_batch_size) * np.matmul(dZ1, self.cache["X"])
        db1 = (1./current_batch_size) * np.sum(dZ1, axis=1, keepdims=True)

        self.grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
        return self.grads
```

## 6. Fonction de Perte

La fonction de perte, ici l'entropie croisée, mesure la différence entre les prédictions et les vraies valeurs.

```python
    def cross_entropy_loss(self, y, output):
        l_sum = np.sum(np.multiply(y.T, np.log(output)))
        m = y.shape[0]
        l = -(1./m) * l_sum
        return l
```

## 7. Entraînement

Le réseau est entraîné sur plusieurs époques en ajustant les poids et les biais.

```python
    def train(self, x_train, y_train, x_test, y_test, epochs=10, 
              batch_size=64, optimizer='momentum', l_rate=0.1, beta=.9):
        self.epochs = epochs
        self.batch_size = batch_size
        num_batches = -(-x_train.shape[0] // self.batch_size)
        
        self.optimizer = optimizer
        if self.optimizer == 'momentum':
            self.momemtum_opt = self.initialize_momemtum_optimizer()
        
        train_accuracies = []
        test_accuracies = []

        start_time = time.time()
        template = "Epoch {}: {:.2f}s, train acc={:.2f}, train loss={:.2f}, test acc={:.2f}, test loss={:.2f}"
        
        for i in range(self.epochs):
            permutation = np.random.permutation(x_train.shape[0])
            x_train_shuffled = x_train[permutation]
            y_train_shuffled = y_train[permutation]

            for j in range(num_batches):
                begin = j * self.batch_size
                end = min(begin + self.batch_size, x_train.shape[0]-1)
                x = x_train_shuffled[begin:end]
                y = y_train_shuffled[begin:end]
                
                output = self.feed_forward(x)
                grad = self.back_propagate(y, output)
                self.optimize(l_rate=l_rate, beta=beta)

            output = self.feed_forward(x_train)
            train_acc = self.accuracy(y_train, output)
            train_loss = self.cross_entropy_loss(y_train, output)

            output = self.feed_forward(x_test)
            test_acc = self.accuracy(y_test, output)
            test_loss = self.cross_entropy_loss(y_test, output)

            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)

            print(template.format(i+1, time.time()-start_time, train_acc, train_loss, test_acc, test_loss))
        
        return train_accuracies, test_accuracies
```

## 8. Optimisation

Les paramètres du réseau sont mis à jour en utilisant SGD ou l'optimisation par moment.

```python
    def optimize(self, l_rate=0.1, beta=.9):
        if self.optimizer == "sgd":
            for key in self.params:
                self.params[key] = self.params[key] - l_rate * self.grads[key]
        elif self.optimizer == "momentum":
            for key in self.params:
                self.momemtum_opt[key] = (beta * self.momemtum_opt[key] + (1. - beta) * self.grads[key])
                self.params[key] = self.params[key] - l_rate * self.momemtum_opt[key]
        else:
            raise ValueError("Optimizer is currently not support, please use 'sgd' or 'momentum' instead.")
```

## Exemple d'utilisation

Pour utiliser le réseau, initialisez-le avec les tailles de couche et la fonction d'activation souhaitées, puis entraînez-le avec les données d'entraînement et de test.

```python
dnn = DeepNeuralNetwork(sizes=[784, 64, 10], activation='sigmoid')
train_acc, test_acc_dnn = dnn.train(x_train, y_train, x_test, y_test, epochs=10, optimizer='momentum', l_rate=0.1, beta=.9)
```