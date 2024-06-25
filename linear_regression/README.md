# Linear Regression from Scratch with Boston House Dataset

This project implements a simple linear regression model from scratch using the Boston House dataset. The model is built using only NumPy and is trained using gradient descent.

## Class: LinearRegression

### Methods

#### `__init__(self)`
Initializes the LinearRegression model with weights and bias set to `None`.

#### `gradient_descent(self, X, y, epochs, learning_rate=0.01)`
Trains the linear regression model using gradient descent.

- **Parameters**:
  - `X`: Features of the dataset.
  - `y`: Target variable.
  - `epochs`: Number of iterations for gradient descent.
  - `learning_rate`: Learning rate for gradient descent (default is 0.01).

- **Returns**:
  - `self.weights`: The final weights after training.
  - `self.bias`: The final bias after training.
  - `epochCosts`: List of costs at each epoch for graphing the convergence.

- **Example**:
    ```python
    model = LinearRegression()
    weights, bias, costs = model.gradient_descent(X_train, y_train, epochs=1000, learning_rate=0.01)
    ```

#### `predict(self, X)`
Predicts the target variable for given features using the trained model.

- **Parameters**:
  - `X`: Features of the dataset.

- **Returns**:
  - Predicted values for the given features.

- **Example**:
    ```python
    predictions = model.predict(X_test)
    ```

## Usage

### 1. Importing Necessary Libraries
Make sure you have the necessary libraries installed. If not, install them using `pip`.

```bash
pip install numpy scikit-learn
```

