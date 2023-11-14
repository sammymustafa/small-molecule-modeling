from setup import * 

def sigmoid(x):
    """Sigmoid activation of the input x."""
    return np.exp(x) / (1 + np.exp(x))

def predict(x, W, b):
    """Returns y_pred given the input and learned parameters."""
    return sigmoid(np.dot(x, W) + b)

def loss(y_pred, y_true):
    """Returns the cross-entropy loss given the prediction and target."""
    epsilon = 1e-15
    loss_per_sample = -((y_true * np.log(y_pred + epsilon)) + ((1 - y_true) * np.log(1 - y_pred + epsilon)))
    return loss_per_sample

def dLossdW(y_pred, y_true, x):
    """Comptues the derivative of the loss with respect to W."""
    gradients = np.array([(y_pred[i] - y_true[i]) * x[i] for i in range(y_pred.shape[0])])

    return gradients

def dLossdb(y_pred, y_true):
    """Comptues the derivative of the loss with respect to b."""
    return y_pred - y_true


def gradient_descent_solver(x, y_true):
    # Initialize weights
    W = np.array([0.0, 0.0])[:, None]
    b = np.array([0])
    alpha = 1.0
    num_steps = 1000

    # Perform steps of gradient descent
    y_pred = predict(x, W, b)
    L_start = loss(y_pred, y_true).mean()
    accuracy_start = ((y_pred > 0.5) == y_true).mean()

    for _ in range(num_steps):
        y_pred = predict(x, W, b)
        L = loss(y_pred, y_true).mean()
        accuracy = ((y_pred > 0.5) == y_true).mean()

        dW = dLossdW(y_pred, y_true, x)
        db = dLossdb(y_pred, y_true)
        W = W - alpha * dW.mean(axis=0)[:, None]
        b = b - alpha * db.mean(axis=0)

    print("Start loss: ", L_start)
    print("Final loss: ", L)

    print("Start accuracy: ", accuracy_start)
    print("Final accuracy: ", accuracy)
    return W, b


def plot_results(x, y_true, W, b):
    plt.figure()
    plt.scatter(x[:, 0], x[:, 1], c=y_true)

    x1 = np.linspace(-10, 10)
    x2 = 0 * x1 - 0
    plt.plot(x1, x2, c="b", label="Starting boundary")

    x1 = np.linspace(-10, 10)
    x2 = -W[0] / W[1] * x1 - b / W[1]
    plt.plot(x1, x2, c="r", label="Final boundary")

    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.legend()
    plt.show()

# Run training
W1, b1 = gradient_descent_solver(X1, Y1)
plot_results(X1, Y1, W1, b1)

W2, b2 = gradient_descent_solver(X2, Y2)
plot_results(X2, Y2, W2, b2)

# Results
# The model achieved near complete accuracy with the final boundary on the first dataset and almost exactly the same accuracy on the second dataset. The model is not appropriate for the second dataset because it requires non-linear clustering in which a line is not an accurate or useful separable tool for clustering.