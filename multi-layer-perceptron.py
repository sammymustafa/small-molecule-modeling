from setup import * 

def relu(x):
    return torch.nn.functional.relu(x)

def sigmoid(x):
    """Sigmoid activation of the input x."""
    return torch.exp(x) / (1 + torch.exp(x))

def predict(x, W1, b1, W2, b2):
    """Returns y_pred given the input and learned parameters."""
    layer1 = relu(torch.mm(x, W1) + b1)
    y_pred = sigmoid(torch.mm(layer1, W2) + b2)
    return y_pred

def loss(y_pred, y_true):
    """Returns the cross-entropy loss given the prediction and target."""
    epsilon = 1e-15
    loss_per_sample = -((y_true * torch.log(y_pred + epsilon)) + ((1 - y_true) * torch.log(1 - y_pred + epsilon)))
    return loss_per_sample

def gradient_descent_solver(x, y_true):
    # Initialize weights
    random = np.random.RandomState(1)
    W1 = random.randn(2, 100) * 0.01
    W2 = random.randn(100, 1) * 0.01
    W1 = torch.nn.Parameter(torch.tensor(W1).float())
    b1 = torch.nn.Parameter(torch.zeros((100,)))
    W2 = torch.nn.Parameter(torch.tensor(W2).float())
    b2 = torch.nn.Parameter(torch.zeros((1,)))
    alpha = 0.1
    num_steps = 1000

    # Perform steps of gradient descent
    x = torch.tensor(x).float()
    y_true = torch.tensor(y_true).float()
    optimizer = torch.optim.SGD([W1, b1, W2, b2], alpha)

    y_pred = predict(x, W1, b1, W2, b2)
    L_start = loss(y_pred, y_true).mean()
    accuracy_start = ((y_pred > 0.5) == y_true).float().mean()

    for _ in range(num_steps):
        optimizer.zero_grad()
        y_pred = predict(x, W1, b1, W2, b2)
        L = loss(y_pred, y_true).mean()
        L.backward()
        optimizer.step()
        accuracy = ((y_pred > 0.5) == y_true).float().mean()

    print("Start loss: ", L_start.item())
    print("Final loss: ", L.item())

    print("Start accuracy: ", accuracy_start.item())
    print("Final accuracy: ", accuracy.item())

# Run training
print("Dataset 1")
gradient_descent_solver(X1, Y1)

print("\nDataset 2")
gradient_descent_solver(X2, Y2)

# Results
# Introducing the ReLu activator function allowed us to account for the linear clustering in the first dataset and the non-linear clustering in the second dataset as they both have accuracies over 98%.