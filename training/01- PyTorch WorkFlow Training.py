import matplotlib.pyplot as plt
import torch
import matplotlib
from torch import nn

# from torch import nn. !!!nn contains all of PyTorch's building blocks for neural networks

# 1.DATA(prepare and load)
# Data can be almost everything in ML
# ML is a game of two parts:
# 1.Get data into a numerical representation
# 2.Build a model to learn patterns in that numerical representation

# To showcase this,let's create some data using the linear regression formula
# We'll use a linear regression formula to make a straight line with known parameters(a parameter is something that a model learns)

# Create *known* parameters
weight = 0.7
bias = 0.3

# Create
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
Y = weight * X + bias

print(X[:10], Y[:10])
print(len(X), len(Y))

###Splitting data into training and data sets(one of the most important concepts in ML in general)
# Training set-The model learns from this data
# Validation set-The model gets tuned on this data
# Testing set-The model gets evaluated on this data to test what it has learned

# Create a train/test split
train_split = int(0.8 * len(X))
X_train, Y_train = X[:train_split], Y[:train_split]
X_test, Y_test = X[train_split:], Y[train_split:]
print(len(X_train), len(Y_train), len(X_test), len(Y_test))


def plot_predictions(train_data=X_train,
                     train_label=Y_train,
                     test_data=X_test,
                     test_label=Y_test,
                     predictions=None):
    """
    Plots training data,test data and compares predictions
    """
    plt.figure(figsize=(10, 7))
    # Plot training data in blue
    plt.scatter(train_data, train_label, c="b", s=4, label="Training data")

    # Plot testing data in green
    plt.scatter(test_data, test_label, c="g", s=4, label="Testing data")

    if predictions is not None:
        # Plot red dots for predictions
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions", marker='o')
    plt.legend(prop={"size": 14})
    plt.show()


plot_predictions()


class LinearRegressionModel(nn.Module):  # <- almost everything in PyTorch inherits from nn.Module
    # Subclass nn.Module(this contains all the building blocks for neural networks)
    def __init__(self):
        super().__init__()
        # Initialize model parameters to be used in various
        # computations(these could be different layers from torch.nn,single parameters,hard-coded values or functions
        self.weights = nn.Parameter(torch.rand(1,
                                               requires_grad=True,
                                               dtype=torch.float))
        self.bias = nn.Parameter(torch.rand(1,
                                            requires_grad=True,
                                            dtype=torch.float))

        # requires_grad = True means the PyTorch will track the gradients of this specific parameter for use with
        # torch.autograd and gradient descent(for many torch.nn modules,requires_grad=True is set by default)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias

        ##
        # What our model does:
        # Start with random values(weight&bias)
        # Look at training data and adjust the random values to better represent (or get closer to) the ideal
        # values(the weight&bias we used to create the data)
        # How does it do so?
        # Through two main algorithms:
        # 1.Gradient descent
        # 2.Backpropagation


# PyTorch model building essentials
# torch.nn-contains all the buildings for computational graphs
# (another word for neural network.a neural network can be considered a computational graph)
# torch.nn.Parameter-what parameters should our model try and learn,often a PyTorch
# layer from torch.nn will set these for use
# torch.nn.Module-the base class for all neural network modules,if you subclass it,you should overwrite forward()
# torch.optim-this where the optimizers in PyTorch live,they will help with gradient descent
# def forward()-All nn.Modules subclasses require you to overwrite
# forward(),this method defines what happens in the forward computation


# torch.utils.data.DataSet-represents a map between key(label) and sample
# (features) pairs of your data.Such as images and their associated data
# torch.utils.data.DataLoader-creates a python iterable over a torch dataset(allows you to iterate over your data)
torch.manual_seed(42)
model_0 = LinearRegressionModel()
print(list(model_0.parameters()))
print(model_0.state_dict())

# Making prediction using 'torch.inference_mode()'
# When we pass data through our model, it's going to run it through the 'forward()' method
# Make predictions with model
with torch.inference_mode():  # Disable gradient tracking
    y_preds = model_0(X_test)
print(y_preds)
plot_predictions(predictions=y_preds)
print(model_0.state_dict())

# Setup a loss function
loss_fn = nn.L1Loss()

# Setup an optimizer (stochastic gradient descent)
optimizer = torch.optim.SGD(model_0.parameters(), lr=0.01)

# A couple of things we need in a training loop:
# 0.Loop through the data
epochs = 1

# 0.Loop through the data
for epoch in range(epochs):
    # Set the mode to training model
    model_0.train()  # train model in PyTorch sets all parameters that require gradients to require gradients
    # model.train() tells your model that you are training the model. training and evaluation.
    # For instance, in training mode, BatchNorm updates a moving average on each new batch;
    # whereas, for evaluation mode, these updates are frozen.
    # model_0.eval()  # turns off gradient tracking

    # 1.Forward pass
    y_pred = model_0(X_train)
    # 2.Calculate the loss
    loss = loss_fn(y_pred, Y_train)
    print(f"Loss:{loss}")
    # 3.Optimizer zero grad
    optimizer = torch.optim.SGD(model_0.parameters(), lr=0.01)

    epochs = 1  # Increase the number of epochs

    for epoch in range(epochs):
        model_0.train()
        y_new_preds = model_0(X_train)
        loss = loss_fn(y_new_preds, Y_train)
        optimizer.zero_grad()
        # 4.Perform backpropagation on the loss with respect to the parameters of the model
        loss.backward()
        # 5.Step the optimizer(perform gradient descent)
        optimizer.step()  # by default how the optimizer changes will accumulate through the loop.
    # we have to zero them above in step 3 for the next iteration of the loop

# Evaluation after training
model_0.eval()
with torch.inference_mode():
    y_preds = model_0(X_test)

plot_predictions(predictions=y_preds)
print(model_0.state_dict())