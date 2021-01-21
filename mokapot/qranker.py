"""
A PyTorch model that emulates the Q-ranker method.

This code has the additional dependency of PyTorch.

The :code:`forward()` method here contains two sections:
    1. Direct classification using a sigmoid loss function.
    2. Ranking optimization using the Q-ranker algorithm.
"""
import torch
from skorch import NeuralNet

from . import utils


class QRankerNet(NeuralNet):
    """
    A neural network that implements the Q-Ranker algorithm.

    Parameters
    ----------
    train_fdr: float
        The FDR threshold for which to optimize.
    layers: tuple, optional
        The dimensions of the hidden layers to use. Use :code:`None` for a
        linear model.
    activation: {"tanh", "relu"}, optional
        The non-linear activation function to use.
    weight_decay: float, optional
        Strength of weight decay regularization.
    """

    def __init__(
        self, train_fdr=0.01, layers=(5,), activation="tanh", weight_decay=0
    ):
        """Initialize a QRankerNet"""
        self.weight_decay = weight_decay
        self.train_fdr = train_fdr

        super().__init__(
            MLP,
            module__layers=layers,
            module__activation=activation,
            criterion=torch.nn.Sigmoid(),
            optimizer=torch.optim.Adam,
        )

    def fit(self, X, y):
        """Fit the model using the Q-Ranker algorithm"""
        self.set_params(module__num_features=X.shape[1])

        # Direct classification
        super().fit(X, y)

        # Ranking optimization
        # 1. Set new criterion and add weight decay.
        self.set_params(
            criterion=torch.nn.MarginRankingLoss(),
            optimizer__weight_decay=self.weight_decay,
        )

        # 2. Subset data according to FDR and eta heuristic.


class MLP(torch.nn.Module):
    """
    A multi-layer perceptron neural network

    Parameters
    ----------
    num_features: int
        The number of input features
    layers: tuple, optional
        The dimensions of hidden layers to use. Use :code:`None` for a linear
        model.
    activation: {"tanh", "relu"}, optional
        The non-linear activation function to use.
    """

    def __init__(self, num_features, layers=(5,), activation="tanh"):
        """Initialize a QRankerNet"""
        super().__init__()

        # Build the network:
        if activation == "tanh":
            act_fun = torch.nn.Tanh
        elif activation == "relu":
            act_fun = torch.nn.ReLU
        else:
            raise ValueError("Unrecognized value for 'activation'")

        if num_features <= 0 or not isinstance(num_features, int):
            raise ValueError("'num_features' must be a positive integer.")

        layers = list(utils.tuplize(layers))
        if not all(layers) or not all(isinstance(l, int) for l in layers):
            raise ValueError(
                "'hidden_dims' must be a list of positive " "integers."
            )

        in_layers = [num_features] + layers
        out_layers = layers + [1]
        for idx, in_layer in enumerate(in_layers):
            self.add_module(
                "linear_{}".format(idx),
                torch.nn.Linear(in_layer, out_layers[idx]),
            )

            # Don't add leaky ReLU to output layer:
            if idx < len(in_layers) - 1:
                self.add_module("{}_{}".format(activation, idx), act_fun())

    def forward(self, X):
        """
        Run an example through the model

        Parameters
        ----------
        X : torch.Tensor
            A tensor to run through the model.
        Returns
        -------
        torch.Tensor
            The model predictions.
        """
        for idx, layer in enumerate(self._modules.values()):
            if idx < len(self._modules) - 1:
                X = layer(X)
            else:
                return layer(X)


# Functions -------------------------------------------------------------------
