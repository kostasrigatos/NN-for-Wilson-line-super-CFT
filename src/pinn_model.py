import torch
import torch.nn as nn
import torch.nn.functional as F

class WilsonNetwork(nn.Module):
    r"""
         A 4-layer neural network for parameter-dependent predictions.

         Parameters
         ----------
         hidden_dim: int
            The number of hidden units inside each of the fully connected hidden layers.

         Attributes
         ----------
         layer1: nn.Linear
             The input layer of the network projecting from 1 feature to hidden_dim features.
         layer2: nn.Linear
              The first hidden layer of the network projecting from hidden_dim features to hidden_dim features.
         layer3: nn.Linear
               The second hidden layer of the network projecting from hidden_dim features to hidden_dim features.
         layer4: nn.Linear
               The output layer of the network projecting from hidden_dim features to 10 outputs.
     """
    def __init__(self, hidden_dim: int = 64, n_hidden_layers: int = 3):
        super().__init__()

        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(1, hidden_dim))

        for idx in range(n_hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.output_layer = nn.Linear(hidden_dim, 10)

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        r"""
             Executes the forward pass of the neural network.

             Parameters
             ----------
             g: torch.Tensor
                A tensor of shape (batch_size, 1) that represents the input parameter; the coupling constant.

             Returns
             -------
             torch.Tensor
               A tensor of shape (batch_size, 10) that contains the output of the neural network; the non-negative predicted
               values across the 10 elements.
         """
        x = g
        for layer in self.hidden_layers:
            x = torch.tanh(layer(x))

        out = F.softplus(self.output_layer(x))

        return out

if __name__ == "__main__":

    model = WilsonNetwork()

    g_single = torch.tensor([[0.5]], dtype=torch.float32)
    result_single = model(g_single)
    print(f"The shape of the result for a single input: {result_single.shape}")
    print(f"Are they all non-negative? {torch.all(result_single >= 0).item()}")

    g_batch = torch.tensor([[0.1], [0.2], [0.3], [0.4], [0.5], [1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
    result_batch = model(g_batch)
    print(f"The shape of the result for the batch input: {result_batch.shape}")
    print(f"Are they all non-negative? {torch.all(result_batch >= 0).item()}")