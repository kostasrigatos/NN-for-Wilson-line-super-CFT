import torch
import torch.nn as nn
import torch.nn.functional as F

class WilsonNetwork(nn.Module):

    def __init__(self, hidden_dim = 64):
        super().__init__()

        self.layer1 = nn.Linear(1, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer4 = nn.Linear(hidden_dim, 10)

    def forward(self, g):
        x = torch.tanh(self.layer1(g))
        x = torch.tanh(self.layer2(x))
        x = torch.tanh(self.layer3(x))

        out = F.softplus(self.layer4(x))

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