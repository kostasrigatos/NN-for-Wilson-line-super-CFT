import torch
from pinn_model import WilsonNetwork

def test_wilson_network_default_matches_original():
    torch.manual_seed(42)
    model_new = WilsonNetwork().double()
    g = torch.tensor([[1.5]], dtype=torch.float64)
    out = model_new(g)
    expected = torch.tensor([[0.7129, 0.5792, 0.7931, 0.6961, 0.7549, 0.5772,
                               0.7626, 0.5983, 0.6387, 0.6083]], dtype=torch.float64)
    assert torch.allclose(out, expected, atol=1e-4)

def test_wilson_network_shapes():
    model = WilsonNetwork(hidden_dim=32, n_hidden_layers=2)
    assert len(model.hidden_layers) == 2
    assert model.hidden_layers[0].out_features == 32
    assert model.output_layer.in_features == 32
    assert model.output_layer.out_features == 10