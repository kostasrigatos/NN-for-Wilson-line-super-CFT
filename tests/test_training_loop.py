import pytest
from spectrum_data import available_g_values
from training_loop import train_and_save
from pinn_loss import build_physics_cache

def test_train_and_save_defaults_reproduce_reference():
    g_values = available_g_values[1:]
    design_matrix_cache, target_vector_cache = build_physics_cache(g_values)
    model, loss_history = train_and_save(
        lam=0, g_values=g_values,
        design_matrix_cache=design_matrix_cache, target_vector_cache=target_vector_cache,
        n_epochs=10, seed=42, save=False
    )
    expected = [37677037.42844857, 31943792.843278304, 27036584.501811415, 22805398.255063232,
                19146981.673253287, 15985583.543878578, 13264105.32666534, 10937792.40087919, 8968084.67411444,
                7318357.2506363485]
    for actual, exp in zip(loss_history, expected):
        assert actual == pytest.approx(exp, rel=1e-6)