import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
import torch.nn as nn

from pinn_loss import build_physics_cache, crossing_loss, smoothness_penalty
from pinn_model import WilsonNetwork
from spectrum_data import available_g_values


def train_and_save(lam: float, g_values: list[float],
                   design_matrix_cache: dict[float, torch.Tensor], target_vector_cache: dict[float, torch.Tensor],
                   n_epochs: int = 30000, seed: int = 42, save: bool = True,
                   milestones: list[int] = None, gamma: float = 0.5
                   ) -> tuple[nn.Module, list[float]]:
    r"""
         Trains the WilsonNetwork model for a given number of epochs, using the physics-informed crossing loss optionally
         regularised by a second-order derivative smoothness penalty and saves the resulting trained weights to disk.

         Parameters
         ----------
         lam: float
            The hyperparameter for the second-order derivative smoothness penalty. If set to 0.0, the regularised penalty
            evaluation is circumvented.

         g_values: list[float]
            A list of values of the coupling constant, g, where the matrices of the system are evaluated.

         design_matrix_cache: dict[float, torch.Tensor]
            A dictionary mapping each value of the coupling constant, g, to its 2D design matrix.

         target_vector_cache: dict[float, torch.Tensor]
            A dictionary mapping each value of the coupling constant, g, to its 1D target vector.

         n_epochs: int
             The number of epochs to train for.

         seed: int, default = 42
             The random seed used to enforce the deterministic network initialisation.

         save: bool, default = True
             Whether to save the trained weights to disk.

         milestones: list[int], default = None
             The epochs at which the learning rate is multiplied by gamma; None disables scheduling entirely.

         gamma: float, default = 0.5
             The multiplicative learning-rate decay factor applied at each milestone.

         Returns
         -------
         model: nn.Module
             The trained WilsonNetwork model.

         loss_history: list[float]
             The list that contains the scalar training loss values recorded at each epoch.
     """
    torch.manual_seed(seed)
    model = WilsonNetwork().double()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma) if milestones is not None else None
    loss_history = []

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        if lam > 0:
            loss = crossing_loss(model, g_values, design_matrix_cache, target_vector_cache) + lam * smoothness_penalty(
                model, g_values)
        else:
            loss = crossing_loss(model, g_values, design_matrix_cache, target_vector_cache)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()
        loss_history.append(loss.item())

        if epoch % 100 == 0:
            print(f"For epoch {epoch} the loss is {loss.item()}")

    if save:
        save_path = Path(__file__).parent.parent / "models" / f"trained_model_lam_{lam}.pt"
        torch.save(model.state_dict(), save_path)

    return model, loss_history

def train_multiple_seeds(lam: float, seeds: list[int], g_values: list[float], design_matrix_cache: dict[float, torch.Tensor], target_vector_cache: dict[float, torch.Tensor], n_epochs: int = 30000) -> list[dict]:
    r"""
         Trains the WilsonNetwork model independently across a list of random seeds, for a fixed regularisation weight,
         collecting each run's trained model and full loss history without writing anything to disk.

         Parameters
         ----------
         lam: float
            The hyperparameter for the second-order derivative smoothness penalty. If set to 0.0, the regularised penalty
            evaluation is circumvented.

         seeds: list[int]
             The list of seeds to train for.

         g_values: list[float]
            A list of values of the coupling constant, g, where the matrices of the system are evaluated.

         design_matrix_cache: dict[float, torch.Tensor]
            A dictionary mapping each value of the coupling constant, g, to its 2D design matrix.

         target_vector_cache: dict[float, torch.Tensor]
            A dictionary mapping each value of the coupling constant, g, to its 1D target vector.

         n_epochs: int
             The number of epochs to train for.

         Returns
         -------
         list[dict]
             A list of dictionaries containing tracking data for each initialisation path.
             Each dictionary contains the following keys:
             a) seed: int. The initialisation seed used.
             b) model: nn.Module. The trained WilsonNetwork model instance.
             c) loss: list[float]. The full scalar loss history recorded across epochs.
     """
    results = []
    for seed in seeds:
        model, loss_history = train_and_save(lam, g_values, design_matrix_cache, target_vector_cache, n_epochs=n_epochs, seed=seed, save=False)
        results.append({'seed': seed, 'model': model, 'loss': loss_history})
    return results

def _train_one_job(args):
    r"""
          Executes a single isolated network training instance for multiprocessing workers.

          Parameters
          ----------
          args: tuple[float, int, int, list[int] | None, float]
              A tuple that contains:
              a) lam: float. The hyperparameter scaling factor.
              b) seed: int. The initialisation seed used to train the network.
              c) n_epochs: int. The total number of optimisation iterations.
              d) milestones: list[int] | None. The epochs at which the learning rate is multiplied by gamma;
                 None disables scheduling entirely.
              e) gamma: float. The multiplicative learning-rate decay factor applied at each milestone.

          Returns
          -------
          dict
              A dictionary containing tracking data for each initialisation path.
              Each dictionary contains the following keys:
              a) lam: float. The hyperparameter scaling factor.
              b) seed: int. The initialisation seed used to train the network.
              c) model: nn.Module. The fully trained model instance.
              d) loss: list[float]. The scalar history tracking logs.
              e) cache_time: float. Time taken in seconds to build the physics tensor cache.
              f) train_time: float. Time taken in seconds to run the optimisation loop.

          Notes
          -----
          This helper function (single-worker) unpacks a set of configurations. It limits PyTorch processing overhead
          to a single compute thread to eliminate multi-process CPU thrashing. Furthermore, it generates isolated tensor
          caches, and trains a single configuration path while logging precise execution timers.
      """
    lam, seed, n_epochs, milestones, gamma = args
    torch.set_num_threads(1)
    t0 = time.perf_counter()
    g_values = available_g_values[1:]
    design_matrix_cache, target_vector_cache = build_physics_cache(g_values)
    t1 = time.perf_counter()
    model, loss_history = train_and_save(lam, g_values, design_matrix_cache, target_vector_cache, n_epochs=n_epochs, seed=seed, save=False, milestones=milestones, gamma=gamma)
    t2 = time.perf_counter()
    return {'lam': lam, 'seed': seed, 'model': model, 'loss': loss_history, 'cache_time': t1 -t0, 'train_time': t2 - t1}

def train_seed_sweep_parallel(lams, seeds, n_epochs=30000, max_workers=10, milestones=None, gamma=0.5):
    r"""
          Executes a concurrent 2D hyperparameter cross-product sweep over a set of lambdas and seeds.

          Parameters
          ----------
          lams: list[float]
              A list of values for the lambda hyperparameter.
          seeds: list[int]
              A list of unique initialisation random seeds to process per value of lambda.
          n_epochs: int, default = 30000
             The maximum number of training iterations for each worker job instance.
          max_workers: int, default = 10
             The maximum number of worker processes running concurrently in the pool.
          milestones: list[int] | None, default = None. The epochs at which the learning rate is multiplied by gamma;
                 None disables scheduling entirely.
          gamma: float, default = 0.5. The multiplicative learning-rate decay factor applied at each milestone.

          Returns
          -------
          list[dict]
              A list of dictionaries, one per completed job, each containing that job's tracking data - see
              the helper function _train_one_job for details.

          Notes
          -----
          This function builds the full (λ, seed) cross-product, dispatches each combination to a worker process via
          ProcessPoolExecutor, and collects each job's returned dictionary, namely trained model, loss history,
          and timing data, into a single list with no change to model architecture across jobs. Note that milestones/gamma
          apply identically to every job in this sweep - a single call cannot mix scheduled and unscheduled runs; running
          both requires two separate calls.
      """
    jobs = [(lam, seed, n_epochs, milestones, gamma) for lam in lams for seed in seeds]
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_train_one_job, jobs))
    return results

