# Wilson-line-CFT

A physics-informed neural network (PINN) approach to the 1D defect CFT
bootstrap for the 1/2-BPS Wilson line in 4d N=4 super Yang-Mills at
strong coupling, benchmarked against the numerical conformal bootstrap
and the reinforcement-learning approach of BootSTOP/MultiSTOP.

**Status:** Completed: special functions, coupling functions,
conformal blocks, integral constraints, strong-coupling reference data.
In progress: convex baselines, then the PINN itself.

## Repository structure

    src/          core physics modules: special functions, coupling
                  functions, conformal blocks, the crossing equation,
                  integral constraints
    data/         data-extraction scripts and the resulting reference CSVs
    tests/        unit tests, one file per src/ module
    notebooks/    supporting figures and derivations

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data provenance

`data/spectrum_strong_coupling.csv` and `data/ope_bounds_strong_coupling.csv`
are extracted from `Conformal_Data.nb`, an ancillary file attached to
Cavaglià, Gromov, Julius, Preti, arXiv:2203.09556. This notebook is the
authors' own supplementary material and is not redistributed in this
repository.

To regenerate the CSVs from scratch:

1. Download `Conformal_Data.nb` from the "Ancillary files" link on
   https://arxiv.org/abs/2203.09556
2. Place it at `data/Conformal_Data.nb`
3. Run:

   ```bash
   python data/extract_conformal_data.py
   python data/extract_OPE_bounds.py
   ```

The two extraction scripts are otherwise self-contained: `resolve_fractions`
and `extract_numbers` (defined in `extract_conformal_data.py`) are reused
by `extract_OPE_bounds.py`.

## References

- Ferrero, Meneghelli, arXiv:2103.10440, arXiv:2312.12550, arXiv:2312.12551
- Cavaglià, Gromov, Julius, Preti, arXiv:2203.09556
- Trenta, Bacciu, Cossu, Ferrero, arXiv:2404.1490

