import pandas as pd
import numpy as np
from pathlib import Path

df = pd.read_csv(Path(__file__).parent.parent / 'data' / 'ope_bounds_strong_coupling.csv', comment = '#')
group_tuples = df.sort_values('g').groupby('n_value')['g'].apply(tuple)

consistency_check = group_tuples.nunique() == 1

assert consistency_check, "g-grids are not consistent across all states"

available_g_ope_values = sorted(df['g'].unique().tolist())

def ope_bounds_at_g(n: int, g: float) -> tuple[float, float]:
    r"""
        Extracts lower and upper bounds of the OPE coefficients for a given state, n, and coupling constant, g.


        Parameters
        ----------
        g: float
            The coupling constant.

        n: int
            The number of the target state.

        Returns
        -------
        Tuple[float, float]
             A tuple that contains the lower and upper bounds for the squares of the 3-point coupling coefficients.

        Notes
        -----
        It raises an error if either 0 or multiple matches are found for a given pair of values (n ,g).
    """
    filtered_df = df[(df['n_value'] == n) & (np.isclose(df['g'], g))]

    if len(filtered_df) == 0:
        raise ValueError(
            f"No bounds found that match the input state {n} and coupling constant {g}. \n"
            f"g has to take value within the range [0.01, 4] and n is equal to 1, 2, 3."
        )
    elif len(filtered_df) > 1:
        raise ValueError(
            f"Data issue. Found {len(filtered_df)} duplicates for n={n} and g={g}."
        )

    lower_bound = float(filtered_df['C2_lower'].iloc[0])
    upper_bound = float(filtered_df['C2_upper'].iloc[0])

    return lower_bound, upper_bound


if __name__ == '__main__':
    print(ope_bounds_at_g(3, 0.5))