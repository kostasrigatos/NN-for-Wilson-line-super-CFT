import pandas as pd
import numpy as np
from pathlib import Path

df = pd.read_csv(Path(__file__).parent.parent / 'data' / 'spectrum_strong_coupling.csv', comment='#')
df = df.drop_duplicates()

group_tuples = df.sort_values('g').groupby('n_value')['g'].apply(tuple)

consistency_check = group_tuples.nunique() == 1

assert consistency_check, "g-grids are not consistent across all states"

available_g_values = sorted(df['g'].unique().tolist())

def spectrum_at_g(g: float) -> list[float]:
    r"""
        Extracts and sorts the values for Delta for a given value of the coupling constant, g, from a DataFrame.


        Parameters
        ----------
        g: float
            The coupling constant.

        Returns
        -------
        list[float]
             A list of 10 sorted 'Delta' values corresponding to the states at the coupling g.

        Notes
        -----
        It depends on the module-level DataFrame, df, already being deduplicated and validated.
        It filters a DataFrame, df, which sits at the module level for rows that the column 'g' matches the given 'g'
        within floating point precision. It enforces that exactly 10 states must match before sorting the rows 'n_value'
        and returning the corresponding values for 'Delta'.
        It raises two distinct failures.
        1) A value for the coupling, g, that does not exist.
        2) A coupling, g, that exists but has the wrong row-count.
    """
    filter = df[np.isclose(df['g'], g)]
    if len(filter) != 10:
        raise ValueError(f"g = {g} does not have exactly 10 states in the data; found {len(filter)} instead.")
    sorted_rows = filter.sort_values('n_value')
    delta_values = sorted_rows['Delta'].tolist()
    return delta_values

if __name__ == "__main__":
    print(spectrum_at_g(available_g_values[50]))