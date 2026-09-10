from pathlib import Path
import csv
from extract_conformal_data import resolve_fractions, extract_numbers

# Extracting the OPE coefficients

def extract_OPE(text: str, state_number: int) -> str:
    r"""
        Extracts the string for the OPE coefficients for a specific state from raw text input.
        It searches for a sub-string that is bounded by a start marker specific to the state and a trailing end marker.
        It returns everything contained between the two delimiters; the start marker and the end marker.

        Parameters
        ----------
        text: str
            The full content of the ConformalData.nb text.
        state_number: int
            The block of which state to extract - runs from 1 to 10.

        Returns
        -------
        str
             The extracted string for the OPE coefficients located between the markers.

        Notes
        -----
        start_marker is of the form (`"BoundsCsquared", "[", "<state_number>", "]"}}], " ", "=", `).
        end_marker is of the form (`]], "Code", `).
        If either the start_marker or the end_marker are not present in the input text,
        it Raises a ValueError.
    """
    start_marker = rf'"BoundsCsquared", "[", "{state_number}", "]"}}], " ", "=", '
    end_marker = rf']], "Code",'

    position = text.index(start_marker)
    start = position + len(start_marker)

    end = text.index(end_marker, start)

    OPE_text = text[start:end]
    return OPE_text

if __name__ == "__main__":

    notebook_path = Path(__file__).parent / 'Conformal_Data.nb'
    with open(notebook_path, encoding='utf-8') as f:
        text = f.read()

    list_of_quads = []

    for n_value in range(1, 4):
        extracted_OPE= extract_OPE(text, n_value)
        resolved = resolve_fractions(extracted_OPE)
        extracted = extract_numbers(resolved)
        g = extracted[0::3]
        lower_bound = extracted[1::3]
        upper_bound = extracted[2::3]
        quadruples = [(n_value, g_val, lb, ub) for g_val, lb, ub in zip(g, lower_bound, upper_bound)]
        list_of_quads.extend(quadruples)

    ope_csv_data_path = Path(__file__).parent / 'ope_bounds_strong_coupling.csv'
    with open(ope_csv_data_path, 'w', newline='') as f:
        f.write('# Source: arXiv:2203.09556 ancillary file Conformal_Data.nb.\n')
        f.write('# Extracted via data/extract_OPE_bounds.py.\n')
        writer = csv.writer(f)
        writer.writerow(['n_value', 'g', 'C2_lower', 'C2_upper'])
        writer.writerows(list_of_quads)
