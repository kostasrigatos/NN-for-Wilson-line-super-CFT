import re
from pathlib import Path
import csv

def resolve_fractions(chunk: str) -> str:
    r"""
        Looks through an arbitrary chunk of text that might or might not contain instances of FractionBox[...],
        transforms and replaces the components of every FractionBox[...] into a string of a floating number.

        Parameters
        ----------
        chunk: str
            The chunk of text to be transformed --- output of extract_block.

        Returns
        -------
        str
             e.g: FractionBox["1", "100"] becomes "0.01".

        Notes
        -----
        Due to the fact that FractionBox[...] is not of a form that the number-extraction regex recognizes,
        it must run before extract_numbers.
    """
    pattern = r'FractionBox\[\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\]'

    def replacer(match):
        a = float(match.group(1))
        b = float(match.group(2))

        result = a / b

        return str(result)

    return re.sub(pattern, replacer, chunk)

def extract_numbers(text: str) -> list[float]:
    r"""
        Extracts all real numbers from a string of text via a 2-step process.
        Firstly, it strips backtick-precision markers; Mathematica's arbitrary precision annotation.
        Subsequently, it pulls the bare numerical tokens.

        Parameters
        ----------
        text: str
            The input string that contains the numbers to be extracted --- output of resolve_fractions.

        Returns
        -------
        list[float]
             A list of all extracted numbers converted properly to floating values.

        Notes
        -----
        It assumes that resolve_fractions has already been executed.
        In other words, fractions are not handled internally in this function.
    """
    precision_pattern = r'`\d*\.?\d*'
    clean_text = re.sub(precision_pattern, '', text)

    number_pattern = r'-?\d+(?:\.\d+)?'
    raw_numbers = re.findall(number_pattern, clean_text)

    number_list = []

    for number in raw_numbers:
        float_value = float(number)
        number_list.append(float_value)

    return number_list

# Extracting the spectrum as a triplet: (n, Δ, g)

def extract_block(text: str, state_number: int) -> str:
    r"""
        Extract the raw text of the Delta[n] assignment from the notebook source.

        Parameters
        ----------
        text: str
            The full content of the ConformalData.nb text.
        state_number: int
            The block of which state to extract - runs from 1 to 10.

        Returns
        -------
        str
             e.g: FractionBox["1", "100"] becomes "0.01".

        Notes
        -----
        The start marker reproduces the literal syntax of the notebook for a Delta[n] = {...} assignment.
        This includes the backslash and bracket prefix, e.g \\[CapitalDelta]. The end marker - ']], "Code", ' - is
        where Mathematica appends the metadata that we should exclude.
    """
    start_marker = rf'\[CapitalDelta]", "[", "{state_number}", "]"}}], " ", "=", " ",'
    end_marker = rf']], "Code",'

    position = text.index(start_marker)
    start = position + len(start_marker)

    end = text.index(end_marker, start)

    block_text = text[start:end]
    return block_text

if __name__ == '__main__':

    notebook_path = Path(__file__).parent / 'Conformal_Data.nb'
    with open(notebook_path, encoding = 'utf-8') as f:
        text = f.read()

    list_of_triples = []

    for n_value in range(1,11):
        extracted_block = extract_block(text, n_value)
        resolved = resolve_fractions(extracted_block)
        extracted = extract_numbers(resolved)
        values_coupling  = extracted[0::2]
        values_conf_dims = extracted[1::2]
        g_Delta_list = zip(values_coupling, values_conf_dims)
        triples = [(n_value, g, Delta) for g, Delta in g_Delta_list]
        list_of_triples.extend(triples)

    spectrum_csv_data_path = Path(__file__).parent / 'spectrum_strong_coupling.csv'
    with open(spectrum_csv_data_path, 'w', newline='') as f:
        f.write('# Source: arXiv:2203.09556 ancillary file Conformal_Data.nb.\n')
        f.write('# Extracted via data/extract_conformal_data.py.\n')
        writer = csv.writer(f)
        writer.writerow(['n_value', 'g', 'Delta'])
        writer.writerows(list_of_triples)
