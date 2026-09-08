import re
from pathlib import Path
import csv

def resolve_fractions(chunk: str) -> str:
    pattern = r'FractionBox\[\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\]'

    def replacer(match):
        a = float(match.group(1))
        b = float(match.group(2))

        result = a / b

        return str(result)

    return re.sub(pattern, replacer, chunk)

def extract_numbers(text: str) -> list[float]:

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
