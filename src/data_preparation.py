import os
import kaggle
import pandas as pd
import numpy as np
from typing import List, Tuple

"""
Select ur data files here. Options:

"data/raw/aspell.txt", "data/raw/birkbeck.txt", "data/raw/spell-testset1.txt", "data/raw/spell-testset2.txt", "data/raw/wikipedia.txt"

"""
SRC_PATH = ["data/raw/wikipedia.txt"]


#This method downloads the dataset from kaggle directly, however, u will need an api key
def download_kaggle_dataset(dataset: str, path: str):
    kaggle.api.dataset_download_files(dataset, path=path, unzip=True)


def process_txt_file(filepath: str) -> List[Tuple[str, str]]:
    """Processes a .txt file where each line contains a misspelled and corrected word pair."""
    errors = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            pair = line.strip().split(':')
            correct_word = pair[0]  # The correct word is always the last one
            for error in pair[1:]:  # The rest are the erroneous versions
                for word in error.split():
                    errors.append((correct_word, word))
    return errors


def process_all_files(filelist: List[str]) -> List[Tuple[str, str]]:
    """Processes all .txt files in the directory and combines the errors."""
    all_errors = []
    for filepath in filelist:
        if filepath.endswith('.txt'):
            errors = process_txt_file(filepath)
            all_errors.extend(errors)
            print(f"Processed {filepath}: {len(errors)} errors")
    return all_errors


def generate_synthetic_errors(words: List[str], n: int) -> List[Tuple[str, str]]:
    """Generates synthetic errors by inserting, deleting, replacing, or swapping characters."""
    import numpy as np
    errors = []
    valid_words = [str(word) for word in words if isinstance(word, (str, int, float)) and str(word).strip()]

    if not valid_words:
        print("Warning: No valid words found for generating synthetic errors.")
        return errors

    for _ in range(n):
        word = np.random.choice(valid_words)
        error_type = np.random.choice(['insert', 'delete', 'replace', 'swap'])

        if error_type == 'insert':
            pos = np.random.randint(0, len(word) + 1)
            char = np.random.choice(list('abcdefghijklmnopqrstuvwxyz'))
            error = word[:pos] + char + word[pos:]
        elif error_type == 'delete' and len(word) > 1:
            pos = np.random.randint(0, len(word))
            error = word[:pos] + word[pos + 1:]
        elif error_type == 'replace':
            pos = np.random.randint(0, len(word))
            char = np.random.choice(list('abcdefghijklmnopqrstuvwxyz'))
            error = word[:pos] + char + word[pos + 1:]
        elif len(word) > 1:  # swap
            pos = np.random.randint(0, len(word) - 1)
            error = word[:pos] + word[pos + 1] + word[pos] + word[pos + 2:]
        else:
            error = word  # If we can't modify the word, keep it as is

        errors.append((word, error ))

    return errors


def main(src_path=None):
    # Create directories
    if src_path is None:
        src_path = SRC_PATH
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)

    # Process all spelling error files in the raw directory
    spelling_errors = process_all_files(src_path)

    # # Generate synthetic errors
    # words = [word for word, _ in spelling_errors]
    # synthetic_errors = generate_synthetic_errors(words, int(0.1*len(spelling_errors)))

    # Combine all errors
    all_errors = spelling_errors #+ synthetic_errors

    # Save processed data
    df = pd.DataFrame(all_errors, columns=['correct', 'error'])
    df.to_csv('data/processed/spelling_errors.csv', index=False)

    print(f"Total spelling errors collected: {len(all_errors)}")


if __name__ == "__main__":
    main()