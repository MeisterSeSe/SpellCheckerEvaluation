import os
import random
import warnings
from contextlib import contextmanager

import pandas as pd
import time
from textblob import TextBlob
from spellchecker import SpellChecker
from symspellpy import SymSpell, Verbosity
from autocorrect import Speller
from spello.model import SpellCorrectionModel
from tqdm import tqdm

@contextmanager
def suppress_spello_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='This model was saved on spell<1.3.0.*')
        yield
class SpellCheckerEvaluator:
    def __init__(self, data_path, correct_word_sample_rate=0.1):
        self.data = pd.read_csv(data_path)
        self.correct_word_sample_rate = correct_word_sample_rate  # Percentage of correct words to check
        self.initialize_spellcheckers()
        self.results = {
            'textblob': {'confusion_matrix': {'true_corrections': 0,  # Word needed correction and was corrected properly
                                            'false_corrections': 0,   # Word needed correction but was corrected wrongly
                                            'false_alarms': 0,        # Word was correct but got "corrected"
                                            'true_negatives': 0},     # Word was correct and left unchanged
                        'time': 0},
            'pyspellchecker': {'confusion_matrix': {'true_corrections': 0, 'false_corrections': 0,
                                                   'false_alarms': 0, 'true_negatives': 0}, 'time': 0},
            'symspell': {'confusion_matrix': {'true_corrections': 0, 'false_corrections': 0,
                                            'false_alarms': 0, 'true_negatives': 0}, 'time': 0},
            'spello': {'confusion_matrix': {'true_corrections': 0, 'false_corrections': 0,
                                          'false_alarms': 0, 'true_negatives': 0}, 'time': 0},
            'autocorrect': {'confusion_matrix': {'true_corrections': 0, 'false_corrections': 0,
                                               'false_alarms': 0, 'true_negatives': 0}, 'time': 0}
        }
    def initialize_spellcheckers(self):
        # TextBlob
        self.textblob = TextBlob

        # PySpellChecker
        self.pyspellchecker = SpellChecker()

        # SymSpell
        self.symspell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        dictionary_path = "frequency_dictionary_en_82_765.txt"
        if not os.path.exists(dictionary_path):
            print("Downloading SymSpell dictionary...")
            import requests
            url = "https://raw.githubusercontent.com/mammothb/symspellpy/master/symspellpy/frequency_dictionary_en_82_765.txt"
            response = requests.get(url)
            with open(dictionary_path, 'wb') as f:
                f.write(response.content)
        self.symspell.load_dictionary(dictionary_path, term_index=0, count_index=1)

        # Spello
        try:
            with suppress_spello_warnings():
                self.spello = SpellCorrectionModel(language='en')
                self.spello.load('data/pythonModels/en.pkl')
        except Exception as e:
            print(f"Error loading Spello model: {e}")
            self.spello = None

        # Autocorrect
        self.autocorrect = Speller(lang='en')

    def correct_word(self, word, checker_name):
        try:
            if checker_name == 'textblob':
                return str(TextBlob(word).correct())
            elif checker_name == 'pyspellchecker':
                return self.pyspellchecker.correction(word)
            elif checker_name == 'symspell':
                suggestions = self.symspell.lookup(word, Verbosity.CLOSEST)
                return suggestions[0].term if suggestions else word
            elif checker_name == 'spello' and self.spello is not None:
                return self.spello.spell_correct(word)['spell_corrected_text']
            elif checker_name == 'autocorrect':
                return self.autocorrect(word)
            return word
        except Exception as e:
            print(f"Error with {checker_name} on word '{word}': {e}")
            return word

    def evaluate(self):
        for checker_name in self.results.keys():
            print(f"\nEvaluating {checker_name}...")
            start_time = time.time()

            total_true_corrections = 0
            total_false_corrections = 0
            total_false_alarms = 0
            total_true_negatives = 0

            # Process entire dataset linearly
            for _, row in tqdm(self.data.iterrows(), total=len(self.data), desc=f"Processing {checker_name}"):
                correct_word = row['correct']
                error_word = row['error']

                # Test correction of misspelled words
                predicted = self.correct_word(error_word, checker_name)
                if predicted == correct_word:
                    total_true_corrections += 1
                else:
                    total_false_corrections += 1

                # Randomly sample correct words to check if they're preserved
                if random.random() < self.correct_word_sample_rate:
                    correct_word_prediction = self.correct_word(correct_word, checker_name)
                    if correct_word_prediction == correct_word:
                        total_true_negatives += 1
                    else:
                        total_false_alarms += 1

            end_time = time.time()

            # Calculate metrics
            total_actual_errors = total_true_corrections + total_false_corrections  # Total errors in dataset
            total_predicted_corrections = total_true_corrections + total_false_alarms  # Total corrections made
            total_valid_words_tested = total_true_negatives + total_false_alarms  # Total valid words tested
            total_cases = total_actual_errors + total_valid_words_tested

            # Precision: Of all corrections the system made (including to valid words), how many were correct?
            precision = total_true_corrections / total_predicted_corrections if total_predicted_corrections > 0 else 0

            # Recall: Of all actual errors in the dataset, how many did we correct properly?
            recall = total_true_corrections / total_actual_errors if total_actual_errors > 0 else 0

            # Accuracy: Of all words processed (both errors and valid words), how many were handled correctly?
            accuracy = (total_true_corrections + total_true_negatives) / total_cases if total_cases > 0 else 0

            # F1 Score
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            print(f"\nDetailed results for {checker_name}:")
            print(f"True Corrections (correctly fixed errors): {total_true_corrections}")
            print(f"False Corrections (incorrectly fixed errors): {total_false_corrections}")
            print(f"False Alarms (incorrectly changed valid words): {total_false_alarms}")
            print(f"True Negatives (correctly preserved valid words): {total_true_negatives}")
            print(f"Total actual errors in dataset: {total_actual_errors}")
            print(f"Total corrections attempted: {total_predicted_corrections}")
            print(f"Total valid words tested: {total_valid_words_tested}")

            self.results[checker_name].update({
                'confusion_matrix': {
                    'true_corrections': total_true_corrections,
                    'false_corrections': total_false_corrections,
                    'false_alarms': total_false_alarms,
                    'true_negatives': total_true_negatives
                },
                'metrics': {
                    'precision': precision,
                    'recall': recall,
                    'accuracy': accuracy,
                    'f1': f1
                },
                'time': end_time - start_time
            })

    def print_results(self):
        metrics_order = [
            'True Corrections',
            'False Corrections',
            'False Alarms',
            'True Negatives',
            'Precision',
            'Recall',
            'Accuracy',
            'F1 Score',
            'Time (s)'
        ]

        results_df = pd.DataFrame({
            checker: {
                'True Corrections': results['confusion_matrix']['true_corrections'],
                'False Corrections': results['confusion_matrix']['false_corrections'],
                'False Alarms': results['confusion_matrix']['false_alarms'],
                'True Negatives': results['confusion_matrix']['true_negatives'],
                'Precision': results['metrics']['precision'],
                'Recall': results['metrics']['recall'],
                'Accuracy': results['metrics']['accuracy'],
                'F1 Score': results['metrics']['f1'],
                'Time (s)': results['time']
            }
            for checker, results in self.results.items()
        }).round(4)

        # Reorder rows according to metrics_order
        results_df = results_df.reindex(metrics_order)

        print("\nFinal Results:")
        print("-" * 50)
        print(results_df.to_string())

        # Print speed comparison
        base_time = min(results['time'] for checker, results in self.results.items())
        print("\nRelative Speed (normalized to fastest):")
        for checker, results in self.results.items():
            print(f"{checker}: {results['time'] / base_time:.2f}x")

        # Save with index (row names)
        results_df.to_csv("results/results.csv")


def main():
    evaluator = SpellCheckerEvaluator('data/processed/spelling_errors.csv')
    evaluator.evaluate()
    evaluator.print_results()


if __name__ == "__main__":
    main()