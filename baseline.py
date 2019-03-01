"""
Parent class for baseline implementations.
"""

from util import read_all_data, doses_in_same_range
import meta

class Baseline:
    def __init__(self, data_path):
        self.samples = read_all_data(data_path)

    # expect predictions to have same length as samples
    def compute_metrics(self, predictions):
        eval_sample_count = 0
        correct_sample_count = 0

        for i in range(len(self.samples)):
            try:
                dose_gold = float(self.samples[i][meta.THERAPEUTIC_DOSE])
            except:
                continue

            dose_pred = predictions[i]
            if (dose_pred is None):
                continue

            eval_sample_count += 1
            if (doses_in_same_range(dose_pred, dose_gold)):
                correct_sample_count += 1

        correct_frac = correct_sample_count / eval_sample_count

        # print results
        print("Total evaluated examples: {total}".format(
            total=eval_sample_count
        ))
        print("Correct examples: {correct}".format(
            correct=correct_sample_count
        ))
        print("Fraction correct: {frac}".format(frac=correct_frac))

        return (eval_sample_count, correct_sample_count, correct_frac)
