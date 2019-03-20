"""
Warfarin Clinical Dosing Algorithm.
"""

import data_processor as dp
import meta
from util import get_dose_range
from warfarin_base import WarfarineBase

class ClinicalDosing(WarfarineBase):
    def __init__(self, data_path, log_path):
        super(ClinicalDosing, self).__init__(
            "ClinicalDosing", data_path, log_path
        )

    # predict square root of dose for a single sample
    def predict_sample(self, sample):

        # age in decade
        age_decade = dp.get_age_in_decade(sample[meta.AGE])
        if (age_decade is None):
            # not evaluating sample with unknown age
            return None

        # get height in cm
        try:
            height = float(sample[meta.HEIGHT])
        except:
            return None

        # get weight in kg
        try:
            weight = float(sample[meta.WEIGHT])
        except:
            return None

        # Asian race
        is_asian = 1 if sample[meta.RACE] == "Asian" else 0

        # black or African American
        is_baa = 1 if sample[meta.RACE] == "Black or African American" else 0

        # race unknown
        race_unk = 1 if sample[meta.RACE] == "Unknown" else 0

        # enzyme inducer status
        enzyme_inducer_status = dp.get_enzyme_inducer_status(sample)

        # amiodarone status
        amiodarone_status = dp.get_amiodarone_status(sample[meta.MEDICATIONS])

        sqrt_dose = (
            4.0376
            - 0.2546 * age_decade
            + 0.0118 * height
            + 0.0134 * weight
            - 0.6752 * is_asian
            + 0.4060 * is_baa
            + 0.0443 * race_unk
            + 1.2799 * enzyme_inducer_status
            - 0.5695 * amiodarone_status
        )
        dose = sqrt_dose ** 2
        return get_dose_range(dose)

    def step(self, sample, feature, t):
        return self.predict_sample(sample)

    def predict(self, sample, feature):
        return self.predict_sample(sample)

if __name__ == "__main__":
    clinicalDosing = ClinicalDosing(
        "./data/warfarin.csv", "./save/clinical_dosing"
    )
    clinicalDosing.train(3)
