import meta
from feature_entry import FeatureEntry
from util import read_all_data
import data_processor as dp
import numpy as np

class FeatureParser:
    def __init__(self):
        feature_tags = [
            (meta.FEATURE_NO, False, None),
            (meta.FEATURE_ENUM, False, None),
            (meta.FEATURE_ENUM, False, None),
            (meta.FEATURE_ENUM, False, None),
            (meta.FEATURE_RANGE, False, None),
            (meta.FEATURE_NUM, False, None),
            (meta.FEATURE_NUM, False, None),
            (meta.FEATURE_LIST_MAX, False, 0),
            (meta.FEATURE_LIST_ENUM, False, None),
            (meta.FEATURE_BIN, False, None),
            (meta.FEATURE_BIN, False, None),
            (meta.FEATURE_BIN, False, None),
            # M
            (meta.FEATURE_LIST_ENUM, False, None),
            (meta.FEATURE_BIN, False, None),
            (meta.FEATURE_BIN, False, None),
            (meta.FEATURE_BIN, False, None),
            # Q
            (meta.FEATURE_BIN, False, None),
            (meta.FEATURE_BIN, False, None),
            (meta.FEATURE_BIN, False, None),
            (meta.FEATURE_BIN, False, None),
            (meta.FEATURE_BIN, False, None),
            (meta.FEATURE_BIN, False, None),
            (meta.FEATURE_BIN, False, None),
            (meta.FEATURE_BIN, False, None),
            (meta.FEATURE_BIN, False, None),
            (meta.FEATURE_BIN, False, None),
            (meta.FEATURE_BIN, False, None),
            (meta.FEATURE_BIN, False, None),
            (meta.FEATURE_BIN, False, None),
            (meta.FEATURE_BIN, False, None),
            # AE
            (meta.FEATURE_BIN, False, None),
            (meta.FEATURE_NUM, False, 0),
            (meta.FEATURE_RANGE, False, 0),
            (meta.FEATURE_BIN, False, None),
            (meta.FEATURE_NO, True, None),
            (meta.FEATURE_NUM, False, 0),
            (meta.FEATURE_BIN, False, None),
            (meta.FEATURE_ENUM, False, None),
            (meta.FEATURE_ENUM, False, None),
            (meta.FEATURE_ENUM, False, None),
            # AO
            (meta.FEATURE_ENUM, False, None),
            (meta.FEATURE_ENUM, False, None),
            (meta.FEATURE_ENUM, False, None),
            (meta.FEATURE_ENUM, False, None),
            # AS
            (meta.FEATURE_ENUM, False, None),
            (meta.FEATURE_ENUM, False, None),
            (meta.FEATURE_ENUM, False, None),
            (meta.FEATURE_ENUM, False, None),
            (meta.FEATURE_ENUM, False, None),
            (meta.FEATURE_ENUM, False, None),
            (meta.FEATURE_ENUM, False, None),
            (meta.FEATURE_ENUM, False, None),
            # BA
            (meta.FEATURE_ENUM, False, None),
            (meta.FEATURE_ENUM, False, None),
            (meta.FEATURE_ENUM, False, None),
            (meta.FEATURE_ENUM, False, None),
            (meta.FEATURE_ENUM, False, None),
            (meta.FEATURE_ENUM, False, None),
            (meta.FEATURE_ENUM, False, None),
            (meta.FEATURE_ENUM, False, None),
            (meta.FEATURE_ENUM, False, None),
            (meta.FEATURE_ENUM, False, None),
            (meta.FEATURE_ENUM, False, None),
        ]
        self.feature_entries = list(map(
            lambda e: FeatureEntry(e[0], e[1], e[2]), feature_tags
        ))

    def parse(self, samples):
        for sample in samples:
            for i in range(len(self.feature_entries)):
                self.feature_entries[i].add_sample(sample[i])

        result = []
        for sample in samples:
            features = []
            for i in range(len(self.feature_entries)):
                feature = self.feature_entries[i].parse(sample[i])
                if (feature is None):
                    features = None
                    break
                features.extend(feature)

            if (features is not None):
                # engineered features
                features.append(dp.get_enzyme_inducer_status(sample))
                features.append(dp.get_amiodarone_status(
                    sample[meta.MEDICATIONS]
                ))

            if (features is None):
                result.append(None)
            else:
                result.append(np.array([features]))

        return result

if __name__ == "__main__":
    parser = FeatureParser()
    samples = read_all_data("./data/warfarin.csv")
    parser.parse(samples)
