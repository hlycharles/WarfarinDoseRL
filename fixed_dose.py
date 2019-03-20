from warfarin_base import WarfarineBase
import meta

class FixedDose(WarfarineBase):
    def __init__(self, data_path, log_path):
        super(FixedDose, self).__init__("FixedDose", data_path, log_path)

        self.weekly_dose = 35

    def step(self, sample, feature, t):
        return meta.DOSE_MD

    def predict(self, sample, feature):
        return meta.DOSE_MD

if __name__ == "__main__":
    fixed_dose = FixedDose("./data/warfarin.csv", "./save/fixed_dose")
    fixed_dose.train(3)
