from baseline import Baseline

class FixedDose(Baseline):
    def __init__(self, data_path):
        super(FixedDose, self).__init__(data_path)

        self.weekly_dose = 35

    def evaluate(self):
        predictions = [self.weekly_dose] * len(self.samples)
        self.compute_metrics(predictions)

if __name__ == "__main__":
    fixed_dose = FixedDose("./data/warfarin.csv")
    fixed_dose.evaluate()
