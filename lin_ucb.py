from util import read_all_data
from feature_parser import FeatureParser
import meta
import numpy as np

class LinUCB:
    def __init__(self, data_path):
        self.samples = read_all_data(data_path)
        self.parser = FeatureParser()

        self.A = []
        self.b = []
        self.AI = []

        self.alpha = 1

    def train(self):
        print("Parsing features...")
        features = self.parser.parse(self.samples)

        print("Initializing parameters...")
        feature_size = None
        for f in features:
            if (f is not None):
                feature_size = len(f)
                break
        for _ in range(3):
            self.A.append(np.identity(feature_size))
            self.b.append(np.zeros((feature_size, 1)))
            self.AI.append(np.identity(feature_size))

        print("Loading gold doses...")
        gold = []
        for sample in self.samples:
            try:
                dose = float(sample[meta.THERAPEUTIC_DOSE])
                gold.append(dose)
            except:
                gold.append(None)

        print("Training...")
        total_sample = len(features)
        for i in range(len(features)):
            feature = features[i]
            if (feature is None):
                continue
            # 1 * d
            f_t = np.array([feature])
            # d * 1
            f = np.transpose(f_t)
            p = np.zeros((1, len(self.A)))
            for a_idx in range(len(self.A)):
                # d * 1
                theta = np.matmul(self.AI[a_idx], self.b[a_idx])
                pa = (
                    np.matmul(np.transpose(theta), f) +
                    self.alpha * np.sqrt(
                        np.matmul(np.matmul(f_t, self.AI[a_idx]),f)
                    )
                )
                p[0][a_idx] = float(pa)

            # select the action with highest bound
            a_max = int(np.squeeze(np.argmax(p, axis=1)))
            # calculate real reward
            r = self.calc_reward(gold[i], a_max)
            # update parameters
            self.A[a_max] += np.matmul(f, f_t)
            self.b[a_max] += r * f
            self.AI[a_max] = np.linalg.inv(self.A[a_max])

            print("{}/{}".format(i, total_sample), end="\r")

    def calc_reward(self, gold, pred):
        if (pred == 0):
            return 0 if gold < 21 else -1
        if (pred == 1):
            return 0 if (21 <= gold and gold <= 45) else -1
        if (pred == 2):
            return 0 if gold > 45 else -1
        return -1

if __name__ == "__main__":
    linUCB = LinUCB("./data/warfarin.csv")
    linUCB.train()
