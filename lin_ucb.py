from util import read_all_data
from feature_parser import FeatureParser
import meta
import numpy as np
import time

class LinUCB:
    def __init__(self, data_path):
        self.samples = read_all_data(data_path)
        self.parser = FeatureParser()

        self.A = []
        self.b = []
        self.AI = []
        self.theta = []

        self.alpha = 0.25

        self.result_file = "./data/eval_" + str(int(time.time())) + ".txt"

    def train(self):
        print("Parsing features...")
        self.features = self.parser.parse(self.samples)

        print("Initializing parameters...")
        feature_size = None
        for f in self.features:
            if (f is not None):
                feature_size = len(f[0])
                break
        for _ in range(3):
            self.A.append(np.identity(feature_size))
            self.b.append(np.zeros((feature_size, 1)))
            self.AI.append(np.identity(feature_size))
            self.theta.append(np.zeros((feature_size, 1)))

        print("Loading gold doses...")
        self.gold = []
        for sample in self.samples:
            try:
                dose = float(sample[meta.THERAPEUTIC_DOSE])
                self.gold.append(dose)
            except:
                self.gold.append(None)

        print("Training...")
        total_sample = len(self.features)
        for i in range(total_sample):
            feature = self.features[i]
            if (feature is None):
                continue
            # 1 * d
            f_t = feature
            # d * 1
            f = np.transpose(f_t)
            p = np.zeros((1, len(self.A)))
            for a_idx in range(len(self.A)):
                # d * 1
                pa = (
                    np.matmul(np.transpose(self.theta[a_idx]), f) +
                    self.alpha * np.sqrt(
                        np.matmul(np.matmul(f_t, self.AI[a_idx]), f)
                    )
                )
                p[0][a_idx] = float(pa)

            # select the action with highest bound
            a_max = int(np.squeeze(np.argmax(p, axis=1)))
            # calculate real reward
            r = self.calc_reward(self.gold[i], a_max)
            # update parameters
            self.A[a_max] += np.matmul(f, f_t)
            self.b[a_max] += r * f
            self.AI[a_max] = np.linalg.inv(self.A[a_max])
            self.theta[a_max] = np.matmul(self.AI[a_max], self.b[a_max])

            # evaluate every 50 steps
            should_eval = (i % 10 == 0) and (i > 0)
            if (should_eval):
                print("train: {}/{}".format(i, total_sample))
                self.evaluate()
            else:
                print("train: {}/{}".format(i, total_sample), end="\r")
        self.evaluate()

    def evaluate(self):
        total_sample = len(self.features)
        eval_sample = 0
        correct_sample = 0
        for i in range(total_sample):
            f_t = self.features[i]
            if (f_t is None):
                continue
            f = np.transpose(f_t)
            p = np.zeros((1, len(self.A)))
            for a_idx in range(len(self.A)):
                # d * 1
                pa = (
                    np.matmul(np.transpose(self.theta[a_idx]), f) +
                    self.alpha * np.sqrt(
                        np.matmul(np.matmul(f_t, self.AI[a_idx]),f)
                    )
                )
                p[0][a_idx] = float(pa)

            # select the action with highest bound
            a_max = int(np.squeeze(np.argmax(p, axis=1)))

            eval_sample += 1
            r = self.calc_reward(self.gold[i], a_max)
            if (r == 0):
                correct_sample += 1
            print("eval: {}/{}".format(i, total_sample), end="\r")

        precent_correct = correct_sample / eval_sample
        print("Precent correct: {}".format(precent_correct))

        with open(self.result_file, "a") as rf:
            rf.write("{}\n".format(precent_correct))


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
