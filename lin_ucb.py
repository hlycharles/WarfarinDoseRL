import numpy as np

from warfarin_base import WarfarineBase
import meta

class LinUCB(WarfarineBase):
    def __init__(self, data_path, log_path):
        super(LinUCB, self).__init__("LinUCB", data_path, log_path)

        self.A = []
        self.b = []
        self.AI = []
        self.theta = []

        self.alpha = 0.25
        self.feature_size = None

    def pre_train(self):
        for f in self.features:
            if (f is not None):
                self.feature_size = len(f[0])
                break

        self.A = []
        self.b = []
        self.AI = []
        self.theta = []
        for _ in range(3):
            self.A.append(np.identity(self.feature_size))
            self.b.append(np.zeros((self.feature_size, 1)))
            self.AI.append(np.identity(self.feature_size))
            self.theta.append(np.zeros((self.feature_size, 1)))

    def step(self, sample, feature, t):
        if (feature is None):
            return None
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
        dose = self.action_to_range(a_max)
        # calculate real reward
        r = self.calc_reward(float(sample[meta.THERAPEUTIC_DOSE]), dose)
        # update parameters
        self.A[a_max] += np.matmul(f, f_t)
        self.b[a_max] += r * f
        self.AI[a_max] = np.linalg.inv(self.A[a_max])
        self.theta[a_max] = np.matmul(self.AI[a_max], self.b[a_max])

        return dose

    def predict(self, sample, feature):
        f_t = feature
        if (f_t is None):
            return None
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
        return self.action_to_range(a_max)

if __name__ == "__main__":
    linUCB = LinUCB("./data/warfarin.csv", "./save/lin_ucb")
    linUCB.train(1)
