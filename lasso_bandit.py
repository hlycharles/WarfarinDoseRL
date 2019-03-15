from util import read_all_data
from feature_parser import FeatureParser
import meta
import numpy as np
from sklearn import linear_model
import time
import math
import random
from baseline import Baseline

class LassoBandit(Baseline):
    def __init__(self, data_path, log_path):
        super(LassoBandit, self).__init__("LassoBandit", data_path, log_path)

        self.q = 80
        self.h = 1.0
        self.tau = []
        self.lmd1 = 0.02
        self.lmd2_init = 0.5
        self.fl_est = []
        self.al_est = []

        self.f_x = []
        self.f_y = []
        self.a_x = []
        self.a_y = []

        self.feature_size = None

    def pre_train(self):
        self.tau = []
        self.fl_est = []
        self.al_est = []
        self.f_x = []
        self.f_y = []
        self.a_x = []
        self.a_y = []

        for f in self.features:
            if (f is not None):
                self.feature_size = len(f[0])
                break

        # initialize forced sample indices
        for i in range(1, 4):
            t = []
            base = 1
            finish = False
            while (not finish):
                for j in range(self.q * (i - 1) + 1, self.q * i + 1):
                    idx = (base - 1) * 3 * self.q  + j
                    if (idx < 6000):
                        t.append(idx)
                    else:
                        finish = True
                        break
                base = base * 2
            self.tau.append(set(t))

        for i in range(3):
            self.fl_est.append(None)
            self.al_est.append(None)
            self.f_x.append([])
            self.f_y.append([])
            self.a_x.append([])
            self.a_y.append([])

    def step(self, sample, feature, t):

        if (feature is None):
            return None

        action = None
        forced_action = None
        for t_idx in range(len(self.tau)):
            if (t in self.tau[t_idx]):
                forced_action = t_idx
                action = t_idx
                break

        if (action is None):
            f_ests = []
            for beta in self.fl_est:
                f_ests.append(beta.predict(feature.tolist())[0])
            boundary = max(f_ests) - self.h / 2

            best_action = None
            best_val = None
            for a in range(3):
                if (f_ests[a] < boundary):
                    continue
                val = float(self.al_est[a].predict(feature.tolist())[0])
                if (best_action is None or val > best_val):
                    best_action = a
                    best_val = val
            action = best_action

        dose = self.action_to_range(action)
        reward = self.calc_reward(float(sample[meta.THERAPEUTIC_DOSE]), dose)
        if (forced_action is not None):
            self.f_x[action].append(feature[0].tolist())
            self.f_y[action].append(reward)
            lasso = linear_model.Lasso(
                alpha=self.lmd1, fit_intercept=True, max_iter=10000
            )
            lasso.fit(self.f_x[action], self.f_y[action])
            self.fl_est[action] = lasso

        self.a_x[action].append(feature[0].tolist())
        self.a_y[action].append(reward)
        lmd2 = self.lmd2_init * math.sqrt(
            (math.log(t) + math.log(self.feature_size)) / t
        )
        for idx in range(3):
            if (len(self.a_x[idx]) == 0):
                continue
            lasso = linear_model.Lasso(
                alpha=lmd2, fit_intercept=True, max_iter=10000
            )
            lasso.fit(self.a_x[idx], self.a_y[idx])
            self.al_est[idx] = lasso

        return dose

    def predict(self, sample, feature):

        if (feature is None):
            return None

        f_ests = []
        for beta in self.fl_est:
            if (beta is None):
                f_ests.append(None)
            else:
                f_ests.append(beta.predict(feature.tolist())[0])
        f_ests_f = list(filter(lambda v: v is not None, f_ests))
        boundary = max(f_ests_f) - self.h / 2

        best_action = None
        best_val = None
        for a in range(3):
            if (f_ests[a] is None or f_ests[a] < boundary):
                continue
            if (self.al_est[a] is None):
                continue
            val = float(self.al_est[a].predict(feature.tolist())[0])
            if (best_action is None or val > best_val):
                best_action = a
                best_val = val
        action = best_action

        return self.action_to_range(action)

if __name__ == "__main__":
    lasso = LassoBandit("./data/warfarin.csv", "./save/lasso_bandit")
    lasso.train(1)
