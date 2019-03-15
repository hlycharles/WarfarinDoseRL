"""
Parent class for baseline implementations.
"""

from util import read_all_data, doses_in_same_range, has_gold_dose
from feature_parser import FeatureParser
import meta
import time
import random
import os

class Baseline:
    def __init__(self, name, data_path, log_path):
        self.name = name

        samples = read_all_data(data_path)
        parser = FeatureParser()
        features = parser.parse(samples)

        idxs = [i for i in range(len(samples))]
        idxs = list(filter(lambda i: has_gold_dose(samples[i]), idxs))
        self.samples = list(map(lambda i: samples[i], idxs))
        self.features = list(map(lambda i: features[i], idxs))

        log_path_t = log_path + "/" + str(int(time.time()))
        os.makedirs(log_path_t)
        self.regret_path = log_path_t + "/" + "regret_"
        self.eval_path = log_path_t + "/" + "eval_"

    def pre_train(self):
        pass

    # take action on next sample and observe reward
    def step(self, sample, feature, t):
        return meta.DOSE_MD

    def predict(self, sample, feature):
        return meta.DOSE_MD

    def train(self, rounds):
        print("Evaluating {}".format(self.name))
        for rnd in range(rounds):
            print("Round {}/{}".format(rnd + 1, rounds))
            idxs = [i for i in range(len(self.samples))]
            random.shuffle(idxs)
            samples = list(map(lambda i: self.samples[i], idxs))
            features = list(map(lambda i: self.features[i], idxs))

            regret_path = self.regret_path + str(rnd + 1) + ".txt"
            eval_path = self.eval_path + str(rnd + 1) + ".txt"

            total_sample = 0
            correct_sample = 0
            self.pre_train()
            with open(regret_path, "w") as rf:
                for i in range(len(samples) + 1):

                    if ((i > 0 and i % 100 == 0) or (i == len(samples))):
                        incorrect_precent = self.evaluate()
                        with open(eval_path, "a") as ef:
                            ef.write("{}:{}\n".format(
                                i, incorrect_precent
                            ))
                    if (i == len(samples)):
                        print(" " * 20, end="\r")
                        break

                    print("Sample {}/{}".format(i + 1, len(samples)), end="\r")

                    sample = samples[i]
                    feature = features[i]
                    gold_dose = float(sample[meta.THERAPEUTIC_DOSE])
                    pred_dose = self.step(sample, feature, i + 1)
                    if (pred_dose is None):
                        continue

                    reward = self.calc_reward(gold_dose, pred_dose)
                    if (reward >= 0):
                        correct_sample += 1
                    total_sample += 1

                    rf.write("{}:{}\n".format(
                        i + 1, total_sample - correct_sample
                    ))

                rf.write("-END-\n")
                rf.write("total:{}\n".format(total_sample))
                rf.write("correct:{}\n".format(correct_sample))
                rf.write('correct_percent:{}\n'.format(
                    correct_sample / total_sample
                ))


    def evaluate(self):
        total_sample = 0
        correct_sample = 0
        for i in range(len(self.samples)):
            sample = self.samples[i]
            feature = self.features[i]
            gold_dose = float(sample[meta.THERAPEUTIC_DOSE])
            pred_dose = self.predict(sample, feature)
            if (pred_dose is None):
                continue
            reward = self.calc_reward(gold_dose, pred_dose)
            if (reward >= 0):
                correct_sample += 1
            total_sample += 1

        return (total_sample - correct_sample) / total_sample

    def action_to_range(self, action):
        if (action == 0):
            return meta.DOSE_LO
        if (action == 1):
            return meta.DOSE_MD
        return meta.DOSE_HI

    def calc_reward(self, gold, pred):
        if (pred == meta.DOSE_LO):
            return 0 if gold < 21 else -1
        if (pred == meta.DOSE_MD):
            return 0 if (21 <= gold and gold <= 49) else -1
        if (pred == meta.DOSE_HI):
            return 0 if gold > 49 else -1
        return -1
