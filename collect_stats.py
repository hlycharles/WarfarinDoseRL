import numpy as np
import sys

class StatsCollector:
    def __init__(self, run_count, sample_count):
        self.run_count = run_count
        self.sample_count = sample_count

    def collect_file(self, path):
        result = [None] * self.sample_count
        with open(path, "r") as f:
            lines = f.readlines()
            lines = list(map(lambda l: l.strip(), lines))
        for line in lines:
            if (line.startswith("-END-")):
                break
            sep_index = line.find(':')
            sample_idx = int(line[:sep_index]) - 1
            sample_val = float(line[sep_index+1:])
            result[sample_idx] = sample_val

        return result

    def calculate_stats(self, vals, log_path, log_empty=True):
        means = [None] * self.sample_count
        margins = [None] * self.sample_count
        for i in range(self.sample_count):
            samples = []
            for r in range(self.run_count):
                if (vals[r][i] is not None):
                    samples.append(vals[r][i])
            if (len(samples) > 0):
                means[i] = np.sum(samples) / len(samples)
                std = np.std(samples)
                margins[i] = 1.96 * std / np.sqrt(len(samples))
        with open(log_path, "w") as f:
            for i in range(len(means)):
                if (means[i] is None and (not log_empty)):
                    continue
                f.write("{},{},{}".format(
                    i + 1,
                    "" if means[i] is None else means[i],
                    "" if margins[i] is None else margins[i]
                ))
                if (i <  len(means) - 1):
                    f.write("\n")


    def collect(self, data_path, log_path, log_empty=True):
        regrets = []
        incorrect = []
        for i in range(self.run_count):
            # collect regret data
            regret_path = "{}/regret_{}.txt".format(data_path, i + 1)
            regrets.append(self.collect_file(regret_path))
            # collect incorrect percentage data
            eval_path = "{}/eval_{}.txt".format(data_path, i + 1)
            incorrect.append(self.collect_file(eval_path))

        # generate stats
        regret_log_path = "{}_regret.csv".format(log_path)
        self.calculate_stats(regrets, regret_log_path, log_empty=log_empty)
        eval_log_path = "{}_eval.csv".format(log_path)
        self.calculate_stats(incorrect, eval_log_path, log_empty=False)


if __name__ == "__main__":
    collector = StatsCollector(8, 5528)
    if (len(sys.argv) >= 3):
        data_path = sys.argv[1]
        log_path = sys.argv[2]
        log_empty = True
        if (len(sys.argv) > 3):
            log_empty = int(sys.argv[3]) > 0
        collector.collect(data_path, log_path, log_empty=log_empty)
