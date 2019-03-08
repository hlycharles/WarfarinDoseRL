import meta

class FeatureEntry:
    def __init__(self, tag, required, default):
        self.tag = tag
        self.required = required
        self.default = default

        self.idx_map = dict()

    def add_sample(self, entry):
        if (len(entry) == 0 or entry == "NA"):
            return

        if (self.tag == meta.FEATURE_LIST_ENUM):
            options = self.split_list(entry)
            for o in options:
                if (o in self.idx_map):
                    continue
                idx = len(self.idx_map)
                self.idx_map[o] = idx

        elif (self.tag == meta.FEATURE_ENUM):
            if (entry in self.idx_map):
                return
            self.idx_map[entry] = len(self.idx_map)

    def parse(self, entry):
        empty = entry == "NA" or len(entry.strip()) == 0
        if (self.required and empty):
            return None

        if (self.tag == meta.FEATURE_NO):
            return []

        if (self.tag == meta.FEATURE_LIST_ENUM):
            options = self.split_list(entry)
            result = [0] * (len(self.idx_map) + 1)
            if (empty):
                result[-1] = 1
                return result
            for o in options:
                result[self.idx_map[o]] = 1
            return result

        if (self.tag == meta.FEATURE_ENUM):
            result = [0] * (len(self.idx_map) + 1)
            if (empty):
                result[-1] = 1
                return result
            result[self.idx_map[entry]] = 1
            return result

        if (self.tag == meta.FEATURE_RANGE):
            if (empty):
                return [0, 1]
            result_str = entry.split('-')[0].strip()
            result_str = result_str.split('+')[0].strip()
            return [float(result_str), 1]

        if (self.tag == meta.FEATURE_NUM):
            if (empty):
                return [0, 1]
            return [float(entry), 0]

        if (self.tag == meta.FEATURE_BIN):
            result = [0, 0]
            if (empty):
                result[-1] = 1
                return result
            result[0] = int(entry.strip())
            return result

        if (self.tag == meta.FEATURE_LIST_MAX):
            if (empty):
                return [0, 1]
            max_entry = entry.split(';')[-1].strip()
            max_entry = max_entry.split(' ')[-1].strip()
            max_entry = max_entry.split(',')[-1].strip()
            return [float(max_entry), 0]


    def split_list(self, entry):
        options = entry.split(';')
        options = list(map(
            lambda o: o.strip().lower(), options
        ))
        return options
