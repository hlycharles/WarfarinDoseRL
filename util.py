import csv

# read samples from data file
def read_all_data(file_path):
    with open(file_path) as f:
        reader = csv.reader(f, delimiter=',')

        is_header = True
        subjects = []

        for row in reader:
            if (is_header):
                # skip the header
                is_header = False
                continue
            subjects.append(row)

        return subjects

# doses are in weekly basis
def doses_in_same_range(d1, d2):
    thresh_lo = 21
    thresh_hi = 49
    return (
        (d1 < thresh_lo and d2 < thresh_lo) or
        (
            d1 >= thresh_lo and
            d1 <= thresh_hi and
            d2 >= thresh_lo and
            d2 <= thresh_hi
        ) or
        (d1 > thresh_hi and d2 > thresh_hi)
    )
