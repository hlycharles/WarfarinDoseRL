import meta

"""
@param age_str string starting with age
@returns decade for the age
"""
def get_age_in_decade(age_str):
    if (len(age_str) < 2):
        return None

    age_start_str = age_str[:2]

    try:
        age = int(age_start_str) // 10
    except:
        age = None

    return age

"""
@param sample a complete warfarine dose sample
@returns True if the sample uses enzyme inducer, False otherwise
"""
def get_enzyme_inducer_status(sample):
    try:
        carbamazepine = int(sample[meta.CARBAMAZEP]) > 0
    except:
        carbamazepine = False

    try:
        phenytoin = int(sample[meta.PHENYTOIN]) > 0
    except:
        phenytoin = False

    try:
        rifampin = int(sample[meta.RIFAMPIN]) > 0
    except:
        rifampin = False

    use_enzyme_inducer = (carbamazepine or phenytoin) or rifampin

    return 1 if use_enzyme_inducer else 0

"""
@param medications current medications in use
@returns True if Amiodarone is used, False otherwise
"""
def get_amiodarone_status(medications):
    if (len(medications) == 0):
        return 0

    if (medications == "NA"):
        return 0

    target = "amiodarone"
    med_arr = medications.split(';')
    for med in med_arr:
        if (med.strip().lower() == target):
            return 1

    return 0
