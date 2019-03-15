import sys
from fixed_dose import FixedDose
from clinical_dosing import ClinicalDosing
from lin_ucb import LinUCB
from lasso_bandit import LassoBandit

if __name__ == "__main__":
    rounds = [30, 30, 30, 30]
    if (len(sys.argv) > 1):
        rounds = sys.argv[1].split(',')
        rounds = list(map(lambda r: int(r), rounds))

    data_path = "./data/warfarin.csv"

    fixed_dose_round = rounds[0]
    clinical_dosing_round = rounds[1]
    lin_ucb_round = rounds[2]
    lasso_bandit_round = rounds[3]

    if (fixed_dose_round > 0):
        fixed_dose = FixedDose(data_path, "./save/fixed_dose")
        fixed_dose.train(fixed_dose_round)
    if (clinical_dosing_round > 0):
        clinical_dosing = ClinicalDosing(data_path, "./save/clinical_dosing")
        clinical_dosing.train(clinical_dosing_round)
    if (lin_ucb_round > 0):
        lin_ucb = LinUCB(data_path, "./save/lin_ucb")
        lin_ucb.train(lin_ucb_round)
    if (lasso_bandit_round > 0):
        lasso_bandit = LassoBandit(data_path, "./save/lasso_bandit")
        lasso_bandit.train(lasso_bandit_round)
