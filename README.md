# WarfarinDoseRL
Reinforcement learning models for warfarin dose online estimation.

## Models
This repository currently implements the following warfarin dose estimation models.
### Fixed dose
Assign 5mg/day to each patient.  
See `fixed_dose.py`.
### Clinical Dosing
Estimate the warfarin dose by a linear combination of patient characteristics (height, weight, race, etc.).  
See `clinical_dosing.py`.
### LinUCB
LinUCB algorithm based on feature vector of length 2344 for each patient.  
See `lin_ucb.py`.
### Lasso Bandit
Lasso bandit algorithm based on feature vector of length 2344 for each patient.  
See `lasso_bandit.py`.  
Reference: H. Bastani and M. Bayati. Online decision-making with high-dimensional covariates. 2015.
