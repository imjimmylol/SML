import numpy as np
import matplotlib.pyplot as plt
import math
import time
from src.model import FiLMResNet2In, flatten_last
from src.normalizer import RunningMeanStd
from src.envpacker import packenv, packenv_batch
from src.utils import transition_ability_batched, update_v_history
import torch
from torch import nn
import torch.nn.functional as F

# Training configs
AGENTS = 50     # number of agents
LEARNING_RATE = 1e-3
TRAINING_STEPS = 30000
BATCH_SIZE = 10
DISPLAY_STEP = 1000 # For visualization
TRAIN_STEP_INTERVAL = 2 # Interval of steps between training episodes


# Bewley model parameters
theta = 1 # CRRA
beta = 0.975 # Discount factor
A = 1 # Technology parameter
alpha = 0.33 # Capital share of income
gamma = 2 # Inverse Frisch elasticity
########################################### (MiLF inputs)
r = 0.04 # Interest rate (Return to savings)
w = 1 # Wage rate (Return to labor)
delta = 0.06 # Depreciation rate of capital
TAX_PARAMS = {
    "tax_consumption": 0.065,          # Consumption tax (fixed)
    "tax_income": 0.2,                # Tax on labor income
    "income_tax_elasticity": 0.5,     # Elasticity of labor supply w.r.t. after-tax income
    "saving_tax_elasticity": 0.5,     # Elasticity of savings w.r.t. after-tax income
    "tax_saving": 0.1                 # Tax on interest income
}
###########################################
p = 2.2e-6
q = 0.99

# shock parameters
# log e' = rho_v * log e + sigma_v * epsilon, epsilon ~ N(0,1)
rho_v = 0.95 # persistence of ability shock
sigma_v = 0.2 # std of ability shock
v_bar = 1.5

# Bounds of shock 
v_min = math.exp(-2 * sigma_v  / math.sqrt(1-rho_v**2))
v_max = math.exp( 2 * sigma_v  / math.sqrt(1-rho_v**2))


def mean_across_agents(x): # Since agent number is fix, thus we can use mean instead of sum
    return torch.mean(x, dim=1, keepdim=True)


def calculate_price(a, v, h):
    a_aggregate, l_aggregate_effective = mean_across_agents(a), mean_across_agents(h*v)
    wage = A * (1-alpha) * ((a_aggregate/l_aggregate_effective) ** alpha)
    ret = A * alpha * (a_aggregate/l_aggregate_effective ** alpha)
    return wage, ret

def taxfunc(ibt, abt, taxparams=TAX_PARAMS):
    it = ibt - (1 - taxparams["tax_income"]) * (ibt**(1-taxparams["income_tax_elasticity"])/(1-taxparams["income_tax_elasticity"])) # individual after tax income
    at = abt - (1-taxparams["tax_saving"]/1-taxparams["saving_tax_elasticity"]) * (abt**(1-taxparams["saving_tax_elasticity"])) # individual after tax saving
    return it, at

def calculate_moneydisposable(wage, ret, v, h, a, delta, is_init=False):
    if is_init:
        ibt = wage * h * v   # individual before tax income
    else:
        ibt = wage * h * v + (1-delta+ret) * a   # individual before tax income

    it, at = taxfunc(ibt = ibt, abt=a)
    money_disposable = it + at

    return money_disposable, ibt


def output_transform(a, money_disposable):

    # The a here is saving rate coming from the NN output
    consumption = money_disposable * (1 - a)
    savings = money_disposable * a
    return consumption, savings


def laborfocloss(a, h, ibt, money_disposable, wage, v, taxparams=TAX_PARAMS):

    loss_foc =  -h ** (-gamma) + ((1-a)*money_disposable/(1+taxparams["tax_consumption"])) * \
        (wage * v) * (1 - taxparams["tax_income"]) * (ibt ** (-taxparams["income_tax_elasticity"]))
    
    return torch.abs(loss_foc)


state_dim = 2*AGENTS + 2 # two state variables for each agent + 2 individual variables
cond_dim = 5 # exogenous variables for all agents in all worlds(Batch)
model = FiLMResNet2In(state_dim=state_dim, cond_dim=cond_dim,
                        hidden_dim=128, num_res_blocks=3, output_dim=3, dropout=0.1)