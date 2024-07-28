import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import skewnorm

#define TLS linear function
alpha = 0.05
beta = 0.75
TLS_line = lambda x: alpha + beta * x


    
