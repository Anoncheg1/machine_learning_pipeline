import numpy as np
import pandas as pd
from exploring import describe
# ---------- Toy DataFrame -----------
# Gaussian distribution N(mu,sigma ^2) - [100]
x = np.random.rand(300)
sigma = 0.1  # mean
mu = 0  # standard deviation
gausian_distr = np.random.normal(mu, scale=sigma, size=x.shape[0])

# numbers 1-100 [100]
numbers = np.linspace(1, 100, 300).astype(int)

df = pd.DataFrame({"gausian": gausian_distr,
              "numbers": numbers,
              "str": ["sta" + str(x) for x in numbers],
              "str_sm": ["stb" + str(int(x % 2 == 0)) + str(int(x % 5 == 0)) for x in numbers],
              "binary": [int(x%10 == 0) for x in numbers]
                   })
describe(df)
print(df[7:11].head())

from exploring import frequency_analysis

# print(df.describe())

frequency_analysis(df, "binary")
