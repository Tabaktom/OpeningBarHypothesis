import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
a =[1, 2, 3, 4, 4, 4, 4, 4,2 ,2, 2, 2, 2, 1, 1, 1, 1, 1, 1]
x = OneHotEncoder(categories=set(a)).fit(a)
print(x)

