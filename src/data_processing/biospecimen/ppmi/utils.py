import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn import metrics
from collections import defaultdict
from scipy import interp
from sklearn.metrics import roc_curve, auc, accuracy_score
import numpy as np

def convert_onehot_inplace(df, col_name):
    one_hot = pd.get_dummies(df[col_name])
    df = df.drop(col_name, axis = 1)
    df = df.join(one_hot)
    return df

def convert_normalize_inplace(df, col_name, type='min_max'):
    if type=='min_max':
        min_col, max_col = df[col_name].min(), df[col_name].max()
        return df[col_name].map(lambda x: (x-min_col)/ (max_col - min_col))
    else:
        mean_col, std_col = df[col_name].mean(), df[col_name].std()
        return df[col_name].map(lambda x: (x-mean_col)/ (std_col))

