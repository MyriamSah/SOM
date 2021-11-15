# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 12:12:03 2021

@author: Myriam
"""

import pandas as pd
import numpy as np
from sklearn_som.som import SOM
import pickle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

X = pd.read_csv (r'C:\Users\Myriam\Excel_pr_python\Data_TRS_Ctrl.csv')
X = np.array(X)
print(X)

XSOM = SOM(m=8, n=8, dim=15)
Res = XSOM.fit(X)
predictions = XSOM.predict(X)

pca = PCA(n_components=2)
ND = pca.fit_transform(X)

print(ND.shape)

print(predictions)
cluster=[]

# print(ND[:,0].shape)

# for i in range(64):
#     for j in range(113):
#         if predictions[j] == i:
#             cluster.append(predictions[j])
#             j+=1
#     i+=1


with open('PSY3008','wb') as outfile:
    pickle.dump(XSOM,outfile)
        







