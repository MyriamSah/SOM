# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 09:56:32 2021

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

with open('PSY3008', 'rb') as infile:
    XSOM = pickle.load(infile)


# Prediction de cluster pour chaque données dans X
predictions = XSOM.predict(X)

# PCA pour la visualisation
pca = PCA(n_components=2)
ND = pca.fit_transform(X)

# Distance euclidienne de chaque observations initial avec chaque cluster center
Res= XSOM.fit_transform(X)

# concatenation des observations transformé avec PCA et leur clusters prédits
pred=predictions[:,np.newaxis]
print(pred.shape)
New_ND=np.concatenate((ND,pred),axis=1)
    
S = np.arange(64)

S = np.zeros_like(S)
#print('Ordre', X[0], X[1], X[2], X[3], X[4] )
for i in range(113):
      x = pred[i]
      S[x[0]]=S[x[0]]+1
      i= i+1

S = S.reshape((8,8))
S = S.T
print("S", S)



fig, ax = plt.subplots()
im = ax.imshow(S, cmap="RdPu")

for i in range(8):
    for j in range(8):
        text = ax.text(j, i, S[i, j],
                        ha="center", va="center", color="k")
        
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel(ylabel="Sample Hits", rotation=-90, va="bottom")
ax.invert_yaxis()
plt.savefig('Sample_Hits.png', transparent=True)
plt.show()

