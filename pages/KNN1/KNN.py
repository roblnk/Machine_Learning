from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
N = 150
centers = [[2, 3], [5, 5], [1, 8]]
n_classes = len(centers)
data, labels = make_blobs(n_samples=N, 
                          centers=np.array(centers),
                          random_state=1)
nhom0 = []
nhom1 = []
nhom2 = []
for i in range(N):
    if labels[i] == 0:
        nhom0.append([data[i,0], data[i,1]])
    elif labels[i] == 1:
        nhom1.append([data[i,0], data[i,1]])
    else:
        nhom2.append([data[i,0], data[i,1]])
nhom0 = np.array(nhom0)
nhom1 = np.array(nhom1)
nhom2 = np.array(nhom2)


