from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib
knn = joblib.load("knn_digit.pkl")
np.random.seed(100)
N = 150
centers = [[2, 3], [5, 5], [1, 8]]
n_classes = len(centers)
data, labels = make_blobs(n_samples=N, 
                          centers=np.array(centers),
                          random_state=1)
min = np.min(data, 0)
x_min = min[0]
y_min = min[1]

max = np.max(data, 0)
x_max = max[0]
y_max = max[1]

for i in range(0, N):
    x = data[i][0]
    y = data[i][1]
    x_moi = (x - x_min)/(x_max - x_min)*300
    y_moi = (y - y_min)/(y_max - y_min)*300
    data[i][0] = x_moi
    data[i][1] = y_moi

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





