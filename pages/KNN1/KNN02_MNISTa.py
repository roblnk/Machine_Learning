import tensorflow as tf
import numpy as np
import cv2
import joblib
knn = joblib.load('knn_digit.pkl')
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
'''
X_test co 10.000 mau
tao 100 so ngau nhien tu [0, 9999] de lay 100 mau ngau nhien trong x_test
'''
index = np.random.randint(0,9999, 100)
digit_data = np.zeros((100, 28, 28), np.uint8)

for i in range(0, 100):
    digit_data[i] = X_test[i]

RESHAPED = 784 # 28*28
digit_data = digit_data.reshape(100, RESHAPED)
predicted = knn.predict(digit_data)
k = 0
for x in range(0, 10):
    for y in range(0, 10):
        print('%2d' % predicted[k], end = '')
        k = k+1
    print()


digit = np.zeros((28*10, 28*10), np.uint8)
k =  0
for x in range(0, 10):
    for y in range(0, 10):
        digit[x*28:(x+1)*28, y*28:(y+1)*28] = X_test[index[k]]
        k = k+1
cv2.imshow(digit)
cv2.readkey()






