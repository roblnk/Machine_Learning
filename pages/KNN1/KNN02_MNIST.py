import tensorflow as tf
import cv2
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph
from sklearn.metrics import accuracy_score
import joblib
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
RESHAPED = 784 # 28*28
X_train = X_train.reshape(60000, RESHAPED)
X_test = X_test.reshape(10000, RESHAPED)

res = train_test_split(X_train, y_train, 
                       train_size=0.8,
                       test_size=0.2,
                       random_state=1)

train_data, validation_data, train_labels, validation_labels = res 
knn = KNeighborsClassifier()
knn.fit(train_data, train_labels)

predicted = knn.predict(validation_data)
accuracy = accuracy_score(predicted, validation_labels)
print('Do chinh xac tren tap validation: %.0f%%' % (accuracy*100))

predicted = knn.predict(X_test)
accuracy = accuracy_score(predicted, y_test)
print('Do chinh xac tren tap test: %.0f%%' % (accuracy*100))

joblib.dump(knn, 'knn_digit.pkl')