import tkinter as tk
import numpy as np
import joblib
import tensorflow as tf
from PIL import ImageTk, Image
import cv2
#from tensorflow.keras import datasets
class App(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title('Nhan dang chu so viet tay')

            self.knn = joblib.load('knn_digit.pkl')
            self.index = None
            (X_train, y_train), (self.X_test, y_test) = tf.keras.datasets.mnist.load_data()
            



            self.cvs_digit = tk.Canvas(self, width = 280, height = 280,
                                            relief = tk.SUNKEN, border = 1)
            self.lbl_ket_qua = tk.Label(self, height = 10, relief = tk.SUNKEN, border = 1, font = ('Consolas', 12))
            btn_create_digit = tk.Button(self, text = 'Create Digit', width = 13, command = self.btn_create_digit_click)
            btn_recognition = tk.Button(self, text = 'Recognition', width = 13, command = self.btn_recognition_click)
            
            self.cvs_digit.grid(row = 0, column = 0, columnspan = 2, padx = 5, pady = 5)
            self.lbl_ket_qua.grid(row = 2, column = 0, columnspan = 2,  padx = 5, pady = 5, sticky = tk.EW) 
            btn_create_digit.grid(row = 1, column = 0, padx = 5, pady = 5, sticky = tk.W)
            btn_recognition.grid(row = 1, column = 1, padx = 5, pady = 5, sticky = tk.E)    

        def btn_create_digit_click(self):
            self.lbl_ket_qua.configure(text = '')
            self.index = np.random.randint(0,9999, 100)
            digit = np.zeros((28*10, 28*10), np.uint8)
            k =  0
            for x in range(0, 10):
                for y in range(0, 10):
                    digit[x*28:(x+1)*28, y*28:(y+1)*28] = self.X_test[self.index[k]]
                    k = k+1
            cv2.imwrite('digit.jpg', digit)
            image = Image.open('digit.jpg')
            img = image.resize((280,280), Image.ANTIALIAS)
            self.image_tk = ImageTk.PhotoImage(img)
            self.cvs_digit.create_image(0, 0, anchor = tk.NW, image = self.image_tk)
            
        def btn_recognition_click(self):
            digit_data = np.zeros((100, 28, 28), np.uint8)

            for i in range(0, 100):
                digit_data[i] = self.X_test[self.index[i]]

            RESHAPED = 784 # 28*28
            digit_data = digit_data.reshape(100, RESHAPED)
            predicted = self.knn.predict(digit_data)
            k = 0
            ket_qua = ''
            for x in range(0, 10):
                for y in range(0, 10):
                    ket_qua = ket_qua + '%3d' % predicted[k]
                    k = k+1
                ket_qua = ket_qua  + '\n'
            ket_qua = ket_qua[:-1]
            self.lbl_ket_qua.configure(text = ket_qua)
            
            
if __name__ == "__main__":
    app = App()
    app.mainloop()