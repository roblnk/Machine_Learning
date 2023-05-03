import tkinter as tk
import numpy as np
from sklearn.datasets import make_blobs
import joblib

knn = joblib.load("knn.pkl")

class App(tk.Tk):
        def __init__(self):
            super().__init__()
            self.geometry('330x330')
            self.resizable(False, False)
            self.title('KNN')
            self.data = None
            self.labels = None
            self.N = 150
            self.tao_so_lieu()
            self.cvs_data = tk.Canvas(self, width = 300, height = 300, relief = tk.SUNKEN, border = 2, background = 'white')
            self.cvs_data.bind("<Button-1>", self.xu_ly_mouse)

            self.cvs_data.place(x = 5, y = 5)
            self.ve_so_lieu()
        

        def tao_so_lieu(self):
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
            self.data = data
            self.labels = labels

        def ve_so_lieu(self):
            nhom0 = []
            nhom1 = []
            nhom2 = []
            for i in range(self.N):
                if self.labels[i] == 0:
                    nhom0.append([self.data[i,0], self.data[i,1]])
                elif self.labels[i] == 1:
                    nhom1.append([self.data[i,0], self.data[i,1]])
                else:
                    nhom2.append([self.data[i,0], self.data[i,1]])
            nhom0 = np.array(nhom0)
            nhom1 = np.array(nhom1)
            nhom2 = np.array(nhom2)
            so_luong = nhom0.shape[0]
            for i in range(0, so_luong):
                x = nhom0[i,0]
                y = nhom0[i,1]
                x1 = x-1
                y1 = y-1
                x2 = x+1
                y2 = y+1
                p = [x1,y1,x2, y2]
                self.cvs_data.create_rectangle(p, fill = 'red', outline = 'red')
            so_luong = nhom1.shape[0]
            for i in range(0, so_luong):
                x = nhom1[i,0]
                y = nhom1[i,1]
                x1 = x-1
                y1 = y-1
                x2 = x+1
                y2 = y+1
                p = [x1,y1,x2, y2]
                self.cvs_data.create_rectangle(p, fill = 'green', outline = 'green')
            so_luong = nhom2.shape[0]
            for i in range(0, so_luong):
                x = nhom2[i,0]
                y = nhom2[i,1]
                x1 = x-1
                y1 = y-1
                x2 = x+1
                y2 = y+1
                p = [x1,y1,x2, y2]
                self.cvs_data.create_rectangle(p, fill = 'blue', outline = 'blue')

        def xu_ly_mouse(self, event):
            x = event.x
            y = event.y
            print(x, y)
            x1 = x-3
            y1 = y-3
            x2 = x+3
            y2 = y+3
            p = [x1,y1,x2, y2]
            self.cvs_data.create_rectangle(p, fill = 'cyan', outline = 'cyan')
            my_test_data = np.array([[x, y]])
            predicted = knn.predict(my_test_data)
            text_id = self.cvs_data.create_text(x+10,y, fill = 'cyan')
            s = str(predicted[0])
            self.cvs_data.itemconfig(text_id, text = s)
          


   