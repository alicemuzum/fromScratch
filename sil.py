import matplotlib.pyplot as plt
import numpy as np


def poly_func(x,y,b):
    plt.scatter(x,y,marker="o", s = 30)
    
    #Predicted response vector
    global y_pred
    y_pred = b[0] + b[1] * x
    
    #Plotting regression line
    plt.plot(x,y_pred,color="g")
    
    #☺Putting labels
    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.show()
    
    return

dataset_x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
dataset_y = np.array([1,2,3,2,4,6,4,7,5,6,8,10,12,11,10,9,11,15,16,15])

N = len(dataset_x)
X = np.c_[np.ones(N),dataset_x]
A = np.linalg.inv(X.T@X)
D = A @ X.T
result = D @ dataset_y

y_pred = []
poly_func(dataset_x, dataset_y, result)

#Mean Square Error
sqr = np.square(y_pred - dataset_y)
sqrError = np.sum(sqr) / N
print("Average error on the training set -> ", sqrError)




