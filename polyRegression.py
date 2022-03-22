import matplotlib.pyplot as plt
import numpy as np


def linreg_func(x,y,b):
    plt.scatter(x,y,marker="o", s = 30)
    
    x_line = np.linspace(x.min(), x.max(),20)
    
    #Predicted response vector
    global y_pred
    y_pred = b[2] * np.square(x_line) + b[1] * x_line + b[0]
    
    regression_line = y_pred 
    
    #Plotting regression line
    plt.plot(x_line,regression_line,color="g")
    
    #☺Putting labels
    plt.title("TRAIN")
    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.show()
    
    return

def linreg_test(x,y,b):
    plt.scatter(x,y,marker="o", s = 30)
    
    x_line = np.linspace(x.min(), x.max(),20)
    
    #Predicted response vector
    global y_pred_test
    y_pred_test = b[2] * np.square(x_line) + b[1] * x_line + b[0]
    
    #Plotting regression line
    plt.plot(x_line,y_pred,color="g")
    
    #☺Putting labels
    plt.title("TEST")
    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.show()
    
    return

def find_error(y_pred,actual_y,phase,N):
    #Mean Square Error
    sqr = np.square(y_pred - actual_y)
    sqrError = np.sum(sqr) / N
    print(f"Average error on the {phase} set -> ", sqrError)
    
train_x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
train_y = np.array([1,2,3,2,4,6,7,7,5,6,8,10,9,8,6,6,5,3,2,0])
test_x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
test_y = np.array([0,2,4,5,3,2,1,3,6,7,8,9,10,9,9,10,11,12,11,14])

#Magic Math
N = len(train_x)
X = np.c_[np.ones(N),train_x,np.square(train_x)]
A = np.linalg.inv(X.T@X)
D = A @ X.T
coeffs = D @ train_y

y_pred = []
y_pred_test = []

linreg_func(train_x, train_y, coeffs)
find_error(y_pred, train_y,"train", N)

linreg_test(test_x, test_y, coeffs)
find_error(y_pred_test, test_y,"test", N)




