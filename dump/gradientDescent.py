def gradient_descent (epoch,slope=1,intercept=0,learning_rate=0.01):
    for i in range(epoch):
        calc_slope=(-2*0.5*(1.4-(intercept+slope*0.5)))+(-2*2.9*(3.2-(intercept+slope*2.9))) + (-2*2.3*(1.9-(intercept+slope*2.3)))
        calc_intercept=(-2*(1.4-(intercept+slope*0.5)))+(-2*(3.2-(intercept+slope*2.9))) + (-2*(1.9-(intercept+slope*2.3)))
        slope -= (calc_slope*learning_rate)
        intercept -= (calc_intercept*learning_rate)
    return f'{i+1}.Step \nSlope: {slope}\nY intercept: {intercept}'


for i in [1,10,100,1000]:
    print(gradient_descent(i),"\n")