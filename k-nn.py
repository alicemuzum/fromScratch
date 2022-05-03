import numpy as np
import math as m
import matplotlib.pyplot as plt
import timeit

start = timeit.default_timer()


def euclidian(row1,row2):
    distance = 0
    for i in range(len(row1)):
        distance += (row1[i]-row2[i])**2
    return m.sqrt(distance)

N = 90
K = 3
CLUSTER = 3
# Dataset oluşturma
dataset = []
for i in range(N):
    #dataset.append(np.random.randint(N/(i + 2),N,size=2))
    if i < N/CLUSTER:
        dataset.append([np.random.randint(0,10),np.random.randint(0,10),0])
    if i >= N/CLUSTER and i < N/CLUSTER * 2:
        dataset.append([np.random.randint(15,25),np.random.randint(15,25),1])
    if i >= N/CLUSTER * 2 and i < N:
        dataset.append([np.random.randint(0,10),np.random.randint(30,40),2])

# =============================================================================
#     if i == 0:
#         
#         plt.scatter([dataset[j][0] for j in range(0,m.floor(N/CLUSTER))],[dataset[i][1] for i in range(0,m.floor(N/CLUSTER))])
#     
#     if i == 1:
#         
#         plt.scatter([dataset[j][0] for j in range(m.floor(N/CLUSTER),m.floor(N/CLUSTER * 2))],[dataset[i][1] for i in range(m.floor(N/CLUSTER),m.floor(N/CLUSTER * 2))])
#         
#     if i == 2:
#             
#         plt.scatter([dataset[j][0] for j in range(m.floor(N/CLUSTER * 2),N)],[dataset[i][1] for i in range(m.floor(N/CLUSTER * 2),N)])
#         
# =============================================================================



inp = 1

while inp != "0":
    inp = input("1 for random point\n0 for exit: ")
    if inp == "1":
        
        for i in range(CLUSTER):
            plt.scatter([dataset[j][0] for j in range(len(dataset)) if dataset[j][2] == i ],[dataset[m][1] for m in range(len(dataset)) if dataset[m][2] == i])
        
        random_point = [np.random.randint(0,25),np.random.randint(0,40),CLUSTER+1]
        plt.scatter(random_point[0],random_point[1])
        plt.show()
        
        dists = []
        for dot in dataset:
            dists.append([euclidian(dot, random_point),dot[2]])
            
        #Sorting dists according to distances
        dists.sort(key=lambda x:x[0])
        
        #Checking nearest neighbors 
        c = [0,0,0]
        for neighbor in range(K):
            if dists[neighbor][1] == 0:
                c[0] += 1
            if dists[neighbor][1] == 1:
                c[1] += 1
            if dists[neighbor][1] == 2:
                c[2] += 1
        
                
        
        maxi = max(c)
        
        if maxi == c[0]:
            dataset.append([random_point[0],random_point[1],0])
        elif maxi == c[1]:
            dataset.append([random_point[0],random_point[1],1])
        elif maxi == c[2]:
            dataset.append([random_point[0],random_point[1],2])
            
        for i in range(CLUSTER):
            plt.scatter([dataset[j][0] for j in range(len(dataset)) if dataset[j][2] == i ],[dataset[m][1] for m in range(len(dataset)) if dataset[m][2] == i])
        plt.show()
        
    
stop = timeit.default_timer()

print('Time: ', stop - start)  