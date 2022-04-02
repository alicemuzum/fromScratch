import numpy as np
import math as m
import matplotlib.pyplot as plt

def euclidian(row1,row2):
    distance = 0
    for i in range(len(row1)):
        distance += (row1[i]-row2[i])**2
    return m.sqrt(distance)

N = 30
K = 3
# Dataset oluşturma
dataset = []
for i in range(N):
    if i < 10:
        dataset.append(np.random.randint(0,5,size=2))
    if i >= 10 and i < 20:
        dataset.append(np.random.randint(5,10,size=2))
    if i >= 20 and i < 30:
        dataset.append(np.random.randint(10,15,size=2))

plt.title("1")
plt.scatter(*zip(*dataset))

# Centerları yazdırma
centers = []
for k in range(K):
    centers.append(np.random.randint(0,15,size=2))

plt.scatter(*zip(*centers), marker="d")    
plt.show()

titleCount = 2
dists = []
c1 = []
c2 = []
c3 = []
# Centerları düzenleme
while True:
    
    # Uzaklık hesaplama 
    c1.clear()
    c2.clear()
    c3.clear()
    for dot  in dataset:
        for idx,center in enumerate(centers):
            
            dists.append((euclidian(dot,center),idx))
        
        mini = min(dists)
        if mini[1] == 0:
            c1.append(dot)
        if mini[1] == 1:
            c2.append(dot)
        if mini[1] == 2:
            c3.append(dot)
            
        dists.clear()
    
    newCenter1 = np.array([sum(i for i,j in c1) / len(c1) , sum(j for i,j in c1) / len(c1)])
    newCenter2 = np.array([sum(i for i,j in c2) / len(c2) , sum(j for i,j in c2) / len(c2)])
    newCenter3 = np.array([sum(i for i,j in c3) / len(c3) , sum(j for i,j in c3) / len(c3)])
  
    centers[0] = newCenter1
    centers[1] = newCenter2
    centers[2] = newCenter3
    
    plt.title(f'{titleCount}')
    titleCount += 1
    plt.scatter(*zip(*c1),c="blue")
    plt.scatter(*zip(*c2),c="red")
    plt.scatter(*zip(*c3),c="green")
    plt.scatter(*zip(*centers), marker="d",c="orange")
    plt.show()

    if titleCount > 10:
        break