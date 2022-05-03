import numpy as np
import math as m
import matplotlib.pyplot as plt

def euclidian(row1,row2):
    distance = 0
    for i in range(len(row1)):
        distance += (row1[i]-row2[i])**2
    return m.sqrt(distance)

N = 60
K = 3
# Dataset oluşturma
dataset = []
for i in range(N):
    #dataset.append(np.random.randint(N/(i + 2),N,size=2))
    if i < 20:
        dataset.append(np.random.randint(0,15,size=2))
   
    if i >= 20 and i < 40:
        dataset.append(np.random.randint(20,40,size=2))

    if i >= 40 and i < 60:
        dataset.append(np.random.randint(45,60,size=2))

plt.title("1")
plt.scatter(*zip(*dataset))

# Centerları yazdırma
centers = []
for k in range(K):
    centers.append(np.random.randint(0,30,size=2))

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
    
    newCenters = []
    newCenters.append(np.array([sum(i for i,j in c1) / len(c1) , sum(j for i,j in c1) / len(c1)]))
    newCenters.append(np.array([sum(i for i,j in c2) / len(c2) , sum(j for i,j in c2) / len(c2)]))
    newCenters.append(np.array([sum(i for i,j in c3) / len(c3) , sum(j for i,j in c3) / len(c3)]))


# =============================================================================
#     for old ,newC in enumerate(newCenters):
#         if m.sqrt((centers[old] - newC)**2) < 1:
#             break
# =============================================================================
    
    centers[0] = newCenters[0]
    centers[1] = newCenters[1]
    centers[2] = newCenters[2]
    
    plt.title(f'{titleCount}')
    titleCount += 1
    plt.scatter(*zip(*c1),c="blue")
    plt.scatter(*zip(*c2),c="red")
    plt.scatter(*zip(*c3),c="green")
    plt.scatter(*zip(*centers), marker="d",c="orange")
    plt.show()

    if titleCount > K*2 :
        break
