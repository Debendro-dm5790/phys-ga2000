import numpy as np
import matplotlib.pyplot as plt
import timeit

def madelungMesh(length):
    L = length
    
    start_time = timeit.default_timer()

    atomArray = np.array(range(-L, L + 1), dtype=np.float32)
    i, j, k = np.meshgrid(atomArray, atomArray, atomArray)

    Madelung = (-1)**(i + j + k)/np.sqrt(i**2 + j**2 + k**2)
    Madelung[(i == 0)*(j == 0)*(k == 0)] = 0
    Madelung = np.sum(Madelung)

    timeElapsed = timeit.default_timer() - start_time

    print("It takes " + str(timeElapsed) + " seconds to compute a Madelung constant of ")
    print(Madelung)
    print("")
    
    return Madelung
    
LArray = [10,20,30,40,50,60,70,80,90,100]
MList = np.array([])

for L in LArray:
    M = madelungMesh(L)
    MList = np.append(MList, M)
    
plt.plot(LArray, MList, 'o')
plt.xlabel('L, the Number of Atoms in All Directions')
plt.ylabel('Estimated Value of Madelung Constant')
plt.title('Approximating the Madelung Constant with Meshgrid')
plt.savefig('MadelungMesh.png')
plt.show()




