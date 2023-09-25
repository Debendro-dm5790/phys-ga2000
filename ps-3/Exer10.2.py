import numpy as np
from random import random
import matplotlib.pyplot as plt

'''
We implement the initial number of Bismuth-213, Thallium, Lead and 
Bismuth-209 atoms. These are stored in the variables NumBi213, NumTl,
NumPb, and NumBi209, respectively. The values stored in these variables
are updated as the code simulates the radioactive decay of Bismuth-213.
'''
NumBi213 = 10000
NumTl = 0
NumPb = 0
NumBi209 = 0

'''
We also implement the half lives of Bismuth-213, Thallium, and Lead in 
seconds. These constants are stored in the variables tauBi213, tauTl, and
tauPb. The timestep of 1 second is stored in the variable h. 
'''
tauBi213 = 46*60
tauTl = 2.2*60
tauPb = 3.3*60

h = 1.0

'''
The probabality of decays are stored in the variables probBi213Decays, 
probTlDecays, and probPbDecays. The probabilities that Bismuth-213 decays 
to Thallium and Lead are stored as constants in the variables BiToPbProb
and BiToTlProb.   
'''
probBi213Decays = 1 - 2**(-1*h/tauBi213)
probTlDecays = 1 - 2**(-1*h/tauTl)
probPbDecays = 1 - 2**(-1*h/tauPb)

BiToPbProb = 0.9791
BiToTlProb = 0.0209

timeMax = 20000

timePoints = np.arange(0.0, timeMax, h)

Bi213Points = np.float32(np.zeros(len(timePoints)))
TlPoints = Bi213Points.copy()
PbPoints = Bi213Points.copy()
Bi209Points = Bi213Points.copy()

'''
We use a for loop to stimulate the decay for 20000 seconds with a time step
of one second. At each time step we determine whether lead, thallium, and 
Bismuth-213 decay in that order and we update the values of NumBi213, NumTl,
NumPb, and NumBi209 and store it into arrays called Bi213Points, TlPoints,
PbPoints, and Bi209Points. These arrays store the populations of these 
nuclei across all timesteps. When the loop is finished, we plot the nuclei
amount over time.
'''

counter = 0
for t in timePoints:
    Bi213Points[counter] = NumBi213
    TlPoints[counter] = NumTl
    PbPoints[counter] = NumPb
    Bi209Points[counter] = NumBi209
    
    decayPb = 0
    decayTl = 0
    decayBi213toPb = 0
    decayBi213toTl = 0
    
    for i in range(NumPb):
        if random() < probPbDecays:
            decayPb += 1
            
    NumPb -= decayPb
    NumBi209 += decayPb
    
    for i in range(NumTl):
        if random() < probTlDecays:
            decayTl += 1
            
    NumTl -= decayTl 
    NumPb += decayTl
    
    for i in range(NumBi213):
        if random() < probBi213Decays:
            if random() < BiToPbProb:
                decayBi213toPb += 1
            else:
                decayBi213toTl += 1
                
    NumBi213 = NumBi213 - (decayBi213toPb + decayBi213toTl)
    NumPb += decayBi213toPb 
    NumTl += decayBi213toTl
    
    counter += 1
    
plt.plot(timePoints, Bi213Points, color = 'blue')
plt.plot(timePoints, TlPoints, color = 'green')
plt.plot(timePoints, PbPoints, color = 'red')
plt.plot(timePoints, Bi209Points, color = 'orange')
plt.xlabel('Time in Seconds')
plt.ylabel('Number of Atoms')
plt.title('Radioactive Decay Chain of Bi-213')
plt.legend(['Bismuth-213', 'Thallium-209', 'Lead-209', 'Bismuth-209'])
plt.savefig('Bismuth-213DecayChain.png')
plt.show()


plt.plot(timePoints, TlPoints, color = 'green')
plt.plot(timePoints, PbPoints, color = 'red')
plt.xlabel('Time in Seconds')
plt.ylabel('Number of Atoms')
plt.title('Zooming Into the Activity of Thallium-209 and Lead-209')
plt.legend(['Thallium-209', 'Lead-209'])
plt.savefig('Bismuth-213DecayChainZoomedIn.png')
plt.show()

plt.plot(timePoints, TlPoints, color = 'green')
plt.xlabel('Time in Seconds')
plt.ylabel('Number of Atoms')
plt.title('Zooming Into the Activity of Thallium-209')
plt.legend(['Thallium-209'])
plt.savefig('Bismuth-213DecayChainZoomedInThallium.png')
plt.show()