import numpy as np
import matplotlib.pyplot as plt

x_ = np.array([0,250])
x = np.array([10, 20, 50, 100, 150, 200, 250]) #300, 400, 500])
y = np.array([2396.48, 2367.70, 2363.41, 2488.29, 2331.46, 2315.81, 2388.94]) #3885.12, 3671.89, 3186.57])

#random = np.array([3249.98,3249.98])
random = np.array([2848.45,2848.45])
prior = np.array([2381.95,2381.95])

plt.xlabel('window size (ms)')
plt.ylabel('mean squared error')
l1, = plt.plot(x,y, 'b-')
l2, = plt.plot(x_,random, 'r--')
l3, = plt.plot(x_,prior, 'k--')
plt.legend([l1,l2,l3],['Posterior', 'Random Guess', 'Prior'])
#plt.show()
plt.savefig('ex1.png')
