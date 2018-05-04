import numpy as np
import matplotlib.pyplot as plt

x = np.array([10, 20, 30, 40, 50, 100])

#y1 = np.array([0.7698, 0.824, 0.871, 0.892, 0.906,0.9465])
#y2 = np.array([0.5605, 0.6005, 0.598, 0.6205, 0.6145,0.6465])
#y3 = np.array([0.7710, 0.822, 0.874, 0.8915, 0.903, 0.955])
#plt.title('Random Forests')

#y1 = np.array([0.671, 0.686, 0.6905, 0.6995, 0.6915, 0.713])
#y2 = np.array([0.5255, 0.555, 0.5635, 0.583, 0.5915, 0.6425])
#y3 = np.array([0.663, 0.6645, 0.681, 0.7005, 0.697, 0.704])
#plt.title('kNN')

y1 = np.array([0.694, 0.735, 0.7435, 0.7435, 0.6425, 0.744])
y2 = np.array([0.5725,0.592, 0.6125, 0.619, 0.641, 0.661])
y3 = np.array([0.749, 0.81, 0.872, 0.8885, 0.9065, 0.945])
plt.title('Logistic Regression')

l1, = plt.plot(x,y1)
l2, = plt.plot(x,y2)
l3, = plt.plot(x,y3)

plt.xlabel('window size (ms)')
plt.ylabel('accuracy')
plt.ylim(0,1)
plt.legend([l1,l2,l3],['LFP signal features only', 'per-tetrode spike train features only', 'both feature sets'])
#plt.show()
plt.savefig('lr.png')
