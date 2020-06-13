import numpy as np
import time
import matplotlib.pyplot as plt

def nonlin(x, deriv=False):
	if(deriv == True):
	    return x*(1-x)
	return 1/(1+np.exp(-x))


X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

np.random.seed(1)

syn0 = 2*np.random.random((3, 4)) - 1
syn1 = 2*np.random.random((4, 1)) - 1
listofLoss = []
listofTime = []
# begin training
start = time.time()
for j in range(1000):

	# Feed forward through layers 0, 1, and 2
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))

    l2_error = y - l2
    end = time.time()
    if (j % 100) == 0:
        # print("Error:" + str(np.mean(np.abs(l2_error))))
        tim = end - start
        listofTime.append(tim)
        listofLoss.append(float(np.mean(np.abs(l2_error))))

    l2_delta = l2_error*nonlin(l2, deriv=True)

    l1_error = l2_delta.dot(syn1.T)

    l1_delta = l1_error * nonlin(l1, deriv=True)

    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)
    
print("Output After Training:")
print(l2)
# print(listofLoss)
plt.xlabel("Time in seconds")
plt.ylabel("Loss")
plt.plot(listofTime, listofLoss)
plt.show()
