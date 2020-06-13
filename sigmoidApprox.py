import numpy as np


def sigmoid_exact(x):
  return 1 / (1 + np.exp(-x))


def sigmoid_approximation(x):
  return (1 / 2) + (x / 4) - (x**3 / 48) + (x**5 / 480)


for lil_number in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
  print("\nInput:" + str(lil_number))
  print("Exact Sigmoid:" + str(sigmoid_exact(lil_number)))
  print("Approx Sigmoid:" + str(sigmoid_approximation(lil_number)))
