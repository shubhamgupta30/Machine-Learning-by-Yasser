from random import choice
from numpy import array, dot, random, average
from numpy.linalg import pinv
import pylab
import sys
sys.path.insert(0, '../homework1')
import perceptron

def create_training_set(size=100):
  f = perceptron.random_f()
  training_data = perceptron.training_set_random(size, f)
  X = array([item[0] for item in training_data])
  Y = array([item[1] for item in training_data])
  return X, Y, f

def solve_one_regression(size = 100, plot_solu = True):
  X,Y,f = create_training_set(size)
  w = pinv(X).dot(Y)
  if plot_solu:
    pylab.xlim(-1, +1)
    pylab.ylim(-1, +1)
    pylab.scatter([item[1] for item in X], [item[2] for item in X],
        c=Y, s=50, alpha=0.5)
    perceptron.plot_line(f)
    perceptron.plot_line(w)
    pylab.show()
  return compute_ein(X,f,w)

def compute_ein(X, f, g):
  total = len(X)
  misclassified = sum(
      [((perceptron.classify(f, point) - perceptron.classify(g, point))**2)/4
        for point in X])
  return float(misclassified)/float(total)

def repeat(size=100, num_times=1000):
  ein_values = [solve_one_regression(size, False) for i in xrange(num_times)]
  print ein_values
  return average(ein_values)
