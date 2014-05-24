import random
from math import floor
from numpy import array, dot, average
from numpy.linalg import pinv
import pylab
import sys
sys.path.insert(0, '../homework1')
import perceptron
from perceptron import sign

def create_training_set(size=100):
  f = perceptron.random_f()
  training_data = perceptron.training_set_random(size, f)
  X = array([item[0] for item in training_data])
  Y = array([item[1] for item in training_data])
  return X, Y, f

def solve_one_regression(
    size = 1000,
    plot_solu = True,
    calculate_eout=False,
    introduce_noise=False,
    noise_percent=0.1):
  X,Y,f = create_training_set(size)
  if introduce_noise:
    flip_ys(Y, noise_percent)
  g = pinv(X).dot(Y)
  if plot_solu:
    pylab.xlim(-1, +1)
    pylab.ylim(-1, +1)
    pylab.scatter([item[1] for item in X], [item[2] for item in X],
        c=Y, s=50, alpha=0.5)
    perceptron.plot_line(f)
    perceptron.plot_line(g)
    pylab.show()
    if calculate_eout: print "eout: %f" % get_eout(f,g)
  return g,compute_ein(X,Y,g)

def compute_ein(X, Y, g):
  total = len(X)
  misclassified = sum(
      [((Y[i] - perceptron.classify(g, X[i]))**2)/4
        for i in xrange(len(X))])
  return float(misclassified)/float(total)

def repeat(size=100, num_times=1000):
  ein_values = [solve_one_regression(size, False)[1] for i in xrange(num_times)]
  print ein_values
  return average(ein_values)

def get_eout(f, g, num_points=1000, num_iters=1000):
  eout_sum = 0.0
  for j in xrange(num_iters):
    misclassified = 0.0
    for i in xrange(num_points):
      point = perceptron.random_point() 
      point = array([1, point[0], point[1]])
      misclassified += (
          perceptron.classify(f, point) - perceptron.classify(g, point))**2/4
    eout_sum += float(misclassified)/float(num_points)
  return eout_sum/float(num_iters)


def flip_ys(Y, percent=0.1):
  size_to_flip = int(floor(len(Y) * percent))
  cords = random.sample(xrange(len(Y)), size_to_flip)
  for i in cords:
    Y[i] = -Y[i]
  
def ein_with_noise(size=1000, iters=1000, noise_percent=0.1):
  return average([nonlinear_linear_regression(
    size=size,
    plot_input=False,
    introduce_noise=True,
    noise_percent=noise_percent)[1] for i in xrange(iters)])

def nonlinear_linear_regression(size=1000,
    introduce_noise=False,
    noise_percent=0.1,
    plot_input=False):
  X = create_training_set(size)[0]
  Y = array([nonlinear_classification_fn(point[1], point[2]) for point in X])
  if introduce_noise:
    flip_ys(Y, percent=noise_percent)
  if plot_input:
    pylab.xlim(-1, +1)
    pylab.ylim(-1, +1)
    pylab.scatter([item[1] for item in X], [item[2] for item in X],
        c=Y, s=50, alpha=0.5)
    pylab.show()
  X_new = transform_X(X)
  g = pinv(X_new).dot(Y)
  return g, compute_ein(X_new,Y,g)

def compute_eout(g, size=1000, iters=1000):
  eout_sum = 0.0
  for j in xrange(iters):
    X = create_training_set(size)[0]
    Y = array([nonlinear_classification_fn(point[1], point[2]) for point in X])
    X = transform_X(X)
    flip_ys(Y, percent=0.1)
    misclassified = 0.0
    for i in xrange(len(X)):
      misclassified += (Y[i] - perceptron.classify(g, X[i]))**2/4
    eout_sum += float(misclassified)/float(size)
  return float(eout_sum)/float(iters)


def transform_X(X):
  X_new = [0]*len(X)
  for i in xrange(len(X)):
    (a,b,c) = X[i]
    X_new[i] = [a,b,c,b*c,b*b,c*c]
  return array(X_new)

def nonlinear_classification_fn(x1, x2):
  return sign(x1*x1 + x2*x2 - 0.6)
