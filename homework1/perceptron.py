from random import choice
from numpy import array, dot, random
import pylab

# Sign Function
def sign(x):
  if x < 0: return -1
  if x ==0: return 0
  return 1

# Random point inside a square with vertices (+-1, +-1)
def random_point():
  return (random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0))

# Generate random target function f
def random_f():
  (x1,y1) = random_point()
  (x2,y2) = random_point()
  return array([x1*y2-x2*y1, y2-y1, x1-x2])

# Classifies a point accroding to a hypothesis function
def classify(h, x):
  return sign(dot(h,x))

# Get a random Training set
def training_set_random(size, h):
  if size <= 0:
    return []
  training_data = [0]*size
  for i in xrange(size):
    point = random_point()
    x = array([1.0, point[0], point[1]])
    training_data[i] = (x, classify(h=h, x=x))
  return training_data

# Plots a line in the bounded "unit" cube
def plot_line(f):
  # x axis?
  if f[1] == 0 and f[0] == 0:
    point1 = (-1,0)
    point2 = (1,0)
  # y axis?
  elif f[2] == 0 and f[0] == 0:
    point1 = (0,1)
    point2 = (0,-1)
  # parallel to x axis?
  elif f[1] == 0:
    point1 = (-1,-f[0]/f[2])
    point2 = (1, -f[0]/f[2])
  # parallel to y axis?
  elif f[2] == 0:
    point1 = (-f[0]/f[1], 1)
    point2 = (-f[0]/f[1], -1)
  else:
    point1 = (-1, (-f[0] + f[1])/f[2])
    point2 = (1, (-f[0] - f[1])/f[2])
  print point1, point2
  (x1, y1) = point1
  (x2, y2) = point2
  pylab.plot([x1,x2], [y1,y2])

# Plots the f and the training points
def plot_data(training_set, f):
  pylab.xlim(-1, +1)
  pylab.ylim(-1, +1)
  plot_line(f)
  X  = [item[0][1] for item in training_set]
  Y  = [item[0][2] for item in training_set]
  C  = [item[1] for item in training_set]
  pylab.scatter(X, Y, c=C, s=50, alpha=.5)
  pylab.show()

def generate_learning_instance(size):
  f = random_f()
  training_data = training_set_random(size, f)
  plot_data(training_data, f)
  return training_data, f

