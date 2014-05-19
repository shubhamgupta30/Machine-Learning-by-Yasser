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

##########################################################################

# Get a misclassified point
def get_misclassified_point(training_set, w):
  for item in training_set:
    if classify(item[0],w) != item[1]:
      return item

# Modify w
def modify(w, (x,y)):
  return w + y*x

# PLA Algorithm
def pla(training_set, num_iters=1024):
  extra_info = []
  w = array([0, 0 , 0])
  extra_info.append(calculate_classification_error(training_set, w))
  iters = 0
  while num_iters > iters:
    try:
      (x,y) = get_misclassified_point(training_set, w)
    except TypeError:
      print "No more misclassified points after %d iterations" % (iters)
      return w, extra_info
    w = modify(w, (x,y))
    extra_info.append(calculate_classification_error(training_set, w))
    iters += 1
  return w, extra_info

##########################################################################

def calculate_classification_error(training_data, w):
  total = 0.0
  misclassified = 0.0
  for item in training_data:
    total += 1.0
    if classify(item[0], w) != item[1]:
      misclassified += 1.0
  return misclassified/total

def plot_error(extra_info):
  X = range(0, len(extra_info))
  pylab.plot(X, extra_info)

def calculate_Eout(f, g, num_iters):
  misclassified = 0.0
  for i in xrange(num_iters):
    (x,y) = random_point()
    point = array([1, x, y])
    if classify(point, f) != classify(point, g):
      misclassified += 1.0
  return misclassified/num_iters


def generate_and_solve_learning_instance(size, algo_iterations = 1024, eout_iters=1024):
  f = random_f()
  training_data = training_set_random(size, f)
  plot_data(training_data, f)
  w, extra_info = pla(training_data, algo_iterations)
  plot_line(w)
  pylab.show()
  plot_error(extra_info)
  pylab.show()
  print "Eout estimate %f" % calculate_Eout(f, w, eout_iters)
  return training_data, f

