from random import choice
from numpy import array, dot, random, average
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
  mis_pts = []
  for item in training_set:
    if classify(item[0],w) != item[1]:
      mis_pts.append(item)
  if mis_pts == []:
    return None
  return choice(mis_pts)

# Modify w
def modify(w, (x,y)):
  return w + y*x

# PLA Algorithm
def pla(training_set, num_iters=1024, run_to_completion=False):
  extra_info = []
  w = array([0, 0 , 0])
  extra_info.append(calculate_classification_error(training_set, w))
  iters = 1
  while run_to_completion or num_iters > iters:
    point = get_misclassified_point(training_set, w)
    if point == None:
      return w, extra_info, iters
    w = modify(w, point)
    extra_info.append(calculate_classification_error(training_set, w))
    iters += 1
  return w, extra_info, iters

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


def generate_and_solve_learning_instance(size, algo_iterations = 1024, run_to_completion=False,
    eout_iters=3000, draw_plots=True):
  f = random_f()
  training_data = training_set_random(size, f)
  w, extra_info, iters = pla(training_data, algo_iterations, run_to_completion)
  eout = calculate_Eout(f, w, eout_iters)
  if(draw_plots):
    plot_data(training_data, f)
    plot_line(w)
    pylab.show()
    plot_error(extra_info)
    pylab.show()
  return iters, eout

def get_average_values(n, num_iters):
  iters = [0]*num_iters
  eout = [0]*num_iters
  for i in xrange(num_iters):
    f = random_f()
    training_data = training_set_random(n, f)
    w, extra_info, iters[i] = pla(training_data, run_to_completion=True)
    eout[i] = calculate_Eout(f, w, num_iters=3000)
  return average(iters), average(eout)
  
