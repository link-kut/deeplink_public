from numpy.random import multivariate_normal
from matplotlib.pyplot import plot

samples = multivariate_normal([-0.5, -0.5], [[1, 0],[0, 1]], 1000)
plot(samples[:, 0], samples[:, 1], '.')

