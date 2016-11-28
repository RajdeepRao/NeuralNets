import numpy

dataset = numpy.loadtxt("optdigits_raining.csv", delimiter=",")

X = dataset[:,0:64]
Y = dataset[:,64]

# create model
