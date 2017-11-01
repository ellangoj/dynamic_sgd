# How to use this code

# Loading the data
train_set, test_set, dimension = datasets.realsim()

# Initializing the model, specifying the hypothesis class and the optimization procedure to use
initial_parameters = numpy.random.rand(dimension)
model = models.LogisticRegression(initial_parameters, models.Opt.SGD)

# Training the model
for t in xrange(10*(len(train_set))):
    training_point = random.choice(train_set)
    model.update_step(training_point, step_size, regularization_constant)

# Measuring the performance of the model on a test set
test_loss = model.loss(test_set)
