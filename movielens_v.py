import numpy
import random
import matplotlib.pyplot as plt
import datasets
import models

if __name__ == "__main__":
    train_set, test_set, m, n = datasets.movielens100k()
    split = int(0.8*len(train_set))
    train_set, val_set = train_set[:split], train_set[split:]
    
    r = 10
    
    L = numpy.random.rand(m, r)
    R = numpy.random.rand(r, n)
    
    model = models.MatrixFactorization(L, R, models.Opt.SAGA)
    
    train_loss = []
    validation_loss = []
    
    steps = 10*len(train_set)
    step_size = 2e-2
    mu = 1e-1
    
    m = 0
    
    for s in xrange(steps):
        if (s % (steps/100) == 0):
            train_loss.append(model.reg_loss(train_set, mu))
            validation_loss.append(model.loss(val_set))
        
        #training_point = random.choice(train_set)
        
        if (s % 2 == 0 and m < len(train_set)):
            j = m
            m += 1
        else:        
            j = random.randrange(m)
        training_point = train_set[j]
        
        model.update_step(training_point, step_size, mu)
    

    plt.figure(1)
    plt.plot(train_loss, 'k.-')
    plt.xlabel('Time')
    plt.ylabel('Loss')
    plt.title('Average train loss')
    
    plt.figure(2)
    plt.plot(validation_loss, 'k.-')
    plt.xlabel('Time')
    plt.ylabel('Loss')
    plt.title('Average validation loss')
    
    plt.show()
