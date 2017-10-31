import numpy
import random
import matplotlib.pyplot as plt
import datasets
import models

if __name__ == "__main__":
    train_set, test_set, m, n = datasets.movielens1m()
    #train_set, test_set, d = datasets.realsim()
    
    split = int(0.5*len(train_set))
    train_set, val_set = train_set[:split], train_set[split:]
    
    r = 10
    L = numpy.random.rand(m, r)
    R = numpy.random.rand(r, n)
    #w = numpy.random.rand(d)
    
    steps = 1*len(train_set)
    step_size = 1e-2
    mu = 1e-1
    
    train_loss = []
    validation_loss = []
    
    #for mu in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 0]:
    for step_size in [5e-2, 2e-2, 1e-2, 5e-3, 1e-3]:
            
        model = models.MatrixFactorization(L, R, models.Opt.SAGA)
        #model = models.LogisticRegression(w, models.Opt.SGD)
        
        m = 0

        for s in xrange(steps):
            #if (s % (steps/50) == 0):
                #train_loss.append(model.reg_loss(train_set, mu))
                #validation_loss.append(model.loss(val_set))
            
            if model.opt == models.Opt.SGD:
                training_point = random.choice(train_set)
            elif model.opt == models.Opt.SAGA:
                if (s % 2 == 0 and m < len(train_set)):
                    j = m
                    m += 1
                else:        
                    j = random.randrange(m)
                training_point = train_set[j]
            
            model.update_step(training_point, step_size, mu)
            
        print step_size, mu, model.loss(val_set)

    #plt.figure(1)
    #plt.plot(train_loss, 'k.-')
    #plt.xlabel('Time')
    #plt.ylabel('Loss')
    #plt.title('Average train loss')
    
    #plt.figure(2)
    #plt.plot(validation_loss, 'k.-')
    #plt.xlabel('Time')
    #plt.ylabel('Loss')
    #plt.title('Average validation loss')
    
    #plt.show()
