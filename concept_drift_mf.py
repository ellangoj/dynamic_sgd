import numpy
import random
import time
import os
import matplotlib.pyplot as plt
import datasets
import models

MOVIELENS_STEP_SIZE = 5e-3
MOVIELENS_MU = 1e-1
OPT = models.Opt.SAGA

def fresh_model(m, n, r):
    init_L = numpy.random.rand(m, r)
    init_R = numpy.random.rand(r, n)
    return models.MatrixFactorization(init_L, init_R, OPT)
    
def carryover_model(model):
    return models.MatrixFactorization(numpy.copy(model.L), numpy.copy(model.R), OPT)
    
def swap_labels(data):
    return [(i, j, 6-val) for (i, j, val) in data]

"""
b: length of stream
rate: rho/lambda
arrivals: lambda
W: window size, in multiples of lambda
"""
def process(b, rate):
    W = 10
    step_size=MOVIELENS_STEP_SIZE
    mu=MOVIELENS_MU
    r = 10
    
    train_set, test_set, m, n = datasets.movielens1m()
    
    arrivals = len(train_set)/b
    rho = int(arrivals * rate)
    
    train_set = train_set[:50*arrivals] + swap_labels(train_set[50*arrivals:])
    test_set_swap = swap_labels(test_set)
    
    loss = {
                'Aware': {'train': [0]*b, 'test': [0]*b},
                'Hop': {'train': [0]*b, 'test': [0]*b},
                'HopCarry': {'train': [0]*b, 'test': [0]*b},
                'Sliding': {'train': [0]*b, 'test': [0]*b},
                'HopKnown': {'train': [0]*5, 'test': [0]*5}
            }
    
    aware = fresh_model(m, n, r)
    
    hop1 = fresh_model(m, n, r)
    hop2 = fresh_model(m, n, r)
    hop = hop1
    
    hopc1 = fresh_model(m, n, r)
    hopc2 = fresh_model(m, n, r)
    hopc = hopc1
    
    sliding = fresh_model(m, n, r)
    
    S = 0   # S_i = train_data[0:S]
    # the sample set is train_data[T0:T1]
    aware_T0 = 0
    aware_T1 = 0
    hop1_T0 = 0
    hop1_T1 = 0
    hop2_T0 = 0
    hop2_T1 = 0
    
    for time in xrange(b):
        if time % 5 == 0:
            if (time/5) % 2 == 0:
                hop1_T0 = arrivals*time
                hop1_T1 = hop1_T0
                hop1 = fresh_model(m, n, r)
                hop = hop2
                hopc1 = carryover_model(hopc2)
                hopc = hopc2
            else:
                hop2_T0 = arrivals*time
                hop2_T1 = hop2_T0
                hop2 = fresh_model(m, n, r)
                hop = hop1
                hopc2 = carryover_model(hopc1)
                hopc = hopc1
        if time == 0:
            hop = hop1
            hopc = hopc1
        if time == 50:
            test_set = test_set_swap
            aware_T0 = arrivals*time
            aware_T1 = aware_T0
            aware = fresh_model(m, n, r)
        
        loss['Aware']['test'][time] = aware.loss(test_set)
        loss['Hop']['test'][time] = hop.loss(test_set)
        loss['HopCarry']['test'][time] = hopc.loss(test_set)
        loss['Sliding']['test'][time] = sliding.loss(test_set)
        # for train error, measure over last W points
        low = max(S - W*arrivals, 0)
        if time != 0:
            loss['Aware']['train'][time] = aware.reg_loss(train_set[low:S], mu)
            loss['Hop']['train'][time] = hop.reg_loss(train_set[low:S], mu)
            loss['HopCarry']['train'][time] = hopc.reg_loss(train_set[low:S], mu)
            loss['Sliding']['train'][time] = sliding.reg_loss(train_set[low:S], mu)
        if time >= 50 and time < 55:
            loss['HopKnown']['train'][time-50] = hop1.reg_loss(train_set[low:S], mu)
            loss['HopKnown']['test'][time-50] = hop1.loss(test_set)
        
        S += arrivals
        
        for s in xrange(rho):
            # Aware
            if (s % 2 == 0 and aware_T1 < S):
                j = aware_T1
                aware_T1 += 1
            else:
                j = random.randrange(aware_T0, aware_T1)
            aware.update_step(train_set[j], step_size, mu)
            
            # Hop1
            if (s % 2 == 0 and hop1_T1 < S):
                j = hop1_T1
                hop1_T1 += 1
            else:
                j = random.randrange(hop1_T0, hop1_T1)
            hop1.update_step(train_set[j], step_size, mu)
            hopc1.update_step(train_set[j], step_size, mu)

            # Hop2
            if (s % 2 == 0 and hop2_T1 < S):
                j = hop2_T1
                hop2_T1 += 1
            else:
                j = random.randrange(hop2_T0, hop2_T1)
            hop2.update_step(train_set[j], step_size, mu)
            hopc2.update_step(train_set[j], step_size, mu)
        
        # Sliding
        sliding = fresh_model(m, n, r)
        T0 = max(S - arrivals*W, 0)
        T1 = T0
        for s in xrange(rho*min(W, time+1)):
            if (s % 2 == 0 and T1 < S):
                j = T1
                T1 += 1
            else:
                j = random.randrange(T0, T1)
            sliding.update_step(train_set[j], step_size, mu)
    
    return loss

       
def plot(output, rate, b, name):
    current_time = time.strftime('%Y-%m-%d_%H%M%S')
    path_png = os.path.join('output', current_time, 'png')
    path_eps = os.path.join('output', current_time, 'eps')
    os.makedirs(path_png)
    os.makedirs(path_eps)
    
    xx = range(0, 100)
    xxx = range(0, 100, 5)
    xxxx = range(50, 55)
    
    plt.figure(1)
    plt.clf()
    plt.plot(xx, output['Aware']['train'], 'm.-', label='Aware')
    plt.plot(xx, output['Hop']['train'], 'r.-', label='Hopping')
    plt.plot(xx, output['Sliding']['train'], 'b.-', label='Sliding')
    plt.plot(xx, output['HopCarry']['train'], 'y.-', label='Hop w/ Carry')
    plt.plot(xxxx, output['HopKnown']['train'], 'g.-')
    for x in xxx: plt.axvline(x=x, color='k', linestyle=':')
    plt.axvline(x=50, color='k', linestyle='-')
    plt.xlabel('Time')
    plt.ylabel('Average training loss over last W points')
    plt.title('{0} training loss, rho/lambda={1}, batch_size={2}'.format(name, rate, 1./b))
    plt.legend()
    plt.xlim(0, b)
    plt.savefig(os.path.join(path_eps,'{0}_r{1}b{2}train.eps'.format(name, rate, b)), format='eps')
    plt.savefig(os.path.join(path_png,'{0}_r{1}b{2}train.png'.format(name, rate, b)), format='png', dpi=200)
    
    plt.figure(2)
    plt.clf()
    plt.plot(xx, output['Aware']['test'], 'm.-', label='Aware')
    plt.plot(xx, output['Hop']['test'], 'r.-', label='Hopping')
    plt.plot(xx, output['Sliding']['test'], 'b.-', label='Sliding')
    plt.plot(xx, output['HopCarry']['test'], 'y.-', label='Hop w/ Carry')
    plt.plot(xxxx, output['HopKnown']['test'], 'g.-')
    for x in xxx: plt.axvline(x=x, color='k', linestyle=':')
    plt.axvline(x=50, color='k', linestyle='-')
    plt.xlabel('Time')
    plt.ylabel('Average test loss')
    plt.title('{0} test loss, rho/lambda={1}, batch_size={2}'.format(name, rate, 1./b))
    plt.legend()
    plt.xlim(0, b)
    plt.savefig(os.path.join(path_eps,'{0}_r{1}b{2}test.eps'.format(name, rate, b)), format='eps')
    plt.savefig(os.path.join(path_png,'{0}_r{1}b{2}test.png'.format(name, rate, b)), format='png', dpi=200)

if __name__ == "__main__":
    b = 100
    rate = 2

    output = process(b, rate)
    
    plot(output, rate, b, 'mf')
