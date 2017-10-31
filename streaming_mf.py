import numpy
import random
import time
import os
import matplotlib.pyplot as plt
import datasets
import models

MOVIELENS_STEP_SIZE = 5e-3
MOVIELENS_MU = 1e-1

"""
b: length of stream
arrivals: lambda
"""
def process(train_set, test_set, b, arrivals, rho, parameters):
    step_size = MOVIELENS_STEP_SIZE
    mu = MOVIELENS_MU
    
    loss = {
                'Inc': {'train': [0]*b, 'test': [0]*b, 'inctrain': [0]*12},
                'Unif': {'train': [0]*b, 'test': [0]*b, 'inctrain': [0]*12},
                'NearUnif': {'train': [0]*b, 'test': [0]*b, 'inctrain': [0]*12},
                'ExpDecay': {'train': [0]*b, 'test': [0]*b, 'inctrain': [0]*12},
                'MostRecent': {'train': [0]*b, 'test': [0]*b, 'inctrain': [0]*12},
                'Offline': {'train': [0]*b, 'test': [0]*b},
                'A': {'train': [0]*12, 'test': [0]*12, 'inctrain': [0]*12},
                'B': {'train': [0]*12, 'test': [0]*12, 'inctrain': [0]*12}                
           }
    
    S = 0   # S_i = train_data[0:S]
    T = 0   # for IncSAGA, the sample set is train_data[0:T]
    m = 0   # offline algorithm is DynaSAGA which samples from train_data[0:m]
    
    for time in xrange(b):
        loss['Inc']['train'][time] = parameters['Inc'].reg_loss(train_set, mu)
        loss['Inc']['test'][time] = parameters['Inc'].loss(test_set)
        loss['Unif']['train'][time] = parameters['Unif'].reg_loss(train_set, mu)
        loss['Unif']['test'][time] = parameters['Unif'].loss(test_set)
        loss['NearUnif']['train'][time] = parameters['NearUnif'].reg_loss(train_set, mu)
        loss['NearUnif']['test'][time] = parameters['NearUnif'].loss(test_set)
        loss['ExpDecay']['train'][time] = parameters['ExpDecay'].reg_loss(train_set, mu)
        loss['ExpDecay']['test'][time] = parameters['ExpDecay'].loss(test_set)
        loss['MostRecent']['train'][time] = parameters['MostRecent'].reg_loss(train_set, mu)
        loss['MostRecent']['test'][time] = parameters['MostRecent'].loss(test_set)
        loss['Offline']['train'][time] = parameters['Offline'].reg_loss(train_set, mu)
        loss['Offline']['test'][time] = parameters['Offline'].loss(test_set)
        
        S_prev = S
        S += arrivals
        
        #if time % (b/12 + 1) == 0:
            #t = time / (b/12 + 1)
            
            ## Algo A
            #mA = 0
            #for s in xrange(20*arrivals*time):
                #if (s % 2 == 0 and mA < S):
                    #j = mA
                    #mA += 1
                #else:
                    #j = random.randrange(mA)
                #parameters['A'][t].update_step(train_set[j], step_size, mu)
            
            ## Algo B
            #mB = 0
            #for s in xrange(rho*time):
                #if (s % 2 == 0 and mB < S):
                    #j = mB
                    #mB += 1
                #else:
                    #j = random.randrange(mB)
                #parameters['B'][t].update_step(train_set[j], step_size, mu)
                
            #loss['Inc']['inctrain'][t] = parameters['Inc'].reg_loss(train_set[:S], mu)
            #loss['Unif']['inctrain'][t] = parameters['Unif'].reg_loss(train_set[:S], mu)
            #loss['NearUnif']['inctrain'][t] = parameters['NearUnif'].reg_loss(train_set[:S], mu)
            #loss['ExpDecay']['inctrain'][t] = parameters['ExpDecay'].reg_loss(train_set[:S], mu)
            #loss['MostRecent']['inctrain'][t] = parameters['MostRecent'].reg_loss(train_set[:S], mu)
            #loss['A']['inctrain'][t] = parameters['A'][t].reg_loss(train_set[:S], mu)
            #loss['B']['inctrain'][t] = parameters['B'][t].reg_loss(train_set[:S], mu)            
            #loss['A']['train'][t] = parameters['A'][t].reg_loss(train_set, mu)
            #loss['A']['test'][t] = parameters['A'][t].loss(test_set)
            #loss['B']['train'][t] = parameters['B'][t].reg_loss(train_set, mu)
            #loss['B']['test'][t] = parameters['B'][t].loss(test_set)
        
        for s in xrange(rho):
            # IncSAGA
            if (s % 2 == 0 and T < S):
                j = T
                T += 1
            else:
                j = random.randrange(T)
            parameters['Inc'].update_step(train_set[j], step_size, mu)
            
            # Uniform
            j = random.randrange(S)
            parameters['Unif'].update_step(train_set[j], step_size, mu)
            
            # NearUniform
            if time == 0:
                j = random.randrange(S)
            else:
                if (random.random() > 0.5):
                    j = random.randrange(S_prev)
                else:
                    j = random.randrange(S_prev, S)
            parameters['NearUnif'].update_step(train_set[j], step_size, mu)
            
            # ExpDecay
            r = random.random()
            i = 1
            ss = 1./(1 << i)
            while (r > ss and i < time + 1):
                i += 1
                ss += 1./(1<<i)
            j = random.randrange(S - i*arrivals, S - (i-1)*arrivals)
            parameters['ExpDecay'].update_step(train_set[j], step_size, mu)            
            
            # MostRecent
            j = random.randrange(S_prev, S)
            parameters['MostRecent'].update_step(train_set[j], step_size, mu)
            
            # Offline
            j = random.randrange(len(train_set))
            #if (s % 2 == 0 and m < len(train_set)):
                #j = m
                #m += 1
            #else:
                #j = random.randrange(m)
            parameters['Offline'].update_step(train_set[j], step_size, mu)
                
    return loss

def ylims(o, split):
    end = [ o['Inc'][split][-1],
            o['Unif'][split][-1],
            o['NearUnif'][split][-1],
            o['MostRecent'][split][-1],
            o['ExpDecay'][split][-1],
            o['Offline'][split][-1]
          ]
    
    lower_bound = min(end) - 0.01
    
    beginning = [ o['Inc'][split][10],
                  o['Unif'][split][10],
                  o['NearUnif'][split][10],
                  o['MostRecent'][split][10],
                  o['ExpDecay'][split][10],
                  o['Offline'][split][10]
                ]
    
    upper_bound = max(beginning) + 0.01
    
    return (lower_bound, upper_bound)
        
def plot(output, rate, b, name):
    current_time = time.strftime('%Y-%m-%d_%H%M%S')
    path_png = os.path.join('output', current_time, 'png')
    path_eps = os.path.join('output', current_time, 'eps')
    os.makedirs(path_png)
    os.makedirs(path_eps)
    
    xx = range(1, b+1)
    xxx = range(1, b+1, (b/12 + 1))
    
    plt.figure(1)
    plt.clf()
    plt.plot(xx, output['Inc']['train'], 'g.-', label='IncSGD')
    plt.plot(xx, output['Unif']['train'], 'm.-', label='Uniform')
    plt.plot(xx, output['NearUnif']['train'], 'b.-', label='NearUniform')
    plt.plot(xx, output['MostRecent']['train'], 'r.-', label='MostRecent')
    plt.plot(xx, output['ExpDecay']['train'], 'y.-', label='ExpDecay')
    plt.plot(xx, output['Offline']['train'], 'k.-', label='Offline')
    plt.xlabel('Time')
    plt.ylabel('Average training loss')
    plt.title('{0} training loss, rho/lambda={1}, batch_size={2}'.format(name, rate, 1./b))
    plt.legend()
    plt.xlim(0, b)
    (lower, upper) = ylims(output, 'train')
    plt.ylim(lower, upper)
    plt.savefig(os.path.join(path_eps,'{0}_r{1}b{2}train.eps'.format(name, rate, b)), format='eps')
    plt.savefig(os.path.join(path_png,'{0}_r{1}b{2}train.png'.format(name, rate, b)), format='png', dpi=200)
    
    plt.figure(2)
    plt.clf()
    plt.plot(xx, output['Inc']['test'], 'g.-', label='IncSGD')
    plt.plot(xx, output['Unif']['test'], 'm.-', label='Uniform')
    plt.plot(xx, output['NearUnif']['test'], 'b.-', label='NearUniform')
    plt.plot(xx, output['MostRecent']['test'], 'r.-', label='MostRecent')
    plt.plot(xx, output['ExpDecay']['test'], 'y.-', label='ExpDecay')
    plt.plot(xx, output['Offline']['test'], 'k.-', label='Offline')
    plt.xlabel('Time')
    plt.ylabel('Average test loss')
    plt.title('{0} test loss, rho/lambda={1}, batch_size={2}'.format(name, rate, 1./b))
    plt.legend()
    plt.xlim(0, b)
    (lower, upper) = ylims(output, 'test')
    plt.ylim(lower, upper)
    plt.savefig(os.path.join(path_eps,'{0}_r{1}b{2}test.eps'.format(name, rate, b)), format='eps')
    plt.savefig(os.path.join(path_png,'{0}_r{1}b{2}test.png'.format(name, rate, b)), format='png', dpi=200)

if __name__ == "__main__":
    train_data, test_data, m, n = datasets.movielens1m()
    
    r = 10
    init_L = numpy.random.rand(m, r)
    init_R = numpy.random.rand(r, n)
    opt = models.Opt.SGD
    
    batches = [100]
    rates = [10]     # rho/lambda
    num_trials = 1
    
    for b in batches:
        for rate in rates:
            arrivals = len(train_data)/b
            rho = int(arrivals * rate)
            
            parameters = {}
            parameters['Inc'] = models.MatrixFactorization(init_L, init_R, opt)
            parameters['Unif'] = models.MatrixFactorization(init_L, init_R, opt)
            parameters['NearUnif'] = models.MatrixFactorization(init_L, init_R, opt)
            parameters['ExpDecay'] = models.MatrixFactorization(init_L, init_R, opt)
            parameters['MostRecent'] = models.MatrixFactorization(init_L, init_R, opt)
            parameters['Offline'] = models.MatrixFactorization(init_L, init_R, opt)
            parameters['A'] = [models.MatrixFactorization(init_L, init_R, opt) for x in xrange(12)]
            parameters['B'] = [models.MatrixFactorization(init_L, init_R, opt) for x in xrange(12)]
            
            output = process(train_data, test_data, b, arrivals, rho, parameters)
            
            # TODO: average outputs over num_trials
            
            # TODO: save output to a file
            
            plot(output, rate, b, 'mf')
