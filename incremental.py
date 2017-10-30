import numpy
import random
import matplotlib.pyplot as plt
import datasets
import models

MOVIELENS100k_STEP_SIZE = 2e-2
MOVIELENS100k_MU = 1e-1

"""
Terminology
b: length of stream
arrivals: lambda
"""
def process(train_set, test_set, b, arrivals, rho, parameters):
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
    
    step_size = MOVIELENS100k_STEP_SIZE
    mu = MOVIELENS100k_MU
    
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
            ## TODO
            
            # MostRecent
            j = random.randrange(S_prev, S)
            parameters['MostRecent'].update_step(train_set[j], step_size, mu)
            
            # Offline
            if (s % 2 == 0 and m < len(train_data)):
                j = m
                m += 1
            else:
                j = random.randrange(m)
            parameters['Offline'].update_step(train_set[j], step_size, mu)  
            
        
        if time % (b/12 + 1) == 0:
            t = time / (b/12 + 1)
            
            # Algo A
            mA = 0
            for s in xrange(20*arrivals*(time+1)):
                if (s % 2 == 0 and mA < S):
                    j = mA
                    mA += 1
                else:
                    j = random.randrange(mA)
                parameters['A'][t].update_step(train_set[j], step_size, mu)
            
            # Algo B
            mB = 0
            for s in xrange(rho*(time+1)):
                if (s % 2 == 0 and mB < S):
                    j = mB
                    mB += 1
                else:
                    j = random.randrange(mB)
                parameters['B'][t].update_step(train_set[j], step_size, mu)
                
            loss['Inc']['inctrain'][t] = parameters['Inc'].reg_loss(train_set[:S], mu)
            loss['Unif']['inctrain'][t] = parameters['Unif'].reg_loss(train_set[:S], mu)
            loss['NearUnif']['inctrain'][t] = parameters['NearUnif'].reg_loss(train_set[:S], mu)
            loss['ExpDecay']['inctrain'][t] = parameters['ExpDecay'].reg_loss(train_set[:S], mu)
            loss['MostRecent']['inctrain'][t] = parameters['MostRecent'].reg_loss(train_set[:S], mu)
            loss['A']['inctrain'][t] = parameters['A'][t].reg_loss(train_set[:S], mu)
            loss['B']['inctrain'][t] = parameters['B'][t].reg_loss(train_set[:S], mu)            
            loss['A']['train'][t] = parameters['A'][t].reg_loss(train_set, mu)
            loss['A']['test'][t] = parameters['A'][t].loss(test_set)
            loss['B']['train'][t] = parameters['B'][t].reg_loss(train_set, mu)
            loss['B']['test'][t] = parameters['B'][t].loss(test_set)
    
    return loss

def ylims(o, split):
    end = [ o['Inc'][split][-1],
            o['Uniform'][split][-1],
            o['NearUniform'][split][-1],
            o['MostRecent'][split][-1],
            o['A'][split][-1],
            o['B'][split][-1]
          ]
    
    lower_bound = min(end) - 0.01
    
    beginning = [ o['Inc'][split][10],
                  o['Uniform'][split][10],
                  o['NearUniform'][split][10],
                  o['MostRecent'][split][10],
                  o['A'][split][1],
                  o['B'][split][1]              
                ]
    
    upper_bound = max(beginning) + 0.01
        
def plot(output, rate, b):
    xx = range(1, b+1)
    xxx = range(1, b+1, (b/12 + 1))
    
    plt.figure(1)
    plt.clf()
    plt.plot(xx, output['Inc']['train'], 'm.-', label='IncSAGA')
    plt.plot(xx, output['Unif']['train'], 'c.-', label='Uniform')
    plt.plot(xx, output['NearUnif']['train'], 'y.-', label='NearUniform')
    plt.plot(xx, output['MostRecent']['train'], 'g.-', label='MostRecent')
    plt.plot(xxx, output['A']['train'], 'r.-', label='Algo A')
    plt.plot(xxx, output['B']['train'], 'b.-', label='Algo B')
    plt.xlabel('Time')
    plt.ylabel('Average training loss')
    plt.title('mf training loss, rho/lambda={0}, batch_size={1}'.format(rate, 1./b))
    plt.legend()
    plt.xlim(0, b)
    (lower, upper) = ylims(output, 'train')
    plt.ylim(lower, upper)
    plt.savefig('output/eps/mf/s{0}b{1}train.eps'.format(rate, b), format='eps')
    plt.savefig('output/png/mf/s{0}b{1}train.png'.format(rate, b), format='png', dpi=200)
    
    plt.figure(2)
    plt.clf()
    plt.plot(xx, output['Inc']['test'], 'm.-', label='IncSAGA')
    plt.plot(xx, output['Unif']['test'], 'c.-', label='Uniform')
    plt.plot(xx, output['NearUnif']['test'], 'y.-', label='NearUniform')
    plt.plot(xx, output['MostRecent']['test'], 'g.-', label='MostRecent')
    plt.plot(xxx, output['A']['test'], 'r.-', label='Algo A')
    plt.plot(xxx, output['B']['test'], 'b.-', label='Algo B')
    plt.xlabel('Time')
    plt.ylabel('Average test loss')
    plt.title('mf test loss, rho/lambda={0}, batch_size={1}'.format(rate, 1./b))
    plt.legend()
    plt.xlim(0, b)
    (lower, upper) = ylims(output, 'test')
    plt.ylim(lower, upper)
    plt.savefig('output/eps/mf/s{0}b{1}test.eps'.format(rate, b), format='eps')
    plt.savefig('output/png/mf/s{0}b{1}test.png'.format(rate, b), format='png', dpi=200)

    xxxx = xxx[1:]
    plt.figure(3)
    plt.clf()
    plt.plot(xxxx, output['Inc']['inctrain'][1:], 'm.-', label='IncSAGA')
    plt.plot(xxxx, output['Unif']['inctrain'][1:], 'c.-', label='Uniform')
    plt.plot(xxxx, output['NearUnif']['inctrain'][1:], 'y.-', label='NearUniform')
    plt.plot(xxxx, output['MostRecent']['inctrain'][1:], 'g.-', label='MostRecent')
    plt.plot(xxxx, output['A']['inctrain'][1:], 'r.-', label='Algo A')
    plt.plot(xxxx, output['B']['inctrain'][1:], 'b.-', label='Algo B')
    plt.xlabel('Time')
    plt.ylabel('Average training loss on Si')
    plt.title('mf training loss, rho/lambda={0}, batch_size={1}'.format(rate, 1./b))
    plt.legend()
    plt.xlim(0, b)
    plt.savefig('output/eps/mf/s{0}b{1}inctrain.eps'.format(rate, b), format='eps')
    plt.savefig('output/png/mf/s{0}b{1}inctrain.png'.format(rate, b), format='png', dpi=200)    

if __name__ == "__main__":
    train_data, test_data, m, n = datasets.movielens100k()
    
    r = 5
    init_L = numpy.random.rand(m, r)
    init_R = numpy.random.rand(r, n)
    opt = models.Opt.SGD
    
    batches = [100]
    rates = [15]     # rho/lambda
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
            
            plot(output, rate, b)
