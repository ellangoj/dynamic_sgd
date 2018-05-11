import numpy as np
import random
import time
import os
import matplotlib.pyplot as plt
import models
import collections
from collections import defaultdict
import math

A9A_STEP_SIZE = 5e-2
A9A_MU = 7e-4
RCV_STEP_SIZE = 5e-1
RCV_MU = 1e-5

def a9a():
    n = 32561
    d = 124

    data = []

    with open('a9a_shuf.data') as file:
        i = 0
        for line in file:
            fields = line.strip().split(' ')
            label = int(fields[0])
            features = {0: 1}
            for j in range(1, len(fields)):
                features[int(fields[j])] = 1
            data.append((i, features, label))
            i += 1
    assert len(data) == n

    n_train = int(0.9 * n)
    train_data = data[:n_train]
    test_data = data[n_train:]

    return train_data, test_data, d


"""
Binary classification of Reuters articles by topic
https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
"""


def rcv():
    n = 20242
    d = 47237

    data = []

    with open('rcv_shuf.data') as file:
        i = 0
        for line in file:
            fields = line.strip().split(' ')
            # label = int(fields[0])
            if int(fields[0]) == -1 :
                label = 0
            else:
                label = 1
            features = {0: 1}
            for j in range(1, len(fields)):
                (index, val) = fields[j].split(':')
                features[int(index)] = float(val)
            data.append((i, features, label))
            i += 1
    assert len(data) == n

    n_train = int(0.9 * n)
    train_data = data[:n_train]
    test_data = data[n_train:]

    return train_data, test_data, d

def poisson_generator(lambdaP):
    l = math.exp(-1 * lambdaP)
    k = 1
    p = np.random.uniform(0,1)
    while(p > l):
        k += 1
        p *= np.random.uniform(0,1)
    return k

def process(train_set, test_set, d, b, r):
    # step_size = RCV_STEP_SIZE
    # mu = RCV_MU
    step_size = A9A_STEP_SIZE
    mu = A9A_MU

    arrivals = len(train_set) / b ## lambda
    rho = int(r * arrivals)

    bb = b
    loss = {
        'Inc': {'train': [0] * bb, 'test': [0] * bb, 'time': [0]*bb, 'subopt': [0]*bb, 'sampleK': [0.0] * bb},
        'B' : {'train': [0] * bb, 'test': [0] * bb, 'time' : [0]*bb, 'subopt': [0]*bb},
        'A': {'train': [0] * bb, 'test': [0] * bb, 'time': [0]*bb}

    }
    iterNum = 3
    for i in range(iterNum):
        init_w = np.random.uniform(0, 1, d)

        parameters = {}
        parameters['Inc'] = models.LogisticRegression(init_w, models.Opt.SAGA)

        parameters['A'] = models.LogisticRegression(init_w, models.Opt.SAGA)

        parameters['B'] = models.LogisticRegression(init_w, models.Opt.SAGA)

        S = 0
        T = 0


        for t in range(0, b):

            S += poisson_generator(arrivals)
            S = min(S, len(train_set)-1)
            S_i = train_set[0:S]

            start = time.time() * 1000.0
            for s in range(rho):
                # Inc
                if (s % 2 == 0 and T < S):
                    T += 1
                    j = T
                else:
                    j = random.randrange(T)
                parameters['Inc'].update_step(train_set[j], step_size, mu)
            end = time.time() * 1000.0
            loss['Inc']['time'][t] += (end - start)

            temp_model_A = models.LogisticRegression(np.random.uniform(0, 1, d), models.Opt.SAGA)
            T_A = 0
            # Algo A : has unlimited computational power
            start = time.time() * 1000.0
            for s in range(20 * (rho + 1) * (t + 1)):
                if(s % 2 == 0 and T_A < S):
                    T_A += 1
                    j = T_A
                else:
                    j = random.randrange(T_A)
                temp_model_A.update_step(train_set[j], step_size, mu)
            parameters['A'].w = np.copy(temp_model_A.w)
            end = time.time() * 1000.0
            loss['A']['time'][t] += (end - start)

            temp_model_B = models.LogisticRegression(np.random.uniform(0, 1, d), models.Opt.SAGA)
            T_B = 0
            # Algo B : rho * i number of SGD steps
            start = time.time() * 1000.0
            for s in range((rho + 1) * (t + 1)):
                if (s % 2 == 0 and T_B < S):
                    T_B += 1
                    j = T_B
                else:
                    j = random.randrange(T_B)
                temp_model_B.update_step(train_set[j], step_size, mu)
            parameters['B'].w = np.copy(temp_model_B.w)
            end = time.time() * 1000.0
            loss['B']['time'][t] += (end - start)


            loss['Inc']['test'][t] += parameters['Inc'].loss(test_set)
            loss['B']['test'][t] += parameters['B'].loss(test_set)
            loss['A']['test'][t] += parameters['A'].loss(test_set)

            loss['Inc']['train'][t] += parameters['Inc'].reg_loss(S_i, mu)
            loss['B']['train'][t] += parameters['B'].reg_loss(S_i, mu)
            loss['A']['train'][t] += parameters['A'].reg_loss(S_i, mu)

            loss['Inc']['subopt'][t] += loss['Inc']['train'][t] - loss['A']['train'][t]
            loss['B']['subopt'][t] += loss['B']['train'][t] - loss['A']['train'][t]
            loss['Inc']['sampleK'][t] += (T * 1.) / (T_B)


    for (k, v) in loss.items():
        for (w, z) in loss[k].items():
            loss[k][w] = [i / iterNum for i in z]

    return loss


def plot(output, r, name, b):
    current_time = time.strftime('%Y-%m-%d_%H%M%S')
    path_png = os.path.join('output', current_time, 'png')
    path_eps = os.path.join('output', current_time, 'eps')
    os.makedirs(path_png)
    os.makedirs(path_eps)
    xx = range(0, b)

    plt.figure(1)
    plt.clf()
    plt.plot(xx, output['Inc']['train'], 'm.-', label='INCSAGA', linewidth = 0.6)
    plt.plot(xx, output['B']['train'], 'r.-', label='DYNASAGA_lim (B)', linewidth = 0.6)
    plt.plot(xx, output['A']['train'], 'b.-', label='DYNASAGA_inf (A)', linewidth = 0.6)
    plt.xlabel('Time')
    plt.ylabel('Average training loss')
    plt.title('{0}, training loss, rho/lambda={1}, batch_size={2}'.format(name, r, 1. / b))
    plt.legend()
    plt.xlim(1, b)
    plt.savefig(os.path.join(path_eps, '{0}_r{1}b{2}train.eps'.format(name, r, b)), format='eps')
    plt.savefig(os.path.join(path_png, '{0}_r{1}b{2}train.png'.format(name, r, b)), format='png', dpi=200)

    plt.figure(2)
    plt.clf()
    plt.plot(xx, output['Inc']['test'], 'm.-', label='INCSAGA', linewidth = 0.6)
    plt.plot(xx, output['B']['test'], 'r.-', label='DYNASAGA_lim (B)', linewidth = 0.6)
    plt.plot(xx, output['A']['test'], 'b.-', label='DYNASAGA_inf (A)', linewidth = 0.6)
    plt.xlabel('Time')
    plt.ylabel('Average test loss')
    plt.title('{0}, test loss, rho/lambda={1}, batch_size={2}'.format(name, r, 1. / b))
    plt.legend()
    plt.xlim(1, b)
    plt.savefig(os.path.join(path_eps, '{0}_r{1}b{2}test.eps'.format(name, r, b)), format='eps')
    plt.savefig(os.path.join(path_png, '{0}_r{1}b{2}test.png'.format(name, r, b)), format='png', dpi=200)

    plt.figure(3)
    plt.clf()
    plt.plot(xx, output['Inc']['time'], 'm.-', label='INCSAGA', linewidth = 0.6)
    plt.plot(xx, output['B']['time'], 'r.-', label='DYNASAGA_lim (B)', linewidth = 0.6)
    plt.plot(xx, output['A']['time'], 'b.-', label='DYNASAGA_inf (A)', linewidth = 0.6)
    plt.xlabel('Time')
    plt.ylabel('Training time (millisecond)')
    plt.title('{0}, training time, rho/lambda={1}, batch_size={2}'.format(name, r, 1. / b))
    plt.legend()
    plt.xlim(1, b)
    plt.yscale('log')
    plt.savefig(os.path.join(path_eps, '{0}_r{1}b{2}time.eps'.format(name, r, b)), format='eps')
    plt.savefig(os.path.join(path_png, '{0}_r{1}b{2}time.png'.format(name, r, b)), format='png', dpi=200)

    # plt.figure(4)
    # plt.clf()
    # plt.plot(xx, output['Inc']['subopt'], 'm.-', label='INCSAGA', linewidth = 0.6)
    # plt.plot(xx, output['B']['subopt'], 'r.-', label='DYNASAGA_lim (B)', linewidth = 0.6)
    # plt.xlabel('Time')
    # plt.ylabel('suboptimality ')
    # plt.title('{0}, suboptimality, rho/lambda={1}, batch_size={2}'.format(name, r, 1. / b))
    # plt.legend()
    # plt.xlim(1, b)
    # plt.savefig(os.path.join(path_eps, '{0}_r{1}b{2}time.eps'.format(name, r, b)), format='eps')
    # plt.savefig(os.path.join(path_png, '{0}_r{1}b{2}time.png'.format(name, r, b)), format='png', dpi=200)


    fig, ax1 = plt.subplots()
    ax1.plot(xx, output['Inc']['subopt'], 'm.-', label='INCSAGA', linewidth = 0.6)
    ax1.plot(xx, output['B']['subopt'], 'r.-', label='DYNASAGA_lim (B)', linewidth = 0.6)
    ax1.set_xlabel('time')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('suboptimality')
    ax2 = ax1.twinx()
    ax2.plot(xx, output['Inc']['sampleK'], 'g.-', label='DYNASAGA_lim (B)', linewidth = 0.6)
    ax2.set_ylabel('t_i/t_B', color='g')
    ax2.tick_params('y', colors='g')
    fig.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(path_eps, '{0}_r{1}b{2}suboptimality.eps'.format(name, r, b)), format='eps')
    plt.savefig(os.path.join(path_png, '{0}_r{1}b{2}suboptimality.png'.format(name, r, b)), format='png', dpi=200)
    plt.show()


if __name__ == "__main__":
    b = 100
    train_set, test_set, d = a9a()


    # for r in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    r = 5

    loss = process(train_set, test_set, d, b, r)


    plot(loss, r, 'a9a Poisson arrival', b)


