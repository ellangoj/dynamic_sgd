import numpy
import csv

"""
Binary classification of UCI adult dataset
https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
"""
def a9a():
    n = 32561
    d = 124
    
    data = []
    
    with open('data/a9a_shuf.data') as file:
        i = 0
        for line in file:
            fields = line.strip().split(' ')
            label = int(fields[0])
            features = {0:1}
            for j in xrange(1, len(fields)):
                features[int(fields[j])] = 1
            data.append((i, features, label))
            i += 1
    assert len(data) == n
    
    n_train = int(0.9*n)
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
    
    with open('data/rcv_shuf.data') as file:
        i = 0
        for line in file:
            fields = line.strip().split(' ')
            label = int(fields[0])
            features = {0:1}
            for j in xrange(1, len(fields)):
                (index, val) = fields[j].split(':')
                features[int(index)] = float(val)
            data.append((i, features, label))
            i += 1
    assert len(data) == n
            
    n_train = int(0.9*n)
    train_data = data[:n_train]
    test_data = data[n_train:]
    
    return train_data, test_data, d
    
"""
Binary classification of UseNet articles from four discussion groups
https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
"""
def realsim():
    n = 72309
    d = 20959
    
    data = []
    
    with open('data/real-sim_shuf.data') as file:
        i = 0
        for line in file:
            fields = line.strip().split(' ')
            label = int(fields[0])
            features = {0:1}
            for j in xrange(1, len(fields)):
                (index, val) = fields[j].split(':')
                features[int(index)] = float(val)
            data.append((i, features, label))
            i += 1
    assert len(data) == n

    n_train = int(0.9*n)
    train_data = data[:n_train]
    test_data = data[n_train:]
    
    return train_data, test_data, d
    
"""
Matrix factorization of MovieLens-100k ratings
https://grouplens.org/datasets/movielens/
"""
def movielens100k():
    m = 943
    n = 1682
    
    data = []
    with open('data/ml-100k/u_shuf.data') as file:
        reader = csv.reader(file, delimiter='\t')
        for line in reader:
            data.append((int(line[0])-1, int(line[1])-1, int(line[2])))
    assert len(data) == 100000
    
    train_data = data[:90000]
    test_data = data[90000:]
    
    return train_data, test_data, m, n
    
"""
Matrix factorization of MovieLens-1M ratings
https://grouplens.org/datasets/movielens/
"""
def movielens1m():
    m = 6040
    n = 3952
    
    data = []
    with open('data/ml-1m/ratings_shuf.dat') as file:
        for line in file:
            fields = line.split('::')
            data.append((int(fields[0])-1, int(fields[1])-1, int(fields[2])))
    assert len(data) == 1000209
    
    train_data = data[:900000]
    test_data = data[900000:]
    
    return train_data, test_data, m, n
