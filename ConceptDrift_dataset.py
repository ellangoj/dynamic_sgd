from sklearn import *
import pandas as pd
from sklearn import preprocessing


def readData(file):
    names = []
    cat_count = 0
    num_count = 0
    mul_count = 0
    time_count = 0
    file_feat = 'AutoML3_input_data1/'+ file + '/' + file + '_feat.type'
    print(file_feat)
    with open(file_feat) as ftype:
        for line in ftype:
            if 'Categorical' in line:
                cat_count += 1
                names.append('Categorical' + str(cat_count))
            elif 'Numerical' in line:
                num_count += 1
                names.append('Numerical' + str(num_count))
            elif 'Multi-value' in line:
                mul_count += 1
                names.append('Multi_value' + str(mul_count))
            elif 'Time' in line:
                time_count += 1
                names.append('Time' + str(time_count))

    # print(names)
    file_train = 'AutoML3_input_data1/' + file + '/' + file + '_train1.data'

    data = pd.read_csv(file_train, sep=' ', header=None)
    data.columns = names

    for i in range(1, 8):
        num = 'Numerical' + str(i)
        x = data[[num]].values.astype(float)
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df_normalized = pd.DataFrame(x_scaled)
        data[num] = df_normalized
        # print(data.head())

    for i in range(1, 18):
        cat = 'Categorical' + str(i)
        data[cat] = pd.Categorical(data[cat])
        # print(data[cat].nunique())
        one_hot = pd.get_dummies(data[cat], prefix=cat)
        data = data.drop(cat, axis=1)
        data = data.join(one_hot)
        # print(data.dtypes)

    file_labels = 'AutoML3_input_data1/' + file + '/' + file + '_train1.solution'
    data_label = pd.read_csv(file_labels, names = ['label'], header = None)
    data = data.join(data_label)

    print(data.head())

# main

d = readData('B')