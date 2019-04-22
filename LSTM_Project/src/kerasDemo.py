import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from pandas import DataFrame
from pandas import concat
from itertools import chain
import matplotlib.pyplot as plt


def cut_data_set(dataset, chunk_size_x, label_size_y):
    chunk_x = []
    label_y = []
    for i in range(chunk_size_x, len(dataset)):
        # numpy.array slice method [first_row: last_row, column_num] 对数据进行切分操作
        chunk_x.append(dataset[i - chunk_size_x: i, 0])
        label_y.append(dataset[i: i + label_size_y, 0])
    train_x = np.array(chunk_x)
    label_y = np.array(label_y)
    return train_x, label_y


def get_train_valid_test_set(url, chunk_size_x):
    data_frame = pd.read_csv(url)
    data_set = data_frame.iloc[:, 1:2].values
    sc = MinMaxScaler(feature_range=(0, 1))
    train_data_set = sc.fit_transform(data_set)
    reframed_train_data_set = np.array(series_to_supervised(train_data_set, chunk_size_x, 1).values)

    train_size = int(len(reframed_train_data_set) * 0.67)
    # 数据集划分,选取前1/3天的数据作为训练集,中间1/4天作为验证集,其余的作为测试集
    train_days = train_size
    valid_days = int(len(reframed_train_data_set) * 0.25)

    train = reframed_train_data_set[:train_days, :]
    valid = reframed_train_data_set[train_days:train_days + valid_days, :]

    test_data = train_data_set[train_days + valid_days + chunk_size_x:, :]
    test = reframed_train_data_set[train_days + valid_days:, :]

    train_x, train_y = train[:, :-1], train[:, -1]
    valid_x, valid_y = valid[:, :-1], valid[:, -1]

    test_x, test_y = test[:, :-1], test[:, -1]

    # 将数据集重构为符合LSTM要求的数据格式,即 [样本，时间步，特征]
    train_x = train_x.reshape((train_x.shape[0], chunk_size_x, 1))
    valid_x = valid_x.reshape((valid_x.shape[0], chunk_size_x, 1))
    test_x = test_x.reshape((test_x.shape[0], chunk_size_x, 1))

    return train_x, train_y, valid_x, valid_y, test_x, test_y, test_data


"""
Frame a time series as a supervised learning dataset.
Arguments:
    data: Sequence of observations as a list or NumPy array.
    n_in: Number of lag observations as input (X).
    n_out: Number of observations as output (y).
    dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
    Pandas DataFrame of series framed for supervised learning.
"""


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def get_reframed_train_data_set(url, chunk_size_x):
    data_frame = pd.read_csv(url)
    data_set = data_frame.iloc[:, 1:2].values
    sc = MinMaxScaler(feature_range=(0, 1))
    train_data_set = sc.fit_transform(data_set)
    reframed_train_data_set = np.array(series_to_supervised(train_data_set, chunk_size_x, 1).values)
    return reframed_train_data_set


def get_correlation(test_data, predict_data):
    ans = np.corrcoef(np.array(test_data), np.array(predict_data))
    return ans


def get_spearman_correlation(test_data, predict_data):
    df2 = pd.DataFrame({'real': test_data, 'prediction': predict_data})
    return df2.corr('spearman')


def plot_img(reframed_train_data_set, train_predict, valid_predict, test_predict):
    plt.figure(figsize=(24, 8))
    plt.plot(reframed_train_data_set[:, -1], c='b')
    plt.plot([x for x in train_predict], c='g')
    plt.plot([None for _ in train_predict] + [x for x in valid_predict], c='y')
    plt.plot([None for _ in train_predict] + [None for _ in valid_predict] + [x for x in test_predict], c='r')
    plt.legend()
    plt.show()


def lstm_model(train_x, label_y, valid_x, valid_y, test_x, test_y, input_epochs, input_batch_size, test_data, chunk_size_x):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2])))

    # todo 可能原因是精确度低 没达到需要dropout的拟合条件，需要对accuracy进一步研究
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.5))

    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.5))

    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='linear'))

    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(train_x, label_y, epochs=input_epochs, batch_size=input_batch_size,
              validation_data=(valid_x, valid_y), verbose=2, shuffle=False)

    # prediction
    train_predict = model.predict(train_x)
    valid_predict = model.predict(valid_x)
    test_predict = model.predict(test_x)

    test_data_list = list(chain(*test_data))
    test_predict_list = list(chain(*test_predict))

    reframed_train_data_set = \
        get_reframed_train_data_set(url=r'/Users/wzc/Desktop/LSTM_experiment/15demo.csv', chunk_size_x=chunk_size_x)

    # res = model.fit(train_x, label_y, epochs=input_epochs, batch_size=input_batch_size,
    #                     validation_data=(valid_x, valid_y), verbose=2, shuffle=False)
    # plt.plot(res.history['loss'], label='train')
    # plt.plot(res.history['val_loss'], label='valid')
    print(model.summary())
    print(get_correlation(test_data_list, test_predict_list))
    print(get_spearman_correlation(test_data_list, test_predict_list))
    plot_img(reframed_train_data_set, train_predict, valid_predict, test_predict)


def main():
    # epochs = 100 batch_size = 50
    chunk_size_x = 3
    input_epochs = 10
    input_batch_size = 100
    train_x, label_y, valid_x, valid_y, test_x, test_y, test_data = \
        get_train_valid_test_set(url=r'/Users/wzc/Desktop/LSTM_experiment/15demo.csv', chunk_size_x=chunk_size_x)

    lstm_model(train_x, label_y, valid_x, valid_y, test_x, test_y,
               input_epochs, input_batch_size, test_data, chunk_size_x=chunk_size_x)


if __name__ == '__main__':
    main()
