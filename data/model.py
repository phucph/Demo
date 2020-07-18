# univariate multi-step lstm for the power usage dataset
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import pyplot
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense
from keras.layers import LSTM


# split a univariate dataset into train/test sets
def split_dataset(data):
    # split into standard weeks
    train, test = data[1:-328], data[-328:-6]
    # restructure into windows of weekly data
    train = array(split(train, len(train) / 7))
    test = array(split(test, len(test) / 7))
    print(train.shape,test.shape)
    return train, test


# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):
    scores = list()
    # calculate an RMSE score for each day
    for i in range(actual.shape[1]):
        # calculate mse
        mse = mean_squared_error(actual[:, i], predicted[:, i])
        # calculate rmse
        rmse = sqrt(mse)
        # store
        scores.append(rmse)
        # calculate overall RMSE
        s = 0
        for row in range(actual.shape[0]):
            for col in range(actual.shape[1]):
                s += (actual[row, col] - predicted[row, col]) ** 2
        score = sqrt(s / (actual.shape[0] * actual.shape[1]))
    return score, scores


# summarize scores
def summarize_scores(name, score, scores):
    s_scores = ', '.join(['%.1f' % s for s in scores])

    print('%s: [%.3f] %s' % (name, score, s_scores))


# convert history into inputs and outputs
def to_supervised(train, n_input, n_out=7):
    # flatten data
    data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))
    X, y = list(), list()
    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # ensure we have enough data for this instance
        if out_end < len(data):
            x_input = data[in_start:in_end, :]
            X.append(x_input)
            y_input = data[in_end:out_end, 0]
            y.append(y_input)
        # move along one time step
        in_start += 1
    return array(X), array(y)


# train the model
def build_model(train, n_input):
    # prepare data
    train_x, train_y = to_supervised(train, n_input)
    print(train_x.shape,train_y.shape)
    # print(train_x[:5])
    # define parameters
    verbose, epochs, batch_size = 2, 130, 16
    # define model
    model = Sequential()
    model.add(LSTM(200, activation='relu',return_sequences=True, input_shape=(train_x.shape[1], train_x.shape[2])))
    model.add(LSTM(100, activation='relu'))
    model.add(Dense(50))
    model.add(Dense(train_y.shape[1]))
    model.compile(loss='mse', optimizer='adam')
    # fit network
    model.fit(train_x, train_y, epochs=epochs, verbose=verbose)
    return model


# make a forecast
def forecast(model, history, n_input):
    # flatten data
    data = array(history)
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
    # retrieve last observations for input data
    input_x = data[-n_input:, :]
    print(input_x.shape)
    # reshape into [1, n_input, 1]
    input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
    # forecast the next week
    yhat = model.predict(input_x, verbose=0)

    # we only want the vector forecast
    yhat = yhat[0]
    return yhat


# evaluate a single model
def evaluate_model(train, test, n_input):
    # fit model
    model = build_model(train, n_input)
    # history is a list of weekly data
    history = [x for x in train]
    # walk-forward validation over each week
    predictions = list()
    for i in range(len(test)):
        # predict the week
        yhat_sequence = forecast(model, history, n_input)
        # store the predictions
        predictions.append(yhat_sequence)
        # print(yhat_sequence)
        # print("aaaaa")
        # print(test[i,:])
        # get real observation and add to history for predicting the next week
        history.append(test[i, :])
        # evaluate predictions days for each week
    predictions = array(predictions)
    # print(predictions[:5])
    # print(test[:, :, 0][:5])
    print(r2_score(test[:, :, 0], predictions))
    score, scores = evaluate_forecasts(test[:, :, 0], predictions)
    return score, scores


# load the new file
dataset = read_csv('household_power_consumption_days.csv', header=0,
                   infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
# split into train and test
sc = MinMaxScaler()
label_sc = MinMaxScaler()
# tital = df['TiDal'].values
# df = df.drop('TiDal',axis=1)
# df = df.drop('MocHoa',axis=1)
#df = df.drop('TanAn',axis=1)
data = sc.fit_transform(dataset.values)
label_sc.fit(dataset.iloc[:, 0].values.reshape(-1, 1))
train, test = split_dataset(data)
# evaluate model and get scores
n_input = 14
score, scores = evaluate_model(train, test, n_input)
# summarize scores
summarize_scores('lstm', score, scores)
# plot scores
days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
print(days)
pyplot.plot(days, scores, marker='o', label='lstm')
pyplot.show()
