import numpy as np
# import requests
from pandas._libs import json
from sklearn.preprocessing import MinMaxScaler

# df = pd.read_csv('VCD_data_1.csv', parse_dates=[0])
sc = MinMaxScaler()
label_sc = MinMaxScaler()

def split_data(df, lookback, pred):

    date = df["Date"].values
    df = df.drop('Date', axis=1)
    df = df.drop('TiDal', axis=1)
    df = df.drop('MocHoa', axis=1)
    df = df.drop('TanAn', axis=1)
    # print(df[0:5])
    data = sc.fit_transform(df.values)
    # print("b")
    label_sc.fit(df.iloc[:, 0].values.reshape(-1, 1))
    # print("A")
    inputs = np.zeros((len(data) - (lookback + pred), lookback, 1))
    # labels = np.zeros((len(data) - (lookback + pred), 1))
    # print(inputs.shape,labels.shape)
    # data[0].type()
    for i in range(lookback, len(data) - (pred)):  # neu la 1 ngay -1 nếu 3 ngaytrong pre -1
        inputs[i - lookback] = data[i - lookback:i]
        # labels[i - lookback] = data[i + pred - 1,0]
    inputs = inputs.reshape(-1, lookback, 1)
    # labels = labels.reshape(-1, 1)
    date =date[lookback:(len(data) - (pred))]
    # print(inputs.shape, labels.shape)
    # print(len(date))
    # print(inputs[1], labels[1])
    # print(inputs[2], labels[2])
    # print(inputs[-1], labels[-1])
    test_portion = int(len(inputs) * 0.03)
    # test_portion = 37
    train_x = inputs[:-test_portion]
    # train_y = labels[:-test_portion]
    test_x = inputs[-test_portion:]
    # test_y = labels[-test_portion:]
    return date, train_x, test_x
def scale_data(df):
    # df = df.sort_values('Date').drop('Date', axis=1)
    # df = df.drop('MocHoa', axis=1)
    # df = df.drop('TanAn', axis=1)
    data = (df[['Vam co dong']].values)
    label_sc.fit(df.iloc[:, 1].values.reshape(-1, 1))
    # portion = int(len(df)*0.3)
    history = [x for x in data[:-5]]
    return history


# def json_response(input_variables, endpoint, lookback):
#     json_res = requests.post(
#         url=endpoint,
#         data=json.dumps(
#             {'instances':
#                  [inputs(input_variables, lookback)]
#              }),
#         headers={'Content-Type': 'application/json'}
#     )
#     return json.loads(json_res)


def fit_data(data):
    data = (np.array([data])).reshape(-1, 1)
    data = sc.fit_transform(data)
    return data


def input_data(flow, data):
    input_variables = data
    input_variables.append(flow)
    # print(input_variables[-3:])
    input_variables = fit_data(input_variables)
    return input_variables


def inputs(data, lookback):
    arr = data[-lookback:]
    arr = (np.array(arr)).reshape(1, lookback, 1)
    return arr


def predict_transf(arr):
    # arr = np.array(arr[0]).reshape(-1, 1)
    return label_sc.inverse_transform(arr)
