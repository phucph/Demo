import os

import flask
import pandas as pd

import tensorflow as tf

from utils.utils import inputs, input_data, scale_data, split_data, predict_transf

df = pd.read_csv('data/VCD_data_1.csv', parse_dates=[0])
model = tf.keras.models.load_model(os.path.join("temp_models/VCD-1.h5"))
model.summary()
app = flask.Flask(__name__, static_folder='static')
data = scale_data(df)
# print(data[-5:])
lookback = 3
# tf.saved_model.save(model, export_path)
labels = ["Một ngày", "Hai ngày", "Ba ngày"]
df['Date'] = df['Date'].apply(lambda x: (x.year, x.month, x.day))
date, train_x, test_x = split_data(df,lookback,1)
predict_data = predict_transf(model.predict(train_x))
@app.route("/", methods=['GET', 'POST'])
def main():
    values = zip(df['Date'], df['Vam co dong'])
    aut = zip(labels, df['Vam co dong'][-3:])

    # print(aut)
    # if flask.request.method == 'GET':
    return (flask.render_template('index.html', predict_1="None", predict_2="None", predict_3="None",
                                  values=values, aut=aut,predict=zip(date,predict_data)))


@app.route('/forescast', methods=['POST'])
# def predict():
    # values = zip(df['Date'], df['Vam co dong'])
    # aut = zip(labels, df['Vam co dong'][-3:])
    # value = []
    # input_variables = data
    # flow = [flask.request.form['flow']]
    # input_variables.append(flow)
    # input_variables = fit_data(flow)
    # predictions = json_response(input_variables,endpoint,lookback).text
    # print(json_response.text)

    # prediction = round(predict_transf(predictions['predictions']).flatten().tolist()[0],2)
    # input_variables = input_variables.tolist()
    # print(type(input_variables))
    # pre = predictions['predictions'][0]
    # print(type(pre), pre)
    # input_variables.append(prediction)
    # predictions_2 = json_response(input_variables,endpoint,lookback).text
    # prediction_2 = round(predict_transf(predictions_2['predictions']).flatten().tolist()[0],2)
    # input_variables.append(predictions_2['predictions'][0])
    # # print('after+1', input_variables)
    # # input_variables = fit_data(input_variables)
    # # print(inputs(input_variables, lookback))
    # prediction_3 = json_response(input_variables,endpoint,lookback).text
    # # print('json2', json_response.text)
    # prediction_3 = round(predict_transf(prediction_3['predictions']).flatten().tolist()[0],2)
    # prediction = model.predict(input_variables)[0]
    # input_variables.append([prediction_3])
def predict():
    values = zip(df['Date'], df['Vam co dong'])
    aut = zip(labels, df['Vam co dong'][-3:])
    value = []
    input_variables = data
    flow = [flask.request.form['flow']]
    # input_variables.append(flow)
    # print(flow)
    # print(inputs(input_data(flow, input_variables), lookback))
    # input_variables = fit_data(flow)
    # predictions = json_response(input_variables,endpoint,lookback).text
    prediction = model.predict(inputs(input_data(flow, input_variables), lookback))
    # print(json_response.text)
    prediction = round(predict_transf(prediction).flatten().tolist()[0], 2)
    # print(prediction)
    value.append(prediction)
    prediction_2 = model.predict(inputs(input_data([prediction], input_variables), lookback))
    prediction_2 = round(predict_transf(prediction_2).flatten().tolist()[0], 2)
    # print(prediction_2)
    value.append(prediction_2)
    prediction_3 = model.predict(inputs(input_data([prediction_2], input_variables), lookback))
    prediction_3 = round(predict_transf(prediction_3).flatten().tolist()[0], 2)
    # print(prediction_3)
    value.append(prediction_3)
    chart = zip(labels, df['Vam co dong'][-3:], value)
    return flask.render_template('index.html',
                                 predict_1=prediction,
                                 predict_2=prediction_2,
                                 predict_3=prediction_3,
                                 values=values,
                                 aut=aut,
                                 chart=chart,
                                 charts=zip(labels,value),
                                 predict=zip(date,predict_data),
                                 dubao=zip(df["Date"][-3:],value)

                                 )


if __name__ == '__main__':
    app.run(debug=True, port=5000)
