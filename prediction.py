from tensorflow.keras.models import load_model
import joblib
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np
import pandas as pd
from Preprocessing import create_dataset, df, data




def load_models():
    model_lstm = load_model('Model/model_LSTM.h5')
    model_gradient = joblib.load('Model/Gradient_boost_model.joblib')
    model_linear = joblib.load('Model/linear_model.joblib')
    model_svr = joblib.load('Model/SVR_model.joblib')
    model_xgboost = joblib.load('Model/Xgboost_model.joblib')

    return model_lstm, model_gradient, model_linear, model_svr, model_xgboost

def predict(model, X_train, X_test):
     
     y_pred_train = model.predict(X_train)
     y_pred_test = model.predict(X_test)

     return y_pred_train, y_pred_test

def evaluation( test_predict, Y_test): 
    mse_loss = tf.keras.losses.MeanSquaredError()(Y_test, test_predict)
    bce_loss = tf.keras.losses.BinaryCrossentropy()(Y_test, test_predict)
    hubber_loss = tf.keras.losses.Huber()(Y_test, test_predict)
    hinge_loss = tf.keras.losses.Hinge()(Y_test, test_predict)
    mape =  (mean_absolute_percentage_error(Y_test, test_predict))*100
    rmse = mean_squared_error(Y_test, test_predict, squared = False)

    return mse_loss, bce_loss, hubber_loss, hinge_loss,mape,rmse

def plot_prediction(data,look_back, train_predict, test_predict, df=df):
     trainPredictPlot = np.empty_like(data)
     trainPredictPlot[:,:] = np.nan
     trainPredictPlot[look_back:len(train_predict)+look_back,:] = train_predict
     testPredictPlot = np.empty_like(data)
     testPredictPlot[:,:] = np.nan
     testPredictPlot[len(train_predict)+(look_back*2)+1:len(data)-1,:] = test_predict
     df['predict_train'] = trainPredictPlot
     df['predict_test'] = testPredictPlot
     
     return df

def result(model,df=df,data= data):
    model_lstm, model_gradient, model_linear, model_svr, model_xgboost = load_models()

    if model == 'LSTM':
          model = model_lstm
          time_step = 39

          scaler=MinMaxScaler(feature_range=(0,1))
          data=scaler.fit_transform(np.array(data).reshape(-1,1))

          training_size=int(len(data)*0.7)
          test_size=len(data)-training_size
          train_data,test_data=data[0:training_size,:],data[training_size:len(data),:1]
          
          X_train, y_train = create_dataset(train_data, time_step)
          X_test, y_test = create_dataset(test_data, time_step)
          X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
          X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
          

          y_pred_train, y_pred_test = predict(model, X_train, X_test)

          train_predict=scaler.inverse_transform(y_pred_train)
          test_predict=scaler.inverse_transform(y_pred_test)
          Y_test = scaler.inverse_transform(y_test.reshape(-1,1))
          Y_train = scaler.inverse_transform(y_train.reshape(-1,1))

          mse_loss, bce_loss, hubber_loss, hinge_loss,mape,rmse = evaluation(test_predict, Y_test)

          df = plot_prediction(data, time_step, train_predict, test_predict )

    elif model == 'Xgboost':
          model = model_xgboost
          time_step = 1
          scaler=MinMaxScaler(feature_range=(0,1))
          data=scaler.fit_transform(np.array(data).reshape(-1,1))
          training_size=int(len(data)*0.7)
          test_size=len(data)-training_size
          train_data,test_data=data[0:training_size],data[training_size:len(data)]
          
          X_train, y_train = create_dataset(train_data, time_step)
          X_test, y_test = create_dataset(test_data, time_step)
          

          y_pred_train, y_pred_test = predict(model, X_train, X_test)

          train_predict=scaler.inverse_transform(y_pred_train.reshape(-1,1))
          test_predict=scaler.inverse_transform(y_pred_test.reshape(-1,1))
          Y_test = scaler.inverse_transform(y_test.reshape(-1,1))
          Y_train = scaler.inverse_transform(y_train.reshape(-1,1))

          mse_loss, bce_loss, hubber_loss, hinge_loss,mape,rmse = evaluation(test_predict, Y_test)

          df = plot_prediction(data, time_step, train_predict, test_predict )


    elif model == 'Support Vector Regression':
          model = model_svr
          time_step = 1
          scaler=MinMaxScaler(feature_range=(0,1))
          data=scaler.fit_transform(np.array(data).reshape(-1,1))
          training_size=int(len(data)*0.7)
          test_size=len(data)-training_size
          train_data,test_data=data[0:training_size],data[training_size:len(data)]
          
          X_train, y_train = create_dataset(train_data, time_step)
          X_test, y_test = create_dataset(test_data, time_step)
          

          y_pred_train, y_pred_test = predict(model, X_train, X_test)

          train_predict=scaler.inverse_transform(y_pred_train.reshape(-1,1))
          test_predict=scaler.inverse_transform(y_pred_test.reshape(-1,1))
          Y_test = scaler.inverse_transform(y_test.reshape(-1,1))
          Y_train = scaler.inverse_transform(y_train.reshape(-1,1))

          mse_loss, bce_loss, hubber_loss, hinge_loss,mape,rmse = evaluation(test_predict, Y_test)

          df = plot_prediction(data, time_step, train_predict, test_predict )

    elif model == 'Linear Regression':
          model = model_linear
          time_step = 1
          scaler=MinMaxScaler(feature_range=(0,1))
          data=scaler.fit_transform(np.array(data).reshape(-1,1))
          training_size=int(len(data)*0.7)
          test_size=len(data)-training_size
          train_data,test_data=data[0:training_size],data[training_size:len(data)]
          
          X_train, y_train = create_dataset(train_data, time_step)
          X_test, y_test = create_dataset(test_data, time_step)
          

          y_pred_train, y_pred_test = predict(model, X_train, X_test)

          train_predict=scaler.inverse_transform(y_pred_train.reshape(-1,1))
          test_predict=scaler.inverse_transform(y_pred_test.reshape(-1,1))
          Y_test = scaler.inverse_transform(y_test.reshape(-1,1))
          Y_train = scaler.inverse_transform(y_train.reshape(-1,1))

          mse_loss, bce_loss, hubber_loss, hinge_loss,mape,rmse = evaluation(test_predict, Y_test)

          df = plot_prediction(data, time_step, train_predict, test_predict )

    elif model == 'Gradient Boost':
          model = model_gradient
          time_step = 1
          scaler=MinMaxScaler(feature_range=(0,1))
          data=scaler.fit_transform(np.array(data).reshape(-1,1))
          training_size=int(len(data)*0.7)
          test_size=len(data)-training_size
          train_data,test_data=data[0:training_size],data[training_size:len(data)]
          
          X_train, y_train = create_dataset(train_data, time_step)
          X_test, y_test = create_dataset(test_data, time_step)
          

          y_pred_train, y_pred_test = predict(model, X_train, X_test)

          train_predict=scaler.inverse_transform(y_pred_train.reshape(-1,1))
          test_predict=scaler.inverse_transform(y_pred_test.reshape(-1,1))
          Y_test = scaler.inverse_transform(y_test.reshape(-1,1))
          Y_train = scaler.inverse_transform(y_train.reshape(-1,1))

          mse_loss, bce_loss, hubber_loss, hinge_loss,mape,rmse = evaluation(test_predict, Y_test)

          df = plot_prediction(data, time_step, train_predict, test_predict )


    return mse_loss, bce_loss, hubber_loss, hinge_loss,mape,rmse, df
