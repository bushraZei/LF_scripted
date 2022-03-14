import os
from read_data import read_data
from process import correct_frequency, ks_imputation
import pandas as pd
from neural_nets.functions import all_dataset_wrapper, plot_history
from neural_nets.nn_archi import cnn_bn, fnn_model
import args
import tensorflow as tf
from os import walk
import matplotlib.pyplot as plt


file_path = '../../load_forecasting/Lastprognose/features_added/weather_calendar/Heikendorf_Hammerstiel.xlsx'
file_dir = '../../load_forecasting/Lastprognose/features_added/weather_calendar'
all_files = next(walk(file_dir), (None, None, []))[2]  # [] if no file


features_to_drop = ['Kiel: Taupunkt [°C]', 'Kiel: Niederschlag [mm/h]', 'Kiel: Luftdruck [hPa]','Datum','Zeit',
       'Kiel: Schneeefall [mm/h]', 'Kiel: Windböhen', 'Status','year','Plausibilität','weekofyear','weekday','Kiel: Windgschwindigkeit [km/h]','Kiel: Windrichtung [°]']

features_to_scale = ['Kiel: Temperatur [°C]', 'Kiel: Luftfeuchtigkeit [%]',
       'Kiel: Sonnenscheindauer [min/h]']

features_to_dummify = ['season', 'dayofweek','hour','time_of_day','month']
   
argss = {'epochs': 1,'loss':tf.keras.losses.Huber(),'optimizer':tf.keras.optimizers.Adam(learning_rate=1e-6),'metric':tf.metrics.MeanAbsoluteError(),
               'window_size':96, 'n_horizon':1, 'batch_size':128, 'shuffle_buffer_size': 1000, 'multi_var':True, 'type_scale':'normalization','shift':1
}

exo_factors = True
one_file = True

if __name__ == "__main__":

   '''
   data = read_data(file_path, 'Zeitstempel',multiple_sheets=True)
   #data = correct_frequency(data)
   #data = ks_imputation(data)
   
   data.drop(features_to_drop, axis = 1, inplace=True)
   data = pd.get_dummies(data, columns = features_to_dummify, drop_first=True)
   if('Unnamed: 0') in data.columns:
      data.drop(['Unnamed: 0'], axis=1, inplace=True)
   
   load_column = "Wert (kW)" if "Wert (kW)" in data.columns else "Wert"

   n_features = data.shape[1]

   ffn = fnn_model(window_size=args.window_size, n_features=n_features, n_horizon=hyperparam['n_horizon'])
   #model_history, model = wrapper(data, ffn, load_column , features_to_scale, scale, args.window_size, args.n_horizon, args.shift, hyperparam)
   
   #all files
   frames = [] # not scaled
   for file in all_files:
      df = read_data(file_dir+'/'+file,'Zeitstempel',multiple_sheets=True)
      df.drop(features_to_drop, axis = 1, inplace=True,  errors='ignore')
      #df.drop(columns=['Zeitstempel'],axis=1,inplace=True)
      df = pd.get_dummies(df, columns = features_to_dummify, drop_first=True)
      if('Unnamed: 0') in df.columns:
         df.drop(['Unnamed: 0'], axis=1, inplace=True)
      
      frames.append(df)

   #print(frames)
   #print(len(frames))
   
   ffn = fnn_model(window_size=args.window_size, n_features=frames[0].shape[1], n_horizon=hyperparam['n_horizon'])
   model_history, model, test_ds = all_ds_wrapper(frames, ffn, hyperparam)

   mae=model_history.history['mae']
   loss=model_history.history['loss']

   epochs=range(len(loss)) # Get number of epochs



   plt.figure()

   epochs_zoom = epochs[50:]
   mae_zoom = mae[50:]
   loss_zoom = loss[50:]

   #------------------------------------------------
   # Plot Zoomed MAE and Loss
   #------------------------------------------------
   plt.plot(epochs_zoom, mae_zoom, 'r')
   plt.plot(epochs_zoom, loss_zoom, 'b')
   plt.title('MAE and Loss')
   plt.xlabel("Epochs")
   plt.ylabel("MAE")
   plt.legend(["MAE", "Loss"])

   plt.figure()
   '''

   frames = [] # not scaled
   for file in all_files:
      df = read_data(file_dir+'/'+file,'Zeitstempel',multiple_sheets=True)
      df.drop(features_to_drop, axis = 1, inplace=True,  errors='ignore')
      df = pd.get_dummies(df, columns = features_to_dummify, drop_first=True)
      if('Unnamed: 0') in df.columns:
         df.drop(['Unnamed: 0'], axis=1, inplace=True)
      
      frames.append(df)
   
   ffn = cnn_bn(window_size=args.window_size, n_features=frames[0].shape[1], n_horizon=argss['n_horizon'])
   model_history, model, test_ds = all_dataset_wrapper(frames, argss, features_to_scale, ffn)
   plot_history(model_history)
   #================================= save model weights ==================================#
   model_weights_dir = '../model_weights/'+model.name
   os.makedirs(model_weights_dir, exist_ok=False)
   model.save(model_weights_dir)


