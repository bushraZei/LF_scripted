one_file_path = '../../load_forecasting/Lastprognose/features_added/weather_calendar/Heikendorf_Hammerstiel.xlsx'
file_dir = '../../load_forecasting/Lastprognose/features_added/weather_calendar'
features_to_drop = ['Kiel: Taupunkt [°C]', 'Kiel: Niederschlag [mm/h]', 'Kiel: Luftdruck [hPa]','Datum','Zeit',
       'Kiel: Schneeefall [mm/h]', 'Kiel: Windböhen', 'Status','year','Plausibilität','weekofyear','weekday','Kiel: Windgschwindigkeit [km/h]','Kiel: Windrichtung [°]']
features_to_scale = ['Kiel: Temperatur [°C]', 'Kiel: Luftfeuchtigkeit [%]',
       'Kiel: Sonnenscheindauer [min/h]']

features_to_dummify = ['season', 'dayofweek','hour','time_of_day','month']

