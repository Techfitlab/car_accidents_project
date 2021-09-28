# Import packages
import os
import sys
from importlib import reload
import imblearn
import pandas as pd
import warnings as w
import numpy as np
import json
import datetime
import time
import copy
import math
import plotly.graph_objects as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from dython.nominal import associations
from dython.model_utils import metric_graph
from collections import Counter
import cufflinks as cf
from IPython.display import Image, display
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from scipy import stats
import statsmodels.api as sm
import sidetable as stb
import sklearn
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, matthews_corrcoef, recall_score, f1_score, \
plot_confusion_matrix, roc_auc_score, classification_report
from sklearn.preprocessing import MaxAbsScaler
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import tensorflow as tf
from sklearn.cluster import KMeans
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

# Set Working Directory
os.environ['root'] = '/Users/greengodfitness/Desktop/Vodafone'
os.chdir(os.environ['root'])

# Load data from csv file
tr_data = pd.read_csv('data/train_data.csv')
ts_data = pd.read_csv('data/test_data.csv')
df = pd.concat([tr_data, ts_data])

# Removing 8 missing values that exist on time column
df = df[df.Time.notnull()]

# Local district based data for population density
pop_density = pd.read_csv('data/population_density.csv')
pop_density = pop_density[['Local_Authority_(Highway)','Population_density']]

# Merging population density with our mai dataset
df = pd.merge(df, pop_density, how = 'left', on = 'Local_Authority_(Highway)')

# Convert Date column into date format
df['Date'] = pd.to_datetime(df['Date'])

# Creating only two classes based on the task description:
# Target variable Did_Police_Officer_Attend_Scene_of_Accident is True when value is equal to 1, False otherwise
df['Did_Police_Officer_Attend_Scene_of_Accident'] = np.where(df['Did_Police_Officer_Attend_Scene_of_Accident']==-1.0, 0,
                                                        np.where(df['Did_Police_Officer_Attend_Scene_of_Accident']==2.0, 0,
                                                            np.where(df['Did_Police_Officer_Attend_Scene_of_Accident']==1.0, 1,
                                                               df['Did_Police_Officer_Attend_Scene_of_Accident'])))
df['Did_Police_Officer_Attend_Scene_of_Accident'] = df['Did_Police_Officer_Attend_Scene_of_Accident'].astype('str')

# Mapping types based on the data
df['Accident_Severity'] = df['Accident_Severity'].astype('str')
df['Day_of_Week'] = df['Day_of_Week'].astype('str')
df['1st_Road_Class'] = df['1st_Road_Class'].astype('str')
df['2nd_Road_Class'] = df['2nd_Road_Class'].astype('str')
df['Road_Type'] = df['Road_Type'].astype('str')
df['Speed_limit'] = df['Speed_limit'].astype('str')
df['Junction_Detail'] = df['Junction_Detail'].astype('str')
df['Junction_Control'] = df['Junction_Control'].astype('str')
df['Pedestrian_Crossing-Human_Control'] = df['Pedestrian_Crossing-Human_Control'].astype('str')
df['Pedestrian_Crossing-Physical_Facilities'] = df['Pedestrian_Crossing-Physical_Facilities'].astype('str')
df['Light_Conditions'] = df['Light_Conditions'].astype('str')
df['Weather_Conditions'] = df['Weather_Conditions'].astype('str')
df['Road_Surface_Conditions'] = df['Road_Surface_Conditions'].astype('str')
df['Special_Conditions_at_Site'] = df['Special_Conditions_at_Site'].astype('str')
df['Carriageway_Hazards'] = df['Carriageway_Hazards'].astype('str')
df['Urban_or_Rural_Area'] = df['Urban_or_Rural_Area'].astype('str')

# Feature Engineering: Traffic Category based on Time
# Converting time feature into traffic categories
# Slice the hour from time column
df['Hour'] = df['Time'].str[0:2]
# Fill values as -1 for unknown hours
df['Hour'] = df['Hour'].fillna(-1)
# convert hour to numeric datetype
df['Hour'] = df['Hour'].astype('int64')
def traffic_categories(hour):
    if hour >= 6 and hour < 10:
        return 'commuters_traffic'
    elif hour >= 10 and hour < 13:
        return 'work_start_traffic'
    elif hour >= 13 and hour < 17:
        return 'afternoon_traffic'
    elif hour >= 17 and hour < 21:
        return 'work_end_traffic'
    elif hour == -1:
        return 'unknown'
    else:
        return 'night_traffic'
df['traffic_category'] = df['Hour'].apply(traffic_categories)
df['Did_Police_Officer_Attend_Scene_of_Accident'] = np.where(df['Did_Police_Officer_Attend_Scene_of_Accident']=='0.0', '0',
                                   np.where(df['Did_Police_Officer_Attend_Scene_of_Accident']=='1.0', '1',
                                   df['Did_Police_Officer_Attend_Scene_of_Accident']))

# Feature Engineering: Studying seasonality by using sine and cosine functions
df['Date_'] = df.Date.astype(str)
df['Time_'] = pd.to_datetime(df['Time'],format= '%H:%M').dt.time.astype(str)
df['Date_Time'] = pd.to_datetime(df['Time_'] + ' ' + df['Date_'])
date_time = pd.to_datetime(df['Date_Time'], format='%Y.%m.%d %H:%M:%S')
timestamp_s = date_time.map(datetime.datetime.timestamp)
day = 24*60*60
df['Day_Sin'] = np.sin(timestamp_s * (2 * np.pi / day))
df['Day_Cos'] = np.cos(timestamp_s * (2 * np.pi / day))

# Feature Engineering: Traffic Category based on RFM Model (Recency, Frequency and Monetary Value)
rfm_dataset = copy.deepcopy(df)
today = datetime.datetime.now()
rfm_dataset['Date'] = pd.to_datetime(rfm_dataset['Date'],format='%Y-%m-%d')
rfm_dataset['Date_Delta'] = (today - rfm_dataset['Date']).dt.days
# Jitter function to create quartiles
nbins = 4
def jitter(a_series, noise_reduction=1000000):
    return (np.random.random(len(a_series))*a_series.std()/noise_reduction)-(a_series.std()/(2*noise_reduction))
# Recency
recency = rfm_dataset.groupby(['Local_Authority_(District)'], as_index=False)['Date_Delta'].mean().round()
recency['Recency'] = pd.qcut(recency['Date_Delta'] + jitter(recency['Date_Delta']), 
                                   nbins, labels=False)+1
# Frequency
frequency = rfm_dataset.groupby(['Local_Authority_(District)']).size().reset_index(name='Accidents')
frequency['Frequency'] = pd.qcut(frequency['Accidents'] + jitter(frequency['Accidents']), 
                                   nbins, labels=False)+1
# Monetary Value
rfm_dataset['Police_Force'] = rfm_dataset['Police_Force'].astype('int64')
monetary = rfm_dataset.groupby(['Local_Authority_(District)'], as_index=False)['Police_Force'].sum().round()
monetary['Monetary'] = pd.qcut(monetary['Police_Force'] + jitter(monetary['Police_Force']), 
                                   nbins, labels=False)+1
# Merge the score to create rfm_score column
rec_fre = pd.merge(recency, frequency, how = 'inner', on = 'Local_Authority_(District)')
rec_fre_mon = pd.merge(rec_fre, monetary, how = 'inner', on = 'Local_Authority_(District)')
# Apply weights
rec_fre_mon['rec_score'] = rec_fre_mon['Recency'] * 15
rec_fre_mon['fre_score'] = rec_fre_mon['Frequency'] * 45
rec_fre_mon['mon_score'] = rec_fre_mon['Monetary'] * 50
rec_fre_mon['rfm_score'] = rec_fre_mon['rec_score'] + rec_fre_mon['fre_score'] + rec_fre_mon['mon_score']
rec_fre_mon['rfm_score_decile'] = pd.qcut(rec_fre_mon['rfm_score'] + jitter(rec_fre_mon['rfm_score']), 
                                   10, labels=False)+1
# Police force index by accidents (under index or over index)
rec_fre_mon['force_allocation'] = (rec_fre_mon['Accidents'] / rec_fre_mon['Police_Force']).apply(lambda x: \
                                                                                        float("{:.2f}".format(x)))
average_index = rec_fre_mon['force_allocation'].mean()
rec_fre_mon['force_index'] = np.where(rec_fre_mon['force_allocation']<average_index, 0,
                                np.where(rec_fre_mon['force_allocation']>=average_index, 1, ''))
# Merge it to main dataframe
rec_fre_mon = rec_fre_mon[['Local_Authority_(District)','rfm_score','rfm_score_decile','force_allocation',
                           'force_index']]
df = pd.merge(df, rec_fre_mon, how = 'inner', on = 'Local_Authority_(District)')

# Feature Engineering: K-Means Clusters based on Multiple Features
clust_df = copy.deepcopy(df)
# Removing unwanted columns and selecting columns for creating clusters
clust_df = clust_df[['Accident_Index', 'Road_Type', 'Speed_limit', 'Junction_Control', 
       'Light_Conditions', 'Weather_Conditions', 'Road_Surface_Conditions', 'Urban_or_Rural_Area']]
# Creating Kmeans function to create raw clusters on our dataset as a feature for the model to train
def Kmeans(data, output='add'):
        n_clusters = 6
        db_id = data['Accident_Index']
        del data['Accident_Index']
        clf = KMeans(n_clusters = n_clusters, random_state=12)
        clf.fit(data)
        y_labels_train = clf.labels_
        if output == 'add':
            data['km_clusters'] = y_labels_train
        elif output == 'replace':
            data = y_labels_train[:, np.newaxis]
        else:
            raise ValueError('Output should be either add or replace')
        db = pd.concat([db_id, data], axis = 1)
        return db
clusters = Kmeans(data=clust_df, output='add')
clusters['km_clusters'] = clusters['km_clusters'].astype('str')
clusters = clusters[['Accident_Index','km_clusters']]
df = pd.merge(df, clusters, how = 'inner', on = 'Accident_Index')

# Removing unwanted columns and seleting overall features for model training
df = df[['Accident_Index', 'Police_Force', 'Accident_Severity',
       'Number_of_Vehicles', 'Number_of_Casualties','1st_Road_Class', 'Road_Type', 
       'Speed_limit', 'Junction_Detail', 'Junction_Control', '2nd_Road_Class',
       'Pedestrian_Crossing-Human_Control', 'Pedestrian_Crossing-Physical_Facilities', 
       'Light_Conditions', 'Weather_Conditions', 'Road_Surface_Conditions',
       'Special_Conditions_at_Site', 'Carriageway_Hazards', 'Day_of_Week',
       'Urban_or_Rural_Area', 'Did_Police_Officer_Attend_Scene_of_Accident',
       'traffic_category', 'rfm_score', 'force_allocation', 'Day_Sin', 
       'Day_Cos', 'km_clusters', 'Population_density']]

# Filling Missing Values
df['Population_density'] = df['Population_density'].fillna(df['Population_density'].mean())

# Removing Outliers
# Remove_outlier fucntion is created to remove the outliers in the dataset
def remove_outliers(skewed_dataset=df):
    columns_ = list(skewed_dataset.select_dtypes(include=['int64','float64']).columns)
    columns_.remove('Population_density')
    str_columns = list(skewed_dataset.select_dtypes(include=['object']).columns)
    str_columns.remove('Accident_Index')
    str_columns.remove('Did_Police_Officer_Attend_Scene_of_Accident')
    db_id = skewed_dataset['Accident_Index']
    db1 = skewed_dataset[columns_]
    db2 = skewed_dataset[str_columns]
    db3 = skewed_dataset[['Population_density']]
    target = skewed_dataset['Did_Police_Officer_Attend_Scene_of_Accident']
    z = (np.abs(stats.zscore(db1)) > 3)
    unskewed_dataset = db1.mask(z).interpolate(method = 'linear')
    db = pd.concat([db_id, unskewed_dataset, db3, db2, target], axis = 1)
    return db
model_dataset = remove_outliers(df)

# Oversampling
model_dataset_o = copy.deepcopy(model_dataset)
X_df, y_df = model_dataset_o.iloc[:,1:-1], model_dataset_o.iloc[:,-1]
df_ids = model_dataset_o.iloc[:,0]
oversample = imblearn.over_sampling.SMOTENC(categorical_features=[8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],\
                                            random_state=0)
X_oresampled, y_oresampled = oversample.fit_resample(X_df, y_df)
model_dataset_oversample = pd.concat([X_oresampled, y_oresampled], axis = 1)
model_dataset_oversample = sklearn.utils.shuffle(model_dataset_oversample)

# One-Hot Encoding
target = model_dataset_oversample['Did_Police_Officer_Attend_Scene_of_Accident']
model_dataset_oversample = model_dataset_oversample.drop(['Did_Police_Officer_Attend_Scene_of_Accident'], axis = 1)
columns_ = list(model_dataset_oversample.select_dtypes(include=['int64','float64']).columns)
str_columns = list(model_dataset_oversample.select_dtypes(include=['object']).columns)
db1 = model_dataset_oversample[columns_]
db2 = pd.get_dummies(model_dataset_oversample[str_columns])
model_dataset_oversample_ = pd.concat([db1, db2, target], axis = 1)

# Data Split
X_pred, y_pred = model_dataset_.iloc[:,:-1], model_dataset_.iloc[:,-1]

# Track the model on below URL and set the SQLite DB
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

experiment = mlflow.get_experiment('0')
print("Name: {}".format(experiment.name))
print("Artifact Location: {}".format(experiment.artifact_location))
print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))
print("Experiment ID: {}".format(experiment.experiment_id))

# Extract Training Features From the Artifact
from mlflow.tracking import MlflowClient
client = MlflowClient()

for mv in client.search_model_versions("name='ca_xgboost_model'"):
    model_versions = dict(mv)
    if model_versions['current_stage'] == 'Production':
        xgb_dict = dict(mv)
    else:
        xgb_dict = '0'

if xgb_dict == '0':
    print('No model version is deployed to production stage..')
else:
    features_filepath = os.path.join('./artifacts/0/',xgb_dict['run_id'],'artifacts/features/xgb_features.csv')

model_features = pd.read_csv(features_filepath)

# Aligning Features with New Dataset
X_pred_id = X_pred['Accident_Index']
del X_pred['Accident_Index']

# Get missing columns in the training set
missing_cols = set(model_features.columns) - set(X_pred.columns)
# Add a missing column in prediction set with default value equal to 0
for i in missing_cols:
    X_pred[i] = 0
    X_pred[i] = X_pred[i].astype('uint8')
    
# Ensure the order of column in the prediction set is in the same order than in training set
X_pred = X_pred[model_features.columns]

print('Loading XGB Model...')
# Automatically Load Production Models
xgb_model_name = "ca_xgboost_model"
stage = 'Production'

xgb_reg_main = mlflow.xgboost.load_model(
    model_uri=f"models:/{xgb_model_name}/{stage}"
)

rfc_model_name = "ca_random_forest_model"

rfc_reg_main = mlflow.sklearn.load_model(
    model_uri=f"models:/{rfc_model_name}/{stage}"
)

# Predict XGB
XGBoost = xgb_reg_main.predict(xgb.DMatrix(X_pred))
car_accidents_1 = pd.DataFrame(XGBoost, columns=['XGB_PROBABILITY'])

# Data Split
X_pred, y_pred = model_dataset_.iloc[:,:-1], model_dataset_.iloc[:,-1]

# Get features for Random Forest Model
for mv in client.search_model_versions("name='ca_random_forest_model'"):
    model_versions = dict(mv)
    if model_versions['current_stage'] == 'Production':
        rf_dict = dict(mv)
    else:
        rf_dict = '0'

if rf_dict == '0':
    print('No model version is deployed to production stage..')
else:
    features_filepath = os.path.join('./artifacts/0/',rf_dict['run_id'],'artifacts/features/rfc_features.csv')

model_features = pd.read_csv(features_filepath)

# Aligning Features with New Dataset
X_pred_id = X_pred['Accident_Index']
del X_pred['Accident_Index']

# Get missing columns in the training set
missing_cols = set(model_features.columns) - set(X_pred.columns)
# Add a missing column in prediction set with default value equal to 0
for i in missing_cols:
    X_pred[i] = 0
    X_pred[i] = X_pred[i].astype('uint8')
    
# Ensure the order of column in the prediction set is in the same order than in training set
X_pred = X_pred[model_features.columns]

# Predict RF
Random_Forest_Flag = rfc_reg_main.predict(X_pred)
Random_Forest_Probability = rfc_reg_main.predict_proba(X_pred)

# Combine XGB and RF predictions and create average probability
car_accidents_submission = pd.concat([db_id, test_data_, 
                         pd.DataFrame(Random_Forest_Probability[:,1], columns=['RF_PROBABILITY']), 
                         pd.DataFrame(Random_Forest_Flag, columns=['RF_FLAG'])], axis = 1)
car_accidents_submission = pd.concat([car_accidents_submission, car_accidents_1], axis = 1)

# Create average prediction column
car_accidents_submission['probability_to_attend'] = car_accidents_submission[['XGB_PROBABILITY','RF_PROBABILITY']].mean(axis=1)

# Final submission
submission = car_accidents_submission[['Accident_Index','probability_to_attend']]

# Save as .csv
submission.to_csv('car_accidents_submission.csv', index=False)

print("Model run is completed...")