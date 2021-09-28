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

# Data Splitting
n = len(model_dataset_oversample_)
train_df = model_dataset_oversample_[0:int(n*0.9)]
test_df = model_dataset_oversample_[int(n*0.9):]
X_train_oversample, y_train_oversample = train_df.iloc[:,:-1], train_df.iloc[:,-1]
X_test_oversample, y_test_oversample = test_df.iloc[:,:-1], test_df.iloc[:,-1]

# Data Splitting for tuning
n = len(model_dataset_oversample_)
train_df = model_dataset_oversample_[0:int(n*0.9)]
val_df = model_dataset_oversample_[int(n*0.9):int(n*0.95)]
test_df = model_dataset_oversample_[int(n*0.95):]
X_train_oversample_1, y_train_oversample_1 = train_df.iloc[:,:-1], train_df.iloc[:,-1]
X_val_oversample_1, y_val_oversample_1 = val_df.iloc[:,:-1], val_df.iloc[:,-1]
X_test_oversample_1, y_test_oversample_1 = test_df.iloc[:,:-1], test_df.iloc[:,-1]

# Track the model on below URL and set the SQLite DB
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
# Set experiment
experiment = mlflow.get_experiment('0')

# Fetch log model information
def fetch_logged_data(run_id):
    client = mlflow.tracking.MlflowClient()
    data = client.get_run(run_id).data
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in client.list_artifacts(run_id, "model")]
    return data.params, data.metrics, tags, artifacts

# enable autologging
mlflow.sklearn.autolog(log_models=True)
mlflow.xgboost.autolog(log_models=True)

# Hyperparameter tuning for XGBoost Model
xgb_reg = xgb.XGBClassifier()
params = {
        'num_boost_round': [5, 10, 15],
        'eta': [0.05, 0.001, 0.1],
        'max_depth': [6, 5, 8],
        'subsample': [0.9, 1, 0.8],
        'colsample_bytree': [0.9, 1, 0.8],
        'alpha': [0.1, 0.3, 0]
    }

with mlflow.start_run(experiment_id=experiment.experiment_id, run_name='car_accidents_model') as run:
    random_search = RandomizedSearchCV(xgb_reg, params, cv=5, n_iter=50, verbose=1)
    start = time.time()
    random_search.fit(X_train_oversample_1,
                      y_train_oversample_1,
                      eval_set=[(X_train_oversample_1, y_train_oversample_1), (X_val_oversample_1, y_val_oversample_1)],
                      early_stopping_rounds=10,
                      verbose=True)
    best_parameters = random_search.best_params_
    print('RandomizedSearchCV Results: ')
    print(random_search.best_score_)
    print('Best Parameters: ')
    for param_name in sorted(best_parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))
    end = time.time()
    print('time elapsed: ' + str(end-start))
    print(' ')
    print('Best Estimator: ')
    print(random_search.best_estimator_)
    y_pred_xgb_oversample = random_search.predict(X_test_oversample_1)

# Set target as integer
y_train_oversample = y_train_oversample.astype(int)
y_test_oversample = y_test_oversample.astype(int)

# Save best parameters for final model training
params = random_search.best_params_
params['eval_metric'] = 'mae'
num_boost_round = 200

# Convert dataset into DMatrix for XGBoost native implementation
dtrain = xgb.DMatrix(X_train_oversample, label=y_train_oversample)
dtest = xgb.DMatrix(X_test_oversample, label=y_test_oversample)

# Training Final Model
with mlflow.start_run(experiment_id=experiment.experiment_id, run_name='xgboost_model') as run:
    xgb_reg_main = xgb.train(params, dtrain, num_boost_round=num_boost_round, evals=[(dtest, "Test")], early_stopping_rounds=20)
    mlflow.set_tag('model_name', 'car_accidents_model_xgb')
    # Log Parameters
    mlflow.log_param('subsample', params['subsample'])
    mlflow.log_param('max_depth', params['max_depth'])
    mlflow.log_param('eta', params['eta'])
    mlflow.log_param('colsample_bytree', params['colsample_bytree'])
    mlflow.log_param('alpha', params['alpha'])
    y_xgb_preds_main = xgb_reg_main.predict(dtest)
    y_xgb_preds_main = [1 if n >= 0.5 else 0 for n in y_xgb_preds_main]
    y_xgb_preds_main_proba = xgb_reg_main.predict(dtest)
    # Calculating Metrics
    auc_xgb_main = roc_auc_score(y_test_oversample, y_xgb_preds_main_proba)
    acc_xgb_main = (y_xgb_preds_main == y_test_oversample).sum().astype(float) / len(y_xgb_preds_main)*100
    f1_xgb_main = f1_score(y_test_oversample, y_xgb_preds_main, average='micro')
    mcc_xgb_main = matthews_corrcoef(y_test_oversample, y_xgb_preds_main)
    features = X_train_oversample.columns
    # Feature Importance
    ax = xgb.plot_importance(xgb_reg_main, max_num_features=25, height=0.5, importance_type='weight')
    fig = ax.figure
    fig.set_size_inches(15, 8)
    fig.savefig('www/xgb_results/xgb_main_imp.png',dpi=100, bbox_inches = 'tight')
    # Test ROC
    roc_xgb_main = metric_graph(y_test_oversample, y_xgb_preds_main_proba, metric='roc', figsize=(15, 8), 
                            filename='www/xgb_results/xgb_main_roc.png')
    # Test Confusion Matrix
    import matplotlib.pyplot as plt
    class_names = np.unique(y_test_oversample)
    matrix = confusion_matrix(y_test_oversample, y_xgb_preds_main)
    plt.clf()
    # place labels at the top
    plt.gca().xaxis.tick_top()
    plt.gca().xaxis.set_label_position('top')
    # plot the matrix per se
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    # plot colorbar to the right
    plt.colorbar()
    fmt = 'd'
    # write the number of predictions in each bucket
    thresh = matrix.max() / 2.
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):

        # if background is dark, use a white number, and vice-versa
        plt.text(j, i, format(matrix[i, j], fmt),
             horizontalalignment="center",
             color="white" if matrix[i, j] > thresh else "black")
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.tight_layout()
    plt.ylabel('True label',size=14)
    plt.xlabel('Predicted label',size=14)
    plt.title('Technique: oversample | Model: XGBoost')
    plt.savefig('www/xgb_results/xgb_main_cm.png',dpi=100, bbox_inches = 'tight')
    # Test Log Metrics
    mlflow.log_metric('test_auc', auc_xgb_main)
    mlflow.log_metric('test_accuracy', acc_xgb_main)
    mlflow.log_metric('test_f1_score', f1_xgb_main)
    mlflow.log_metric('test_mcc_score', mcc_xgb_main)
    mlflow.log_artifacts('www/xgb_results')
    # Log Features
    pd.DataFrame(columns = X_train_oversample.columns).to_csv('xgb_features.csv', index=False)
    mlflow.log_artifact('xgb_features.csv', artifact_path='features')

# Hyperparameter tuning for Random Forest Model
rfc_reg = RandomForestClassifier()

params = {
        'bootstrap': [True, False],
        'class_weight': [None, 'balanced', 'balanced_subsample'],
        'max_features': ['auto', 'sqrt', 'log2'],
        'min_samples_leaf': [3, 4, 6, 8],
        'min_samples_split': range(2, 10),
        'n_estimators': [100, 250, 500],
        'max_depth': [4, 6, 8, None],
        'ccp_alpha': [0.1, 0.0, 0.01, 0.001]
    }

with mlflow.start_run(experiment_id=experiment.experiment_id, run_name='car_accidents_model_2') as run:
    random_search = RandomizedSearchCV(rfc_reg, params, cv=5, n_iter=50, verbose=1)
    start = time.time()
    random_search.fit(X_train_oversample,
                      y_train_oversample)
    best_parameters = random_search.best_params_
    print('RandomizedSearchCV Results: ')
    print(random_search.best_score_)
    print('Best Parameters: ')
    for param_name in sorted(best_parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))
    end = time.time()
    print('time elapsed: ' + str(end-start))
    print(' ')
    print('Best Estimator: ')
    print(random_search.best_estimator_)
    y_pred_rfc_oversample = random_search.predict(X_test_oversample)

# Final Random Forest Model
with mlflow.start_run(experiment_id=experiment.experiment_id, run_name='random_forest_model') as run:
    rfc_reg_main = RandomForestClassifier(**random_search.best_params_)
    rfc_reg_main.fit(X_train_oversample, y_train_oversample)
    mlflow.set_tag('model_name', 'car_accidents_model_rf')
    # Save Model
    signature = infer_signature(X_train_oversample, rfc_reg_main.predict(X_train_oversample))
    mlflow.sklearn.log_model(rfc_reg_main, 'rf_model', signature=signature)
    y_preds_rfc_main = rfc_reg_main.predict(X_test_oversample)
    y_preds_rfc_main_proba = rfc_reg_main.predict_proba(X_test_oversample)
    # Calculating Metrics
    auc_rfc_main = roc_auc_score(y_test_oversample, y_preds_rfc_main_proba[:, 1])
    acc_rfc_main = (y_preds_rfc_main == y_test_oversample).sum().astype(float) / len(y_preds_rfc_main)*100
    f1_rfc_main = f1_score(y_test_oversample, y_preds_rfc_main, average='micro')
    mcc_rfc_main = matthews_corrcoef(y_test_oversample, y_preds_rfc_main)
    features = X_train_oversample.columns
    # Feature Importance
    rfc_importances_main = pd.DataFrame({'Feature': features, 'Importance': rfc_reg_main.feature_importances_})
    rfc_importances_main = rfc_importances_main.sort_values(by='Importance', ascending=False)
    rfc_importances_main = rfc_importances_main.set_index('Feature')
    imp_rfc_main = rfc_importances_main[:25].plot.bar(figsize=(15,8))
    fig = imp_rfc_main.get_figure()
    fig.savefig('www/rfc_results/rfc_main_imp.png',dpi=100, bbox_inches = 'tight')
    # Test ROC
    roc_rfc_main = metric_graph(y_test_oversample, y_preds_rfc_main_proba[:,1], metric='roc', figsize=(15, 8), 
                            filename='www/rfc_results/rfc_main_roc.png')
    # Test Confusion Matrix
    class_names = np.unique(y_test_oversample)
    disp_rfc = plot_confusion_matrix(rfc_reg_main, X_test_oversample, y_test_oversample,
                                     display_labels=class_names,
                                     cmap=plt.cm.Blues)
    disp_rfc.ax_.set_title("Technique: oversample | Model: Random Forest")
    plt.savefig('www/rfc_results/rfc_main_cm.png',dpi=100, bbox_inches = 'tight')
    # Test Log Metrics
    mlflow.log_metric('test_auc', auc_rfc_main)
    mlflow.log_metric('test_accuracy', acc_rfc_main)
    mlflow.log_metric('test_f1_score', f1_rfc_main)
    mlflow.log_metric('test_mcc_score', mcc_rfc_main)
    mlflow.log_artifacts('www/rfc_results')
    # Log Features
    pd.DataFrame(columns = X_train_oversample.columns).to_csv('rf_features.csv', index=False)
    mlflow.log_artifact('rf_features.csv', artifact_path='features')

print("Models are trained...")