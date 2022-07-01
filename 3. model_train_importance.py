#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import sys


import shap
import dataframe_image as dfi
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import sklearn
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

from sklearn import tree
from sklearn.inspection import permutation_importance

from sklearn.tree import DecisionTreeRegressor
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

# In[2]:


# define which disease to target
target = 'NOPTP'
# target = 'NSPSP'


# SHAP analysis takes time. Skip it if it's unnecessary
performance_only = True


# In[5]:


# Data Call
features = np.load('../Data_Preprocessing/dataset_%s/features.npy'%target)
label = np.load('../Data_Preprocessing/dataset_%s/label.npy'%target)

feature_name = ['Sex','Age','Household income','Education','Self-rated stress level','Blood pressure','Glycated hemoglobin',
                'Dental visit within a year','HDL cholesterol','Triglycerides','Abdominal obesity','Fasting blood glucose',
               'Metabolic syndrome','Body mass index','Diabetes','Hs-CRP','Smoking','Use of interdental cleaning aid']

                


# In[7]:


# List of classifiers to train
model_name = ['Linear Regression','Logistic Regression','CART','GBM','Random Forest','XGBoost','MLP']

# Declare numpy arrays to save results

feature_importance_total = np.zeros((len(model_name),len(feature_name)))
fpr_formaxauc_list = [np.array([]) for i in range(len(model_name))]
tpr_formaxauc_list = [np.array([]) for i in range(len(model_name))]

xgboost_shap = np.zeros((1,len(feature_name)))
mlp_shap = np.zeros((1,len(feature_name)))



cv_num =12
section = int(features.shape[0]/cv_num)

fold_table = np.zeros(shape=(cv_num,2))
for i in range(cv_num):
    fold_table[i,0] = section*i
    fold_table[i,1] = section*(i+1)

fold_table[cv_num-1,1] = features.shape[0]

performance_name = ['AUC','Sensitivity','Specificity','PPV','NPV','Accuracy']
performance_total = np.zeros((len(model_name),len(performance_name),cv_num))


# In[8]:



# Split dataset into training and test
def data_split_func(fold_num,features_input,label_input):
    location_start = int(fold_table[fold_num%cv_num,0])
    location_end = int(fold_table[fold_num%cv_num,1])

    test_features     = features_input[location_start:location_end,:].copy()
    test_label        = label_input[location_start:location_end,].copy()

    training_features = features_input[0:location_start,:].copy()
    training_features = np.append(training_features,features_input[location_end:features_input.shape[0],:],axis=0)

    training_label = label_input[0:location_start,].copy()
    training_label = np.append(training_label,label_input[location_end:label_input.shape[0],],axis=0)

    
    # normalizing dataset based on minmax scaler
    scaler = MinMaxScaler()
    training_features = scaler.fit_transform(training_features )
    test_features = scaler.transform( test_features )

    # transform the format to pandas dataframe
    X_train_func = pd.DataFrame(training_features, columns = feature_name)
    X_test_func = pd.DataFrame(test_features, columns = feature_name)
    y_train_func =  training_label
    y_test_func  =test_label 
    return X_train_func, X_test_func, y_train_func, y_test_func


# In[9]:


def performance_func(predicted_input,y_test_input,standard):
    
    fpr, tpr, thresholds = metrics.roc_curve(y_test_input,predicted_input, pos_label=1)
    
    mean_fpr = np.linspace(0, 1, 100)
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    

    
    temp = (predicted>standard).astype('int')
    cm=metrics.confusion_matrix(y_test_input, temp)

    TP = cm[1,1]
    FN = cm[1,0]
    FP = cm[0,1]
    TN = cm[0,0]
    # Sensitivity, hit rate, recall, or true positive rate
    sensitivity = TP/(TP+FN)
    # Specificity or true negative rate
    specificity = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    
    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    
    return interp_tpr,metrics.auc(fpr, tpr),sensitivity,specificity,PPV,NPV,ACC


# In[10]:


# declare gold standard based on the training result
def standard_selection(pred,true):
    pos_average= np.average(pred[true==1])
    neg_average= np.average(pred[true==0])
    standard = (pos_average+neg_average)/2
    return standard


# # Linear Regression

# In[11]:


model_num = int(0)


for fold_num in range(cv_num):
    X_train, X_test, y_train, y_test = data_split_func(fold_num,features,label)

    # train model
    model_linear = LinearRegression()
    model_linear.fit(X_train, y_train)

    
    # predict test set
    training_pred = model_linear.predict(X_train)
    standard = standard_selection(training_pred,y_train)
    predicted = model_linear.predict(X_test)
    interp_tpr,performance_total[model_num,0,fold_num],performance_total[model_num,1,fold_num],      performance_total[model_num,2,fold_num],performance_total[model_num,3 ,fold_num],       performance_total[model_num,4,fold_num],performance_total[model_num,5,fold_num]    = performance_func(predicted,y_test,standard)
    
    
    # save feature importance
    feature_importance_total[model_num,:] += np.abs(model_linear.coef_).reshape(-1,)

feature_importance_total[model_num,:] = feature_importance_total[model_num,:] /cv_num


# # Logistic Regression

# In[12]:


model_num = 1
model_num = int(model_num)


for fold_num in range(cv_num):
    X_train, X_test, y_train, y_test = data_split_func(fold_num,features,label)
    
    # train model
    model_log = LogisticRegression(C=500, random_state=0)
    model_log.fit(X_train, y_train)
    
    training_pred = model_log.predict_proba(X_train)[:,1]
    standard = standard_selection(training_pred,y_train)
    
    
    # AUC 
    predicted = model_log.predict_proba(X_test)[:,1]
    interp_tpr,performance_total[model_num,0,fold_num],performance_total[model_num,1,fold_num],      performance_total[model_num,2,fold_num],performance_total[model_num,3 ,fold_num],       performance_total[model_num,4,fold_num],performance_total[model_num,5,fold_num]    = performance_func(predicted,y_test,standard)
    
    feature_importance_total[model_num,:] += np.abs(model_log.coef_).reshape(-1,)

        
feature_importance_total[model_num,:] = feature_importance_total[model_num,:] /cv_num


# # CART 

# In[13]:


model_num =2
model_num = int(model_num)

for fold_num in range(cv_num):
    X_train, X_test, y_train, y_test = data_split_func(fold_num,features,label)

    # train model
    model_cart = DecisionTreeRegressor(criterion="mse",random_state = 0,max_depth=10,min_weight_fraction_leaf=0.05,min_impurity_decrease=0.0001)

    model_cart.fit(X_train, y_train)

    training_pred = model_cart.predict(X_train)
    standard = standard_selection(training_pred,y_train)
    
    # AUC 
    predicted = model_cart.predict(X_test)

    interp_tpr,performance_total[model_num,0,fold_num],performance_total[model_num,1,fold_num],      performance_total[model_num,2,fold_num],performance_total[model_num,3 ,fold_num],       performance_total[model_num,4,fold_num],performance_total[model_num,5,fold_num]    = performance_func(predicted,y_test,standard)
    
    feature_importance_total[model_num,:] += model_cart.feature_importances_.reshape(-1,)

    
feature_importance_total[model_num,:] = feature_importance_total[model_num,:] /cv_num


# # GBM

# In[14]:


model_num =3
model_num = int(model_num)

for fold_num in range(cv_num):
    X_train, X_test, y_train, y_test = data_split_func(fold_num,features,label)
    
    # train model
    model_gbm  = GradientBoostingClassifier(random_state=0,loss='deviance',  learning_rate=0.1,n_estimators=100,                                           subsample=0.01,criterion='friedman_mse'                                           )
    model_gbm.fit(X_train, y_train)
    
    training_pred = model_gbm.predict(X_train)
    standard = standard_selection(training_pred,y_train)
    
    predicted = model_gbm.predict(X_test)
  
    interp_tpr,performance_total[model_num,0,fold_num],performance_total[model_num,1,fold_num],      performance_total[model_num,2,fold_num],performance_total[model_num,3 ,fold_num],       performance_total[model_num,4,fold_num],performance_total[model_num,5,fold_num]    = performance_func(predicted,y_test,standard)

    feature_importance_total[model_num,:] += model_gbm.feature_importances_.reshape(-1,)

    
feature_importance_total[model_num,:] = feature_importance_total[model_num,:] /cv_num


# # Random Forest

# In[15]:


model_num =4
model_num = int(model_num)

tprs = []
aucs = []

for fold_num in range(cv_num):
    X_train, X_test, y_train, y_test = data_split_func(fold_num,features,label)
    
    # train model
    model_randomforest = RandomForestClassifier(n_estimators=100,min_samples_leaf =2,
                                            random_state=0)    
    model_randomforest.fit(X_train, y_train)
    predicted = model_randomforest.predict(X_test)
    
    
    # AUC 
    training_pred = model_randomforest.predict(X_train)
    standard = standard_selection(training_pred,y_train)
    

    interp_tpr,performance_total[model_num,0,fold_num],performance_total[model_num,1,fold_num],      performance_total[model_num,2,fold_num],performance_total[model_num,3 ,fold_num],       performance_total[model_num,4,fold_num],performance_total[model_num,5,fold_num]    = performance_func(predicted,y_test,standard)
    
    feature_importance_total[model_num,:] += model_randomforest.feature_importances_.reshape(-1,)

    
feature_importance_total[model_num,:] = feature_importance_total[model_num,:] /cv_num


# # XGBoost

# In[28]:


model_num =5
model_num = int(model_num)


for fold_num in range(cv_num):
    X_train, X_test, y_train, y_test = data_split_func(fold_num,features,label)
    
    # train model
    model_xgboost = xgb.XGBRegressor(random_state=0,colsample_bytree = 0.5,                                      learning_rate = 0.01,verbosity = 1,    booster='gbtree',gamma=1,subsample=0.2,max_depth = 20, alpha = 3, n_estimators = 200)
    model_xgboost.fit(X_train, y_train)
    predicted = model_xgboost.predict(X_test)
    

    # AUC 
    training_pred = model_xgboost.predict(X_train)
    standard = standard_selection(training_pred,y_train)
    interp_tpr,performance_total[model_num,0,fold_num],performance_total[model_num,1,fold_num],      performance_total[model_num,2,fold_num],performance_total[model_num,3 ,fold_num],       performance_total[model_num,4,fold_num],performance_total[model_num,5,fold_num]    = performance_func(predicted,y_test,standard)
        
    if not performance_only:
        feature_importance_total[model_num,:] += model_xgboost.feature_importances_.reshape(-1,)
        explainer = shap.TreeExplainer(model_xgboost)
        shap_values = explainer.shap_values(X_test)    


        shap_values = np.abs(shap_values)

        xgboost_shap += np.average(shap_values,axis=0).reshape(-1,)


feature_importance_total[model_num,:] = feature_importance_total[model_num,:] /cv_num

xgboost_shap = xgboost_shap /cv_num
    


# # MLP

# In[17]:


model_num =6
model_num = int(model_num)


for fold_num in range(cv_num):
    X_train, X_test, y_train, y_test = data_split_func(fold_num,features,label)
    
    # train model
    mlp = MLPRegressor(solver='lbfgs',random_state=0, hidden_layer_sizes=(42),
                       max_iter=150, activation='tanh',early_stopping=True,
                       validation_fraction=0.1,
                                learning_rate_init=0.05, alpha=0.1)

    mlp.fit(X_train, y_train)
    predicted = mlp.predict(X_test)
    
    
    # AUC 
    training_pred = mlp.predict(X_train)
    standard = standard_selection(training_pred,y_train)
  
    interp_tpr,performance_total[model_num,0,fold_num],performance_total[model_num,1,fold_num],      performance_total[model_num,2,fold_num],performance_total[model_num,3 ,fold_num],       performance_total[model_num,4,fold_num],performance_total[model_num,5,fold_num]    = performance_func(predicted,y_test,standard)
   
    feature_importance_total[model_num,:] += permutation_importance(mlp, X_test, y_test,
                          n_repeats=30,
                     random_state=0).importances_mean



    if not performance_only:
        X_train_summary = shap.kmeans(X_train, 10)
        explainer = shap.KernelExplainer(mlp.predict, X_train_summary)
        shap_values = explainer.shap_values(X_test)  
        shap_values = np.abs(shap_values)

        mlp_shap += np.average(shap_values,axis=0).reshape(-1,)


    
feature_importance_total[model_num,:] = feature_importance_total[model_num,:] /cv_num


# # Save results

# In[18]:


np.save("%s/performance_total"%target,performance_total)

if not performance_only:
    np.save("%s/feature_importance_total"%target,feature_importance_total)
    np.save("%s/xgboost_shap"%target,xgboost_shap)


# # Basic Graph

# ## Peformance summary

# In[20]:


performance_total_df= pd.DataFrame(columns=performance_name,index=model_name)

for index_model,name_model in enumerate(model_name):
    for index_per,name_per in enumerate(performance_name):
        performance_total_df.loc['%s'%name_model,'%s'%name_per] =        r'%0.3f $\pm$ %0.2f'%(np.average(performance_total[index_model,index_per,:]),np.std(performance_total[index_model,index_per,:]))

# performance_total_df


# # Feature importance

# In[23]:


feature_importance_rank = pd.DataFrame()
feature_name_array = np.array(feature_name)

for index,name in enumerate(model_name):
    rank = feature_importance_total[index,:].argsort()[::-1]
    feature_importance_rank[name] =  feature_name_array[rank]
    
xgboost_shap = xgboost_shap.reshape(len(feature_name),)
rank = xgboost_shap.argsort()[::-1]
feature_importance_rank['XGBoost-SHAP'] = feature_name_array[rank]

# feature_importance_rank


# # SHAP summary plot

# In[ ]:


total_test_features = np.empty((0,len(feature_name)), float)
total_xgboost_shap_values = np.empty((0,len(feature_name)), float)

for fold_num in range(cv_num):
    temp_test_features = np.load("%s/SHAP/%s_test_features.npy"%(target,fold_num)).copy()
    total_test_features = np.append(total_test_features,temp_test_features,axis=0)
    
    temp_xgboost_shap_values = np.load("%s/SHAP/%s_xgboost_shap_values.npy"%(target,fold_num)).copy()
    total_xgboost_shap_values = np.append(total_xgboost_shap_values,temp_xgboost_shap_values,axis=0)
    
total_test_features_df = pd.DataFrame(total_test_features ,columns=feature_name) 


# In[ ]:


fig, ax = plt.subplots(1,1,figsize=(8,8))
shap.summary_plot(total_xgboost_shap_values,total_test_features_df, max_display=len(feature_name))


# 
# # SHAP-Dependence plot

# In[ ]:


plt.figure(figsize=(6, 6))
dpi_plus=0
for i in range(len(feature_name)):
    if feature_name[i]=="Smoking" or feature_name[i]=="Education":
        shap.dependence_plot(i, total_xgboost_shap_values, total_test_features_df,feature_names=feature_name, alpha=0.5,                             x_jitter=0.2,show=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




