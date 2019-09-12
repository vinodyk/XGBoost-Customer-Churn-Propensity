#!/usr/bin/env python
# coding: utf-8

# # # Identifying various features of customer Churn with XGBoost (Extreme Gradient Boosting)
# XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. It implements machine learning algorithms under the Gradient Boosting framework.
# In[101]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn import preprocessing


# For the predictive models
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score, classification_report, confusion_matrix
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier as GBSklearn
from xgboost import XGBClassifier as XGB


# Warning Removal
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")


# In[75]:


def count_by_group(data, feature, target):
    df = data.groupby([feature, target])[target].agg(['count'])
    temp = data.groupby([feature])[target].agg(['count'])
    df['pct'] = 100*df.div(temp, level = feature).reset_index()['count'].values
    return df.reset_index()

def classification_report_to_dataframe(true, predictions, predictions_proba, model_name, balanced = 'no'):
    a = classification_report(true, predictions, output_dict = True)
    zeros = pd.DataFrame(data = a['0'], index = [0]).iloc[:,0:3].add_suffix('_0')
    ones = pd.DataFrame(data = a['1'], index = [0]).iloc[:,0:3].add_suffix('_1')
    df = pd.concat([zeros, ones], axis = 1)
    temp = list(df)
    df['Model'] = model_name
    df['Balanced'] = balanced
    df['Accuracy'] = accuracy_score(true, predictions)
    df['Balanced_Accuracy'] = balanced_accuracy_score(true, predictions)
    df['AUC'] = roc_auc_score(true, predictions_proba, average = 'macro')
    df = df[['Model', 'Balanced', 'Accuracy', 'Balanced_Accuracy', 'AUC'] + temp]
    return df

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


# In[57]:


dataset = pd.read_csv('churn-data.csv')


# In[15]:


dataset.head()


# In[16]:


dataset.info()


# In[17]:


dataset.isna().sum()


# In[18]:


round(dataset.describe(),3)


# In[20]:


exited = len(dataset[dataset['Churn'] == 'Yes']['Churn'])
not_exited = len(dataset[dataset['Churn'] == 'No']['Churn'])
exited_perc = round(exited/len(dataset)*100,1)
not_exited_perc = round(not_exited/len(dataset)*100,1)

print('Number of clients that have exited the program: {} ({}%)'.format(exited, exited_perc))
print('Number of clients that haven\'t exited the program: {} ({}%)'.format(not_exited, not_exited_perc))


# In[24]:


multiplelines = list(dataset['MultipleLines'].unique())
gender = list(dataset['gender'].unique())
paymentmethod = list(dataset['PaymentMethod'].unique())
contract = list(dataset['Contract'].unique())
internetservice = list(dataset['InternetService'].unique())

print(multiplelines)
print(gender)
print(paymentmethod)
print(contract)
print(internetservice)


# In[26]:


dataset['Exited_str'] = dataset['Churn']
dataset['Exited_str'] = dataset['Exited_str'].map({1: 'Exited', 0: 'Stayed'})


# In[28]:


gender_count = dataset['gender'].value_counts()
gender_pct= gender_count / len(dataset.index)

gender = pd.concat([gender_count, round(gender_pct,2)], axis=1)        .set_axis(['count', 'pct'], axis=1, inplace=False)
gender


# In[29]:


multiplelines_count = dataset['MultipleLines'].value_counts()
multiplelines_pct= multiplelines_count / len(dataset.index)

multiplelines = pd.concat([multiplelines_count, round(multiplelines_pct,2)], axis=1)        .set_axis(['count', 'pct'], axis=1, inplace=False)
multiplelines


# In[30]:


contract_count = dataset['Contract'].value_counts()
contract_pct= contract_count / len(dataset.index)

contract = pd.concat([contract_count, round(contract_pct,2)], axis=1)        .set_axis(['count', 'pct'], axis=1, inplace=False)
contract


# In[31]:


internetservice_count = dataset['InternetService'].value_counts()
internetservice_pct= internetservice_count / len(dataset.index)

internetservice = pd.concat([internetservice_count, round(internetservice_pct,2)], axis=1)        .set_axis(['count', 'pct'], axis=1, inplace=False)
internetservice


# In[32]:


paymentmethod_count = dataset['PaymentMethod'].value_counts()
paymentmethod_pct= paymentmethod_count / len(dataset.index)

paymentmethod = pd.concat([paymentmethod_count, round(paymentmethod_pct,2)], axis=1)        .set_axis(['count', 'pct'], axis=1, inplace=False)
paymentmethod


# In[33]:


def count_by_group(data, feature, target):
    df = data.groupby([feature, target])[target].agg(['count'])
    temp = data.groupby([feature])[target].agg(['count'])
    df['pct'] = 100*df.div(temp, level = feature).reset_index()['count'].values
    return df.reset_index()


# In[34]:


count_by_group(dataset, feature = 'gender', target = 'Churn')


# In[12]:


#Stacked histogram: Age
figure = plt.figure(figsize=(15,8))
plt.hist([
        dataset[(dataset.Churn=='No')]['tenure'],
        dataset[(dataset.Churn=='Yes')]['tenure']
        ], 
         stacked=True, color = ['blue','r'],
         bins = 'auto',label = ['Stayed','Exited'],
         edgecolor='black', linewidth=1.2)
plt.xlabel('tenure (years)')
plt.ylabel('Number of customers')
plt.legend()


# In[17]:


fig, axes = plt.subplots(nrows=2, ncols=2, figsize = (15,15))
fig.subplots_adjust(left=0.2, wspace=0.6)
ax0, ax1, ax2, ax3 = axes.flatten()

ax0.hist([
        dataset[(dataset.Churn==0)]['MonthlyCharges'],
        dataset[(dataset.Churn==1)]['MonthlyCharges']
        ], 
         stacked=True, color = ['grey','r'],
         bins = 'auto',label = ['Stayed','Exited'],
         edgecolor='black', linewidth=1.2)
ax0.legend()
ax0.set_title('MonthlyCharge')

ax1.hist([
        dataset[(dataset.Churn==0)]['tenure'],
        dataset[(dataset.Churn==1)]['tenure']
        ], 
         stacked=True, color = ['grey','r'],
         bins = 'auto',label = ['Stayed','Exited'],
         edgecolor='black', linewidth=1.2)
ax1.legend()
ax1.set_title('Tenure')

ax2.hist([
        dataset[(dataset.Churn==0)]['SeniorCitizen'],
        dataset[(dataset.Churn==1)]['SeniorCitizen']
        ], 
         stacked=True, color = ['grey','r'],
         bins = 'auto',label = ['Stayed','Exited'],
         edgecolor='black', linewidth=1.2)
ax2.legend()
ax2.set_title('SeniorCitizen')

ax3.hist([
        dataset[(dataset.Churn==0)]['Contract'],
        dataset[(dataset.Churn==1)]['Contract']
        ], 
         stacked=True, color = ['grey','r'],
         bins = 'auto',label = ['Stayed','Exited'],
         edgecolor='black', linewidth=1.2)
ax3.legend()
ax3.set_title('Contract')

fig.tight_layout()
plt.show()


# In[59]:



# One-Hot encoding categorical features
list_cat = ['gender',	'Partner',	'Dependents',	'PhoneService',	'MultipleLines',	'InternetService',	'OnlineSecurity',	'OnlineBackup',	'DeviceProtection',	'TechSupport',	'StreamingTV',	'StreamingMovies',	'Contract',	'PaperlessBilling',	'PaymentMethod']
dataset = pd.get_dummies(dataset, columns = list_cat, prefix = list_cat)
dataset.head()


# In[62]:


dataset.info()


# In[64]:


features = list(dataset.drop(['Churn'], axis = 1))
target = 'Churn'


# In[65]:


dataset.info()


# In[66]:


train, test = train_test_split(dataset, test_size = 0.2, random_state = 1)

print('Number of customers in the dataset: {}'.format(len(dataset)))
print('Number of customers in the train set: {}'.format(len(train)))
print('Number of customers in the test set: {}'.format(len(test)))


# In[68]:


sc = StandardScaler()

# fit on training set
train[features] = sc.fit_transform(train[features])

# only transform on test set
test[features] = sc.transform(test[features])


# In[69]:


parameters = {'max_depth': [2, 3, 4, 6, 10, 15],
              'n_estimators': [50, 100, 300, 500]}
GB = GBSklearn()
model_GB = GridSearchCV(GB, parameters, cv = 5, n_jobs = 10, verbose = 1).fit(train[features], train[target])
pd.DataFrame(model_GB.cv_results_)


# In[72]:


print(model_GB.best_params_)


# In[73]:


model = GBSklearn(**model_GB.best_params_)
model.fit(train[features], train[target])

importances = model.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize = (15, 8))
plt.title('Feature Importances: Complete Gradient Boosting (Sklearn)')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# In[91]:


pred = model_GB.predict(test[features])
predp = model_GB.predict_proba(test[features])[:,1]

cm = confusion_matrix(test[target], pred)
plot_confusion_matrix(cm, target_names = ['Not Exited', 'Exited'], normalize = False)
plot_confusion_matrix(cm, target_names = ['Not Exited', 'Exited'], normalize = True, title = 'Confusion Matrix (Normalized)')

#temp = classification_report_to_dataframe(test[target], pred, predp, model_name = 'Gradient Boosting (Sklearn)')
#temp
#self.adj = defaultdict(list)


# In[85]:


def resample_data(data, target):
    data_1 = data[data[target] == 'Yes']
    data_0 = data[data[target] == 'No']
    percentage = len(data_1)/len(data_0)
    temp = data_0.sample(frac = percentage, random_state = 1)

    data_new = data_1.append(temp)
    data_new.sort_index(inplace = True)
    return data_new


# In[86]:


trainB = resample_data(train, target = target)
print('Number of clients in the dataset is : {}'.format(len(dataset)))
print('Number of clients in the balanced train set is : {}'.format(len(trainB)))
print('Number of clients in the test set is : {}'.format(len(test)))


# In[88]:


model_XGB = XGB(max_depth = 6,
            learning_rate = .1,
            n_estimators = 100,
            reg_lambda = 0.5,
            reg_alpha = 0,
            verbosity = 1,
            n_jobs = -1,
            tree_method = 'exact').fit(trainB[features], trainB[target])

pred = model_XGB.predict(test[features])
predp = model_XGB.predict_proba(test[features])[:,1]

importances = model_XGB.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize = (15, 8))
plt.title('Feature Importances: Balanced Extreme Gradient Boosting (XGBoost)')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# In[92]:


cm = confusion_matrix(test[target], pred)
plot_confusion_matrix(cm, target_names = ['Not Exited', 'Exited'], normalize = False)
plot_confusion_matrix(cm, target_names = ['Not Exited', 'Exited'], normalize = True, title = 'Confusion Matrix (Normalized)')

#temp = classification_report_to_dataframe(test[target], pred, predp, model_name = 'Gradient Boosting (XGBoost)', balanced = 'yes')
#temp

