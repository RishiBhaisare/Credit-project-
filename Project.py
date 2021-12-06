

#Importing important libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc("font", size=14)
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

dataset=pd.read_csv('C:/Users/DELL/Desktop/Data Science/banking_train.csv')

# To See the Size of the Dataset from different aspects
dataset.head()
dataset.tail()
dataset.info()
dataset.shape

# Creating a list to collect the names of variables
numeric_cols=[];
character_cols=[];
column_name=list(dataset.columns.tolist())
for columns in column_name:
    if dataset[columns].dtype in ('int64','float64'):
        numeric_cols.append(columns)
        print(dataset[columns])
    else:
        character_cols.append(columns) 
        print(dataset[columns])

#Data Description Report
desc_stat=dataset.describe()
skew_chk=dataset.skew()
cov=dataset.cov()
corr=dataset.corr()        
        
#Missing Value Report
total_miss=dataset.isnull().sum().sort_values(ascending=False)
miss_pct=((dataset.isnull().sum()/dataset.isnull().count())*100).sort_values(ascending=False)
missing_info=pd.concat([total_miss,miss_pct],axis=1,keys=['Miss_Count','Miss_PCT'])
missing_info=missing_info.reset_index()
missing_info.rename(columns={'index':'Variables'},inplace=True)

#Dropping the Variable as a selected percentage
drop_vars=missing_info[missing_info['Miss_PCT']>4].Variables
dataset.drop(drop_vars,axis=1,inplace=True) 

'''Null Values Dataframe'''
null_vals=pd.DataFrame(dataset.isnull().sum().sort_values(ascending=False))
null_vals.columns=['Null_Count']
null_vals=null_vals.reset_index()
null_vals.rename(columns={'index':'Variables'},inplace=True)

"Value Imputation"
#Impute by Mean 
for column in null_vals[null_vals['Null_Count']>0].Variables:
    if dataset[column].dtype in ('int64','float64'):
        dataset[column]=dataset[column].fillna(dataset[column].mean())
    elif dataset[column].dtype=='O':
        dataset[column]=dataset[column].fillna('NA')
        
#Outlier Report

Outliers=[]
data=[]
for cols in column_name:
    if dataset[cols].dtype=='int64' or dataset[cols].dtype=='float64':
        Q1=dataset[cols].quantile(0.25)
        Q3=dataset[cols].quantile(0.75)
        IQR=Q3-Q1
        Low_Boundary=Q1-(1.5*IQR)
        High_Boundary=Q3+(1.5*IQR)
        low_outliers=(dataset[cols]<Low_Boundary).sum()
        high_outliers=(dataset[cols]>High_Boundary).sum()
        data=[cols,low_outliers,high_outliers,low_outliers+high_outliers]
        Outliers.append(data)
        
Outliers_df=pd.DataFrame(Outliers,columns=['Feature','Low_Boundary_Otlrs','High_Boundary_Otlrs','Total_Otlrs'])   

#Outlier Treatment (Make it in a form of a function)

for cols in Outliers_df["Feature"]:
    dataset[cols].clip(lower=Low_Boundary,inplace=True)
    dataset[cols].clip(lower=Low_Boundary,inplace=True)

#Complete Description - EDA


'''Univariate Analysis'''
# Histograms for Univariate Analysis of Normal Distribution
dataset.hist(bins=50, figsize=(20,15))
plt.savefig("attribute_histogram_plots.jpg")
plt.show()

# Exploring the imbalance in data

# Y is quite imbalanced, lets see the subscription rate
count_no_sub = len(dataset[dataset['y']==0])
count_sub = len(dataset[dataset['y']==1])
pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
print("percentage of no subscription is", pct_of_no_sub*100)
pct_of_sub = count_sub/(count_no_sub+count_sub)
print("percentage of subscription", pct_of_sub*100)

#Lets see the subscription division as per several groups (Numerical)
subs_data=dataset.groupby('y').mean()

#Lets see the subscription division as per Variable
# Visualized
pd.crosstab(dataset.job,dataset.y).plot(kind='bar')
plt.title('Purchase Frequency for Job Title')
plt.xlabel('Job')
plt.ylabel('Frequency of Purchase')

# Numerical Way
def analysis(cat_var):
    Analysis=pd.DataFrame({"Y_Count": dataset.groupby(cat_var).count().y, "Y_Sum": dataset.groupby(cat_var).sum().y})
    Analysis['Pct']=(Analysis['Y_Sum']/Analysis['Y_Count'])*100
    Analysis=Analysis.sort_values(by=['Pct'])
    Analysis['Cats']=Analysis.index
    return Analysis

#Education
analysis_job=analysis('education')
sns.catplot(x='Pct',y='Cats',data=analysis_job,kind='bar')

'''Feature Reduction Decision'''

subs_data_job_c=dataset.groupby('job').count()
subs_data_job_m=dataset.groupby('job').mean()

subs_data_edu_c=dataset.groupby('education').count()
subs_data_edu_m=dataset.groupby('education').mean()

subs_data_marital_c=dataset.groupby('marital').count()
subs_data_marital_m=dataset.groupby('marital').mean()

'''Bivariate Analysis'''
corr=dataset.corr()
sns.heatmap(corr, 
            xticklabels = corr.columns.values,
            yticklabels = corr.columns.values,
            annot = True);  

#Start Observing the Distribution of Variables one by one (Univariate Analysis)

d=sns.pairplot(dataset,diag_kind='kde')
d.savefig("Pair_plots.jpg")

#For Numerical Data
dataset.duration.plot.line()
dataset.campaign.plot.line()

%matplotlib inline
pd.crosstab(dataset.job,dataset.y).plot(kind='bar')
plt.title('Purchase Frequency for Job Title')
plt.xlabel('Job')
plt.ylabel('Frequency of Purchase')

'''
Observation :
The frequency of purchase of the deposit depends a 
great deal on the job title. 
Thus, job title can be a good predictor of the outcome variable.
'''

'''Deleting unnecessary Variables'''
#Creating a copy of the original data frame
dataset_treated = dataset.copy()
#Dropping the unknown job level
dataset_treated = dataset_treated[dataset_treated.job != 'unknown']
#Dropping the unknown marital status
dataset_treated = dataset_treated[dataset_treated.marital != 'unknown']
#Dropping the unknown and illiterate education level
dataset_treated = dataset_treated[dataset_treated.education != 'unknown']
dataset_treated = dataset_treated[dataset_treated.education != 'illiterate']
#Deleting the 'default' column
del dataset_treated['default']
#Deleting the 'duration' column
del dataset_treated['duration']
#Dropping the unknown housing loan status
dataset_treated = dataset_treated[dataset_treated.housing != 'unknown']
#Dropping the unknown personal loan status
dataset_treated = dataset_treated[dataset_treated.loan != 'unknown']

#Combining entrepreneurs and self-employed into self-employed
dataset_treated.job.replace(['entrepreneur', 'self-employed'], 'self-employed', inplace=True)
#Combining administrative and management jobs into admin_management
dataset_treated.job.replace(['admin.', 'management'], 'administration_management', inplace=True)
#Combining blue-collar and tecnician jobs into blue-collar
dataset_treated.job.replace(['blue-collar', 'technician'], 'blue-collar', inplace=True)
#Combining retired and unemployed into no_active_income
dataset_treated.job.replace(['retired', 'unemployed'], 'no_active_income', inplace=True)
#Combining services and housemaid into services
dataset_treated.job.replace(['services', 'housemaid'], 'services', inplace=True)
#Combining single and divorced into single
dataset_treated.marital.replace(['single', 'divorced'], 'single', inplace=True)
#Combining basic school degrees
dataset_treated.education.replace(['basic.9y', 'basic.6y', 'basic.4y'], 'basic_school', inplace=True)

#Dropping NAs from the dataset
dataset_treated = dataset_treated.dropna()

''' Creating Dummy Features '''

dataset_treated = pd.get_dummies(dataset_treated, drop_first=True)
dataset_dummies = pd.get_dummies(dataset, drop_first=True)
dataset_treated=dataset_dummies.copy()

#Feature Scaling

dataset_treated_temp=dataset_treated.drop('y',axis=1)
column_name=list(dataset_treated_temp.columns.tolist())
my_scaler=StandardScaler()
my_scaler.fit(dataset_treated_temp)
dataset_scaled_temp=my_scaler.transform(dataset_treated_temp)
dataset_scaled_temp=pd.DataFrame(dataset_scaled_temp,columns=column_name)
dataset_scaled=dataset_scaled_temp.join(dataset_treated['y'])

#Splitting the variables into predictor and target variables
X = dataset_scaled.drop('y', axis=1)
y = dataset_scaled.y

#Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Feature Selection 
sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))
sel.fit(X_train, y_train)
selected_feat= X_train.columns[(sel.get_support())]
print(selected_feat)
selected_feat_df=pd.DataFrame(selected_feat)

import numpy as np
from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Model Evaluation metrics 
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
print('Accuracy Score : ' + str(accuracy_score(y_test,y_pred)))
print('Precision Score : ' + str(precision_score(y_test,y_pred)))
print('Recall Score : ' + str(recall_score(y_test,y_pred)))
print('F1 Score : ' + str(f1_score(y_test,y_pred)))


# calculate the fpr and tpr for all thresholds of the classification
import sklearn.metrics as metrics
probs = classifier.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'g', label = 'AUC = %0.3f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'g-')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig("ROC.jpg")
plt.show()

'''Hyperparameter Optimization'''
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

# Use the random grid to search for best hyperparameters
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)
rf_random.best_params_

y_pred_GS = rf_random.predict(X_test)

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy

best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, X_train, y_train)

# New Model Evaluation metrics 
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
print('Accuracy Score : ' + str(accuracy_score(y_test,y_pred_GS)))
print('Precision Score : ' + str(precision_score(y_test,y_pred_GS)))
print('Recall Score : ' + str(recall_score(y_test,y_pred_GS)))
print('F1 Score : ' + str(f1_score(y_test,y_pred_GS)))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# calculate the fpr and tpr for all thresholds of the classification
import sklearn.metrics as metrics
probs = classifier.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig("ROC.jpg")
plt.show()

# Applying PCA
from sklearn.decomposition import PCA
pca = decomposition.PCA()
pca = PCA(n_components = 8)
X_train_PCA = pca.fit_transform(X_train)
X_test_PCA = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
Values=pd.DataFrame(column_name, rf.feature_importances_)
Values=Values.reset_index()
Values.sort_values(by=['index'], inplace=True,ascending=False)
plt.barh(column_name, rf.feature_importances_)

'''
Working on K-Fold Cross Validation
'''

# Applying k-Fold Cross Validation
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
Log_Classifier=classifier.fit(X_train, y_train)
CART_Analysis = DecisionTreeClassifier(min_samples_split=10,min_samples_leaf=5).fit(X_train, y_train)
randomforest = RandomForestClassifier(n_estimators=70).fit(X_train, y_train)

from sklearn.model_selection import cross_val_score

accuracies_log = cross_val_score(estimator = Log_Classifier, X = X_train, y = y_train, cv = 10)
accuracies_log.mean()
accuracies_log.std()


accuracies_CART = cross_val_score(estimator = CART_Analysis, X = X_train, y = y_train, cv = 10)
accuracies_CART.mean()
accuracies_CART.std()

accuracies_RF = cross_val_score(estimator = randomforest, X = X_train, y = y_train, cv = 10)
accuracies_RF.mean()
accuracies_RF.std()