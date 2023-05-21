#!/usr/bin/env python
# coding: utf-8

# # E-Commerce Shipping Dataset : 
# 
# ### Product Shipment Delivered on time or not ; To Meet E-Commerce Customer Demand
# 
# 
# The data contains the following information:  
# 1. **ID** : ID Number of Customers.
# 2. **Warehouse block** : The Company have big Warehouse which is divided in to block such as A,B,C,D,E.
# 3. **Mode of shipment** :The Company Ships the products in multiple way such as Ship, Flight and Road.
# 4. **Customer care calls** : The number of calls made from enquiry for enquiry of the shipment.
# 5. **Customer rating** : The company has rated from every customer. 1 is the lowest (Worst), 5 is the highest (Best).
# 6. **Cost of the product** : Cost of the Product in US Dollars.
# 7. **Prior purchases** : The Number of Prior Purchase.
# 8. **Product importance** : The company has categorized the product in the various parameter such as low, medium, high.
# 9. **Gender** : Male and Female.
# 10. **Discount offered** : Discount offered on that specific product. 
# 11. **Weight in gms** : It is the weight in grams.
# 12. **Reached on time** : It is the target variable, where 1 Indicates that the
# product has NOT reached on time and 0 indicates it has reached on time.
# 

# # Data Pre-processing

# ## Load & Describe Data

# ### Import library

# In[53]:


import numpy as np
import pandas as pd

#Visualization
import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('fivethirtyeight') # use style fivethirtyeight
import seaborn as sns
from matplotlib import rcParams
import warnings 
warnings.filterwarnings("ignore")

# Scaling
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Selection
from scipy.stats import chi2_contingency

# Splitting the data into Train and Test
from sklearn.model_selection import train_test_split

# Algorithm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Evaluation metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report

# Hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


# ### Import file

# In[54]:


df = pd.read_csv(r"C:\Users\HP\Downloads\Train.csv")


# ### Rename column target

# In[55]:


df.rename(columns={'Reached.on.Time_Y.N':'is_late'}, inplace=True)
df.head()


# Because of the target's name is too long, so we simplify the name to ease the next step.

# ### Get the shape of dataset

# In[56]:


df.shape


# ### Get list of columns

# In[57]:


df.columns


# ### Change all column names to lower case

# In[58]:


df.columns = df.columns.str.lower()


# ### Get dataset information

# In[59]:


df.info()


# In[60]:


df.describe()


# **Based on the information above :**
# 1. Dataframe has 10999 rows and 12 columns.
# 2. No missing values are found.
# 3. There are only 2 data types, integer and object.
# 4. Classification target `is_late` and others we call features.

# ### Separate numeric & categorical column

# In[61]:


# Categorical data
categorical = ['warehouse_block','mode_of_shipment','product_importance', 'gender', 'is_late', 'customer_rating']
# Numerical data
numeric = ['customer_care_calls', 'cost_of_the_product', 'prior_purchases', 'discount_offered', 'weight_in_gms']


# ## Data Cleansing & Feature Engineering

# #### Reload dataset

# In[62]:


df_dt = df.copy()


# #### Identify missing values

# In[63]:


df_dt.isna().values.any() # Missing value detection


# In[64]:


df_dt.isna().sum()  # Calculate missing values


# Just for making sure that no missing values are found.

# #### Identify duplicated values

# In[65]:


# Select all duplicate rows based on all columns
df_dt[df_dt.duplicated(keep=False)] 


# In[66]:


# Select all duplicate rows based on selected column
df_dt[df_dt.duplicated(subset=['id'],keep=False)] # Display all duplicated rows based on column 'id'


# Luckily, there is no duplicated value in the dataframe.

# #### Identify outliers

# In[67]:


# Identify using boxplot
plt.figure(figsize=(20,8))
for i in range(0,len(numeric)):
    plt.subplot(1, len(numeric), i+1)
    sns.boxplot(y=df_dt[numeric[i]], color='orange')
    plt.tight_layout()


# In[68]:


# Identify outlier using IQR
for col in numeric:
    
    # Menghitung nilai IQR
    Q1 = df_dt[col].quantile(0.25)
    Q3 = df_dt[col].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define value 
    nilai_min = df_dt[col].min()
    nilai_max = df_dt[col].max()
    lower_lim = Q1 - (1.5*IQR)
    upper_lim = Q3 + (1.5*IQR)
    
    # Identify low outlier

    if (nilai_min < lower_lim):
        print('Low outlier is found in column',col,'<', lower_lim,'\n')
        #display total low outlier 
        print('Total of Low Outlier in column',col, ':', len(list(df_dt[df_dt[col] < lower_lim].index)),'\n')
    elif (nilai_max > upper_lim):
        print('High outlier is found in column',col,'>', upper_lim,'\n')
        #display total high outlier 
        print('Total of High Outlier in column',col, ':', len(list(df_dt[df_dt[col] > upper_lim].index)),'\n')
        
    else:
        print('Outlier is not found in column',col,'\n')
        


# We found outliers in `discount_offered` & `prior_purchases` with almost 30% of data.

# In[69]:


# We handle outlier with replace the value with upper_bound or lower_bound 
for col in ['prior_purchases', 'discount_offered']:
    # Initiate Q1
    Q1 = df_dt[col].quantile(0.25)
    # Initiate Q3
    Q3 = df_dt[col].quantile(0.75)
    # Initiate IQR
    IQR = Q3 - Q1
    # Initiate lower_bound & upper_bound 
    lower_bound = Q1 - (IQR * 1.5)
    upper_bound = Q3 + (IQR * 1.5)

    # Filtering outlier & replace with upper_bound or lower_bound 
    df_dt[col] = np.where(df_dt[col] >= upper_bound,
                         upper_bound, df_dt[col])
    df_dt[col] = np.where(df_dt[col] <= lower_bound,
                         lower_bound, df_dt[col])


# In[70]:


# Identify using boxplot
plt.figure(figsize=(20,8))
for i in range(0,len(numeric)):
    plt.subplot(1, len(numeric), i+1)
    sns.boxplot(y=df_dt[numeric[i]], color='orange')
    plt.tight_layout()


# In[71]:


sns.boxplot(y= df_dt['prior_purchases'], color = 'orange', orient = 'h');


# In[72]:


sns.boxplot(y= df_dt['discount_offered'], color = 'orange', orient = 'h');


# We didn't remove the outliers, but replacing with upper bound and lower bound. And we can see in the visualization above, there is no outliers detected.

# #### Feature Transformation : Log transform

# In[73]:


# Check data distribution
plt.figure(figsize=(20,5))
for i in range(0,len(numeric)):
    plt.subplot(1, len(numeric), i+1)
    sns.distplot(df_dt[numeric[i]], color='orange')
    plt.tight_layout()


# In[74]:


# Apply log transformation
for col in numeric:
    df_dt[col] = (df_dt[col]+1).apply(np.log)


# In[75]:


# Visualize after log transformation
plt.figure(figsize=(20,5))
for i in range(0,len(numeric)):
    plt.subplot(1, len(numeric), i+1)
    sns.distplot(df_dt[numeric[i]], color='orange')
    plt.tight_layout()


# #### Feature Scaling : Standardization

# In[76]:


# Apply standardization
for col in numeric:
    df_dt[col]= StandardScaler().fit_transform(df_dt[col].values.reshape(len(df_dt), 1))


# In[77]:


df_dt.describe()


# #### Feature Selection : Chi squared method

# In[78]:


# Selection for categorial feature
# Import module
from scipy.stats import chi2_contingency

category = ['warehouse_block','mode_of_shipment','product_importance', 
            'gender','customer_rating']
chi2_check = []
# Iteration
for col in category:
    # If pvalue < 0.05 
    if chi2_contingency(pd.crosstab(df_dt['is_late'], df_dt[col]))[1] < 0.05 :
        chi2_check.append('Reject Null Hypothesis')
    # If pvalue > 0.05
    else :
        chi2_check.append('Fail to Reject Null Hypothesis')
        
# Make the result into dataframe
res = pd.DataFrame(data = [category, chi2_check]).T
# Rename columns
res.columns = ['Column', 'Hypothesis']
res


# In[79]:


# Adjusted P-Value use the Bonferroni-adjusted method

# Initiate empty dictionary
check = {}
# Iteration for product_importance column
for i in res[res['Hypothesis'] == 'Reject Null Hypothesis']['Column']:
    # One hot encoding product_importance column
    dummies = pd.get_dummies(df_dt[i])
    # Initiate Bonferroni-adjusted formula
    bon_p_value = 0.05/df_dt[i].nunique()
    for series in dummies:
        if chi2_contingency(pd.crosstab(df_dt['is_late'], dummies[series]))[1] < bon_p_value:
            check['{}-{}'.format(i, series)] = 'Reject Null Hypothesis'
        else :
            check['{}-{}'.format(i, series)] = 'Fail to Reject Null Hypothesis'
# Make the result into dataframe
res_chi_ph = pd.DataFrame(data=[check.keys(), check.values()]).T
# Rename the columns
res_chi_ph.columns = ['Pair', 'Hypothesis']
res_chi_ph


# From the result above, `product_importance` with **high** category has a correlation with our target.

# #### Feature Encoding : One hot encoding

# In[80]:


# one hot encoding feature product_importance and keep high category
onehots = pd.get_dummies(df_dt['product_importance'], prefix = 'product_importance')
df_dt = df_dt.join(onehots)

# drop all categorical columns & 'id, except product_importance_high
df_dt.drop(columns=['warehouse_block','gender','mode_of_shipment',
                   'product_importance', 'product_importance_low',
                   'product_importance_medium','id'], inplace = True)
# check dataframe after encoding
df_dt.info()


# ## Exploratory Data Analysis (EDA)

# In[81]:


# Copy dataset
df_eda = df.copy()


# ### Target Visualization

# In[82]:


delay = pd.DataFrame(df_eda.groupby(['is_late'])['id'].count()/len(df_eda)).reset_index()
plt.pie(delay['id'],labels=delay['is_late'],autopct='%.2f%%');


# The class of target looks balance.

# ### Descriptive Statistic

# #### Categorical values

# In[83]:


# for categorical column
for col in categorical:
    print('Value count kolom', col, ':')
    print(df_eda[col].value_counts())
    print()


# In[84]:


# Plot categorical columns
for col in categorical:
    plt.figure(figsize=(15, 5))
    
    plt.subplot(141);
    sns.countplot(df_eda[col], palette = 'colorblind', orient='v');
    plt.title('Countplot')
    plt.tight_layout();
    
    
    plt.subplot(143);
    df_eda[col].value_counts().plot.pie(autopct='%1.2f%%');
    plt.title('Pie chart')
    plt.legend()  


# **Summary :**
# - **Warehouse_Block** has 5 unique values and dominated with `Warehouse_block_f`.
# - **Mode_of_Shipment** has 3 unique values and mostly used ship.
# - **Product_importance** has 3 unique values and mostly priority of products are low.
# - Female customers are often shopping than male.
# 
# 
# 
# 

# #### Numeric values

# In[85]:


df_eda[numeric].describe()


# **Summary :**
# - Distribution of **customer_care_calls**, **Customer_rating**, **Cost_of_the_Product**, **Prior_purchases** look normal, beacuse the mean and the median are close, while  **discount_offered** and **weight_in_grams** are indicated skewed.
# 
# 
# 

# #### Correlation Heatmap

# In[86]:


plt.figure(figsize=(7,6));
sns.heatmap(df_eda.corr(), annot = True, fmt = '.2f', cmap = 'Reds');


# Based on the *Correlation heatmap* above :  
# 1. Target *is_late* has a moderate positive correlation with *discount_offered* & weak negative correlation with *weight_in_gms*.
# 2. Feature *customer_care_calss* has a weak positive correlation with *cost_of_the_product* and negative correlation with *weight_in_gms*.
# 3. Feature *discount_offered* has a moderate negative correlation with *weight_in_gms*.
# 

# #### Categorical - Categorical

# ##### Based on Gender

# In[87]:


i=1
plt.figure(figsize=(15,10))
for col in ['mode_of_shipment', 'warehouse_block', 'product_importance']:
    plt.subplot(2,2,i)
    sns.countplot(df_eda[col], hue=df_eda['gender'], palette="ch:.25")
    i+=1


# **Summary :**  
# - Total parcels of female customers in the warehouse_block are more dominant than male customers, except in warehouse_block B.

# ##### Based on Product Importance

# In[88]:


i=1
plt.figure(figsize=(15,10))
for col in ['mode_of_shipment', 'warehouse_block']:
    plt.subplot(2,2,i)
    sns.countplot(df_eda[col], hue=df_eda['product_importance'], palette="ch:.25")
    i+=1


# **Summary :**  
# - Mostly high & low priority parcels used ship.
# 
# 

# ##### Warehouse block - Mode of Shipment

# In[89]:


sns.catplot(x="warehouse_block", kind="count", hue='mode_of_shipment',
            palette="ch:.25", data=df_eda);


# ##### Based on target 'is_late'

# In[90]:


i=1
plt.figure(figsize=(15,10))
for col in ['mode_of_shipment', 'warehouse_block', 'product_importance',
            'gender','customer_rating']:
    plt.subplot(2,3,i)
    sns.countplot(df_eda[col], hue=df_eda['is_late'], palette="ch:.25")
    i+=1
    plt.legend(['on_time','late']);


# **Summary :**  
# - Most of parcels are stored in warehouse_block F.
# - The ship contributes the most late delivery.
# - Most of parcels in all shipment priority are delivered late.

# # Machine Learning Modelling & Evaluation

# ### Separate feature & target column

# In[91]:


# Inititate feature & target
X = df_dt.drop(columns = 'is_late')
y = df_dt['is_late']


# ### Split train & test data

# In[92]:


# Split Train & Test Data
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.30, random_state=42)


# ### Fit & Evaluation Model

# In[93]:


# Create function to fit model & model evaluation
def fit_evaluation(Model, Xtrain, ytrain, Xtest, ytest):
    model = Model # initiate model
    model.fit(Xtrain, ytrain) # fit the model
    y_pred = model.predict(Xtest)
    y_pred_train = model.predict(Xtrain)
    train_score = model.score(Xtrain, ytrain) # Train Accuracy
    test_score = model.score(Xtest, ytest)    # Test Accuracy
    fpr, tpr, thresholds = roc_curve(ytest, y_pred, pos_label=1)
    AUC = auc(fpr, tpr) # AUC
    return round(train_score,2), round(test_score,2), round(precision_score(ytest, y_pred),2),            round(recall_score(ytrain, y_pred_train),2),round(recall_score(ytest, y_pred),2),            round(f1_score(ytest, y_pred),2), round(AUC,2)


# #### Default Parameter

# In[94]:


# Inititate algorithm
lr = LogisticRegression(random_state=42)
dt = DecisionTreeClassifier(random_state=42)
rf = RandomForestClassifier(random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
svc = SVC(random_state=42)

# Create function to make the result as dataframe 
def model_comparison_default(X,y):  
    
    # Logistic Regression
    lr_train_score, lr_test_score, lr_pr, lrtr_re, lrte_re, lr_f1, lr_auc = fit_evaluation(
        lr, Xtrain, ytrain, Xtest, ytest)
    # Decision Tree
    dt_train_score, dt_test_score, dt_pr, dttr_re, dtte_re, dt_f1, dt_auc = fit_evaluation(
        dt, Xtrain, ytrain, Xtest, ytest)
    # Random Forest
    rf_train_score, rf_test_score, rf_pr, rftr_re, rfte_re, rf_f1, rf_auc = fit_evaluation(
        rf, Xtrain, ytrain, Xtest, ytest)
    # KNN
    knn_train_score, knn_test_score, knn_pr, knntr_re, knnte_re, knn_f1, knn_auc = fit_evaluation(
        knn, Xtrain, ytrain, Xtest, ytest)
    # SVC
    svc_train_score, svc_test_score, svc_pr, svctr_re, svcte_re, svc_f1, svc_auc = fit_evaluation(
        svc, Xtrain, ytrain, Xtest, ytest)
  
    
    models = ['Logistic Regression','Decision Tree','Random Forest',
             'KNN','SVC']
    train_score = [lr_train_score, dt_train_score, rf_train_score, 
                   knn_train_score, svc_train_score]
    test_score = [lr_test_score, dt_test_score, rf_test_score,
                  knn_test_score, svc_test_score]
    precision = [lr_pr, dt_pr, rf_pr, knn_pr, svc_pr]
    recall_train = [lrtr_re, dttr_re, rftr_re, knntr_re, svctr_re]
    recall_test = [lrte_re, dtte_re, rfte_re, knnte_re, svcte_re]
    f1 = [lr_f1, dt_f1, rf_f1, knn_f1, svc_f1]
    auc = [lr_auc, dt_auc, rf_auc, knn_auc, svc_auc]
    
    model_comparison = pd.DataFrame(data=[models, train_score, test_score, 
                                          precision, recall_train, recall_test,
                                          f1,auc]).T.rename({0: 'Model',
                                                             1: 'Accuracy_Train',
                                                             2: 'Accuracy_Test',
                                                             3: 'Precision',
                                                             4: 'Recall_Train',
                                                             5: 'Recall_Test',
                                                             6: 'F1 Score',
                                                             7: 'AUC'
                                                                                  }, axis=1)
    
    return model_comparison


# In[95]:


model_comparison_default(X,y)


# From the result above, only **Logistic Regression and SVC which are neither overfitting nor underfiting**. Logistic Regression has the highest recall. Let's see with tuned parameter.

# ### Hyperparameter

# #### Logistic Regression

# In[96]:


# List Hyperparameters 
penalty = ['l2','l1','elasticnet']
C = [0.0001, 0.001, 0.002] # Inverse of regularization strength; smaller values specify stronger regularization.
hyperparameters = dict(penalty=penalty, C=C)

# Inisiasi model
logres = LogisticRegression(random_state=42) # Init Logres dengan Gridsearch, cross validation = 5
model = RandomizedSearchCV(logres, hyperparameters, cv=5, random_state=42,  scoring='recall')

# Fitting Model & Evaluation
model.fit(Xtrain, ytrain)
y_pred = model.predict(Xtest)
model.best_estimator_


# #### Decision Tree

# In[97]:


# Let's do hyperparameter tuning using RandomizesearchCV

# Hyperparameter lists to be tested
max_depth = list(range(1,10)) 
min_samples_split = list(range(5,10)) 
min_samples_leaf = list(range(5,15)) 
max_features = ['auto', 'sqrt', 'log2'] 
criterion = ['gini','entropy']
splitter = ['best','random']

# Initiate hyperparameters
hyperparameters = dict(max_depth=max_depth, 
                       min_samples_split=min_samples_split, 
                       min_samples_leaf=min_samples_leaf,
                       max_features=max_features,
                       criterion = criterion,
                       splitter = splitter)

# Initiate model
dt_tun = DecisionTreeClassifier(random_state=42)
model = RandomizedSearchCV(dt_tun, hyperparameters, cv=10, scoring='recall',random_state=42) 
model.fit(Xtrain, ytrain)
y_pred_tun = model.predict(Xtest)
model.best_estimator_


# #### Random Forest

# In[98]:


# Initiate hyperparameters
params = {'max_depth':[50],'n_estimators':[100,150], 
          'criterion':['gini', 'entropy']}
# Initiate model
model = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                     param_grid=params,scoring='recall', cv=5)
# Fit model
model.fit(Xtrain,ytrain)
y_pred = model.predict(Xtest)
# Get best estimator
model.best_estimator_


# #### KNN

# In[99]:


#List Hyperparameters that we want to tune.
leaf_size = list(range(1,50))
n_neighbors = list(range(1,30))
p=[1,2]

#Convert to dictionary
hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)

#Create new KNN object
KNN_2 = KNeighborsClassifier()

#Use RandomizedSearchCV
clf = RandomizedSearchCV(KNN_2, hyperparameters, cv=10, scoring = 'recall')

#Fit the model
best_model = clf.fit(X,y)
# Get best estimator
clf.best_estimator_


# #### SVC

# In[100]:


# Hyperparameter lists to be tested
kernel = ['linear', 'poly', 'rbf', 'sigmoid']
C = [0.0001, 0.001, 0.002] 
gamma = ['scale', 'auto']

#Convert to dictionary
hyperparameters = dict(kernel=kernel, C=C, gamma=gamma)

# Initiate model
svc = SVC(random_state=42) 
model = RandomizedSearchCV(svc, hyperparameters, cv=5, random_state=42, 
                           scoring='recall')

# Fitting Model & Evaluation
model.fit(Xtrain, ytrain)
y_pred = model.predict(Xtest)
model.best_estimator_


# #### Tuned Parameter

# In[103]:


# Inititate best estimator
lr_tune = LogisticRegression(C=0.0001, random_state=42)
dt_tune = DecisionTreeClassifier(criterion='entropy', max_depth=4, max_features='sqrt',
                       min_samples_leaf=12, min_samples_split=6,
                       random_state=42)
rf_tune = RandomForestClassifier(max_depth=50, random_state=42)

knn_tune = KNeighborsClassifier(leaf_size=24, n_neighbors=3, p=1)

svc_tune = SVC(C=0.0001, kernel='linear', random_state=42)

# Create function to make the result as dataframe 
def model_comparison_tuned(X,y):  
    
    # Logistic Regression
    lr_train_score, lr_test_score, lr_pr, lrtr_re, lrte_re, lr_f1, lr_auc = fit_evaluation(
        lr_tune, Xtrain, ytrain, Xtest, ytest)
    # Decision Tree
    dt_train_score, dt_test_score, dt_pr, dttr_re, dtte_re, dt_f1, dt_auc = fit_evaluation(
        dt_tune, Xtrain, ytrain, Xtest, ytest)
    # Random Forest
    rf_train_score, rf_test_score, rf_pr, rftr_re, rfte_re, rf_f1, rf_auc = fit_evaluation(
        rf_tune, Xtrain, ytrain, Xtest, ytest)
    # KNN
    knn_train_score, knn_test_score, knn_pr, knntr_re, knnte_re, knn_f1, knn_auc = fit_evaluation(
        knn_tune, Xtrain, ytrain, Xtest, ytest)
    # SVC
    svc_train_score, svc_test_score, svc_pr, svctr_re, svcte_re, svc_f1, svc_auc = fit_evaluation(
        svc_tune, Xtrain, ytrain, Xtest, ytest)
    
    
    models = ['Logistic Regression','Decision Tree','Random Forest',
             'KNN','SVC']
    train_score = [lr_train_score, dt_train_score, rf_train_score, 
                   knn_train_score, svc_train_score]
    test_score = [lr_test_score, dt_test_score, rf_test_score,
                  knn_test_score, svc_test_score]
    precision = [lr_pr, dt_pr, rf_pr, knn_pr, svc_pr]
    recall_train = [lrtr_re, dttr_re, rftr_re, knntr_re, svctr_re]
    recall_test = [lrte_re, dtte_re, rfte_re, knnte_re, svcte_re]
    f1 = [lr_f1, dt_f1, rf_f1, knn_f1, svc_f1]
    auc = [lr_auc, dt_auc, rf_auc, knn_auc, svc_auc]
    
    model_comparison = pd.DataFrame(data=[models, train_score, test_score, 
                                          precision, recall_train, recall_test,
                                          f1,auc]).T.rename({0: 'Model',
                                                             1: 'Accuracy_Train',
                                                             2: 'Accuracy_Test',
                                                             3: 'Precision',
                                                             4: 'Recall_Train',
                                                             5: 'Recall_Test',
                                                             6: 'F1 Score',
                                                             7: 'AUC'
                                                                                  }, axis=1)
    
    return model_comparison


# In[104]:


model_comparison_tuned(X,y)


# Decision Tree algorithm with hyper-parameter tuning has a good balance between its score, also neither underfitting nor overfitting.

# ### Confusion matrix

# In[105]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def get_confusion_matrix(model, X_train, y_train, X_test, y_test, labels=None):
    # Train the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Set display labels
    if labels is None:
        labels = ['Negative', 'Positive']
    
    # Plot the confusion matrix using Seaborn heatmap
    plt.figure(figsize=(3, 3))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()


# In[106]:


# After hyperparameter tuning
get_confusion_matrix(dt_tune, Xtrain, ytrain, Xtest, ytest,labels=['on_time','late'])


# In[107]:


# Before hyperparameter tuning
get_confusion_matrix(dt, Xtrain, ytrain, Xtest, ytest,labels=['on_time','late'])


# ### Feature Importance

# In[108]:


feat_importances = pd.Series(dt_tune.feature_importances_, index=X.columns)
ax = feat_importances.nlargest(25).plot(kind='barh', figsize=(10, 8))
ax.invert_yaxis()

plt.xlabel('Score');
plt.ylabel('Feature');
plt.title('Feature Importance Score');


# **Recommendation for E-Commerce :**
# - The operation team should add more manpower when there is a sale program, especially for the discount more than 10% and the parcel weight is 1 - 4 Kg.
# - The parcel should not be centralized in the warehouse block F, so that the handling is not too crowded which can cause the late shipment.
# - Adding more features can imporve model performance, such as delivery time estimation, delivery date, customer address, and courier.
# 

# In[ ]:




