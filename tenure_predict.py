import numpy as np
import pandas as pd

pd.set_option('display.max_columns', 25)

data = pd.read_csv('Telecom Customers Churn.csv')
data.head()

data.info()

for column in data.columns:
    unique_values = data[column].nunique()
    if unique_values < 20:
        print(f"Column: {column} - Unique values: {unique_values}")
        print(data[column].unique())
        print("\n")

len(data.PaymentMethod.unique())

data.PaymentMethod.unique()

df = data

def categorize_ch(row):
    if 'Yes' in row['Churn']:
        return 1
    else:
        return 0

data['churn'] = data.apply(categorize_ch, axis=1)

def categorize_pay(row):
    if 'Bank transfer (automatic)' in row['PaymentMethod']:
        return 1
    elif 'Credit card (automatic)' in row['PaymentMethod']:
        return 2
    elif 'Electronic check' in row['PaymentMethod']:
        return 3
    else:
        return 4

data['payment'] = data.apply(categorize_pay, axis=1)

def categorize_contract(row):
    if 'Month-to-month' in row['Contract']:
        return 1
    elif 'One year' in row['Contract']:
        return 12
    else:
        return 24

data['contract'] = data.apply(categorize_contract, axis=1)

def categorize_partner(row):
    if 'No' in row['Partner']:
        return 0
    else:
        return 1

data['partner'] = data.apply(categorize_partner, axis=1)

def categorize_churnvuln(row):
    if 'Yes' in row['Churn']:
        return 'Already'
    elif 'Month-to-month' in row['Contract'] and 'Electronic check' in row['PaymentMethod'] and 'No' in row['Partner']:
        return 'High'
    elif ('Month-to-month' in row['Contract'] and 'Electronic check' in row['PaymentMethod']) or ('Month-to-month' in row['Contract'] and 'No' in row['Partner']) or ('Electronic check' in row['PaymentMethod'] and 'No' in row['Partner']):
        return 'Moderate'
    elif 'Month-to-month' in row['Contract'] or 'Electronic check' in row['PaymentMethod'] or 'No' in row['Partner']:
        return 'Low'
    else:
        return 'None'

data['churnvuln'] = data.apply(categorize_churnvuln, axis=1)

def categorize_churnvulnval(row):
    if 'Yes' in row['Churn']:
        return 4
    elif 'Month-to-month' in row['Contract'] and 'Electronic check' in row['PaymentMethod'] and 'No' in row['Partner']:
        return 3
    elif ('Month-to-month' in row['Contract'] and 'Electronic check' in row['PaymentMethod']) or ('Month-to-month' in row['Contract'] and 'No' in row['Partner']) or ('Electronic check' in row['PaymentMethod'] and 'No' in row['Partner']):
        return 2
    elif 'Month-to-month' in row['Contract'] or 'Electronic check' in row['PaymentMethod'] or 'No' in row['Partner']:
        return 1
    else:
        return 0

data['churnvulnval'] = data.apply(categorize_churnvulnval, axis=1)

data.head()

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False)

data_encode = encoder.fit_transform(data[['PaymentMethod','Contract','churnvuln']])

data_encode = pd.DataFrame(data_encode,columns=encoder.get_feature_names_out())

data_encode.head()

skew_scores = {}
for column in data.select_dtypes(include=[np.number]).columns:
    skew_score = data[column].skew()
    skew_scores[column] = skew_score

skew_scores

kurt_scores = {}
for column in data.select_dtypes(include=[np.number]).columns:
    kurt_score = data[column].kurt()
    kurt_scores[column] = kurt_score

kurt_scores

import matplotlib.pyplot as plt

plt.locator_params(axis='both', integer=True)
plt.hist(data.sort_values(by=['churnvulnval'])['churnvuln'])

plt.locator_params(axis='both', integer=True)
plt.hist(data[data['churn']==1]['partner'])

plt.locator_params(axis='both', integer=True)
plt.hist(data[data['churn']==1]['payment'])

plt.locator_params(axis='both', integer=True)
plt.hist(data[data['churn']==1]['contract'])

plt.locator_params(axis='x', integer=True)
data['churnvulnval'].plot(kind='density')

import seaborn as sns

feature = ['churnvulnval','payment','partner','contract']
target = ['tenure']

plt.figure(figsize=(12,8))
sns.heatmap(pd.concat([df[feature]], axis=1).corr(), annot=True)

from feature_engine.outliers import OutlierTrimmer, Winsorizer

winsor = Winsorizer(capping_method='gaussian', tail='both', fold=3, variables=['churnvuln'])

x = df[feature]
y = df[target]

from sklearn.preprocessing import OrdinalEncoder

ord = OrdinalEncoder(categories=[['0','1','2','3','4']])

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=15)

print('Train Size : ', x_train.shape)
print('Test Size  : ', x_test.shape)

x_train_cat = x_train[['payment','partner','contract']]
x_train_ord = x_train[['churnvulnval']]

x_test_cat = x_test[['payment','partner','contract']]
x_test_ord = x_test[['churnvulnval']]

x_train_cat = pd.DataFrame(encoder.fit_transform(x_train_cat),columns=encoder.get_feature_names_out())
x_test_cat = pd.DataFrame(encoder.transform(x_test_cat),columns=encoder.get_feature_names_out())
x_train_ord = pd.DataFrame(ord.fit_transform(x_train_ord),columns=ord.get_feature_names_out())
x_test_ord = pd.DataFrame(ord.transform(x_test_ord),columns=ord.get_feature_names_out())

def plot_dist(df, col):
    fig, axes = plt.subplots(ncols = 2)

    # histogram
    sns.histplot(df[col],ax = axes[0], bins=30) # Changed from df[churn_col] to df[col]
    axes[0].set_title(f"Histogram '{col}'")
    axes[0].axvline(df[col].mean(), color = 'red', linestyle = 'dashed', label = 'mean') # Changed from df[churn_col] to df[col]
    axes[0].axvline(df[col].median(), color = 'green', linestyle = 'dashed', label = 'median') # Changed from df[churn_col] to df[col]
    axes[0].legend()

    # boxplot
    sns.boxplot(y=df[col], ax =  axes[1]) # Changed from df[churn_col] to df[col]
    axes[1].set_title(f"Boxplot '{col}'")

    plt.show()

    # skewness
    print(df[col].name + ' Kurtosis: ' + str(df[col].kurt())) # Changed from df[churn_col] to df[col]
    print(df[col].name + ' Skewness: ' + str(df[col].skew())) # Changed from df[churn_col] to df[col]
    if -0.5 <= df[col].skew() <= 0.5: # Changed from df[churn_col] to df[col]
      avg = df[col].mean() # Changed from df[churn_col] to df[col]
      std = df[col].std() # Changed from df[churn_col] to df[col]
      up_bound = avg + 3 * std
      low_bound = avg - 3 * std
      outlier = df[col][(df[col] <low_bound) | (df[col]>up_bound)] # Changed from df[churn_col] to df[col]
      print(f'Percentage of outliers: {len(outlier)*100/len(df[col]):.2f}%%') # Changed from df[churn_col] to df[col]
      print("Columns '{}' normal distribution".format(col))

    elif df[col].skew() > 0.5 and df[col].skew() <= 1 : # Changed from df[churn_col] to df[col]
      q1 = df[col].quantile(0.25) # Changed from df[churn_col] to df[col]
      q3 = df[col].quantile(0.75) # Changed from df[churn_col] to df[col]
      iqr = q3-q1
      up_bound = q3 + 1.5*iqr
      low_bound = q1 - 1.5*iqr
      outlier = df[col][(df[col] <low_bound) | (df[col]>up_bound)] # Changed from df[churn_col] to df[col]
      print(f'Percentage of outliers: {len(outlier)*100/len(df[col]):.2f}%%') # Changed from df[churn_col] to df[col]
      print("Columns '{}' right moderately skewed".format(col))

    elif df[col].skew() > 1: # Changed from df[churn_col] to df[col]
      q1 = df[col].quantile(0.25) # Changed from df[churn_col] to df[col]
      q3 = df[col].quantile(0.75) # Changed from df[churn_col] to df[col]

for i in [0]:
  plot_dist(x_train,feature[i])

print('x_train_cat : ',len(x_train_cat))
print('x_train_ord : ',len(x_train_ord))
print('y_train : ',len(y_train))

x_train_fix = np.concatenate([x_train_cat,x_train_ord], axis=1)
x_test_fix = np.concatenate([x_test_cat,x_test_ord], axis=1)

columns = list(encoder.get_feature_names_out()) + list(ord.get_feature_names_out())

x_train_final = pd.DataFrame(x_train_fix,columns = list(columns))
x_test_final = pd.DataFrame(x_test_fix,columns = list(columns))

x_train_final.head()

from sklearn.linear_model import LinearRegression

model = LinearRegression(fit_intercept=False)

model.fit(x_train_final, y_train)

y_pred_train = model.predict(x_train_final)
y_pred_test = model.predict(x_test_final)

from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, r2_score)

print('MAE train = ', mean_absolute_error(y_train, y_pred_train))
print('MAE test = ', mean_absolute_error(y_test, y_pred_test))

print('RMSE train = ', np.sqrt(mean_squared_error(y_train, y_pred_train)))
print('RMSE test = ', np.sqrt(mean_squared_error(y_test, y_pred_test)))

print('r2score train = ', r2_score(y_train, y_pred_train))
print('r2score test = ', r2_score(y_test, y_pred_test))

df.tenure.mean()

model.intercept_

from feature_engine.transformation import BoxCoxTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

categorical_features = ["payment","partner","contract"]
categorical_transformer = OneHotEncoder(sparse_output=False)

ordinal_feature = ["churnvulnval"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, categorical_features),
        ('passthrough','passthrough',ordinal_feature)
    ]
)

reg = Pipeline(
    steps=[("preprocessor", preprocessor), ("model", LinearRegression(fit_intercept=False))]
)

pd.DataFrame(preprocessor.fit_transform(x_train, y_train),columns=preprocessor.get_feature_names_out())

reg.fit(x_train, y_train)

import pickle

with open('pipeline_model.pkl', 'wb') as file_1:
  pickle.dump(reg, file_1)

with open('pipeline_model.pkl', 'rb') as file_2:
    model_pipe = pickle.load(file_2)

x_train.describe()

data_inf = pd.DataFrame(data)

inf_pred = model_pipe.predict(data_inf)

data_inf['PredictedTenure'] = inf_pred

data_inf
