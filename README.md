# TermDeposit(Portugal Financial Institution)

## Financial Analysis

### Project Overview: TermDeposit Analysis(A deposit is taken or not)

This project aims to predict client subscriptions to term deposits based on demographic, economic, and social factors using data collected by a bank.

Initially processed in Excel and Pocessed in Jupyter Notebook, the dataset features variables such as age, job type, balance, and loan status.

The analysis utilizes Python and its libraries for data manipulation, visualization, and machine learning.

Algorithms such as SVM, decision trees, random forest, and logistic regression are applied to build predictive models.

## Objectives

1.)The primary objective is to predict TermDeposit subscriptions accurately.

2.)Secondary goals include identifying key predictors influencing subscription decisions and providing actionable insights for targeted marketing strategies.

## Tools Used
Python and its libraries (Pandas, NumPy, Matplotlib, Seaborn) facilitate data handling, visualization, and statistical analysis.

Scikit-learn is employed for machine learning tasks, integrating algorithms for predictive modeling.

Jupyter Notebook serves as the environment for interactive data exploration, model development, and result interpretation.

## Exploratory Data Analysis (EDA) Steps
### 1.Find Unwanted Columns

There are total 17 columns in the dataset:-

- Housing:Has a house loan or not(most of the data have "no")
- 
- Loan:Having personal loan or not(most of them does not have personal loan)
- 
- pdays:More than:days passed after the client called customer(93% of the data is "-1")
- 
- default:98% of the column is "no",this column does not tell us anything
- 
are unwanted columns

### 2.Find Missing Values

There are no missing values for each column

### 3.Find Features with One Value

Identified all the columns which are with unique values from the datatset. 

### 4.Explore the Categorical Features

For categorical variables, examined the unique values and their frequencies.

### 5.Find Categorical Feature Distribution

Visualized the distribution of each categorical feature using bar plots or count plots.

### 6.Relationship Between Categorical Features and Label

Explored how categorical features relate to the target variable (TermDeposit). This is done using stacked bar plots or cross-tabulations.

key insights from plotting are:-

1.from job,the blue-collars are mostly contacted,followed by managment.

2.from maritial,married persons are mostly contacted.

3.from education,the persons who completed secondary education are been contacted more.

4.from default,the persons who has credits from their bank are very less(in the contacted list).

5.from housing,most of the people who are contacted has house loans.

6.from loan,most of them dose not have personal loan,who are contacted.

7.from contact,most of the people who were bank contacted,most of them contain mobile phones.

8.from month,the bank employyes contacted to take the term deposit,the graph usually increased to contact most.no.of people in may month.

9.from poutcome,if a member survey is happening the person did not give particular feedback,thus the unknown is high.

### 7.Explore the Numerical Features

Examine summary statistics (mean, median, min, max) for numerical variables.

key insights from plotting are:-

1.the blue-collars are the most contacted but,the managment job type who told a yes to term deposit are more,compared to blue-collars.

2.the married ones are who mostly said yes to the term Deposit.

3.the education who studied secondary is called most and mostly 10% of them said yes to a term-deposit.

4.the persons who does not have credits in their bank are most contactes and most of them yes to the term deposit.

5.the persons who have housing loan is less taken term deposit compared to customer who does not have a housing loan.

6.the costomer who has personal loan did not take term deposit,the person who has a loan did not take term-deposit seriously.

7.most of the people who were bank contacted,most of them contain mobile phones,most of people updated to new technology.

8.the may month is most contacted and most term-deposits are taken on may month itself.

9.so,the costumers who took term-deposit from last campaign,most of them took the campaign again.(this costumer details are useful,helps us in buisness).

### 8.Find Discrete Numerical Features

number of distinct values are zero.

### 9.Relation Between Discrete Numerical Features and Labels

Not Applicable

### 10.Find Continuous Numerical Features

Continous features count is also 'zero'

### 11.Distribution of Continuous Numerical Features

Visualized the distribution of each continuous numerical feature using histograms or kernel density plots.

here we see:-

- age and day are so close to normal distribution.

- balance,duration,campaign,pdays,previous are right skewed and may have outliers.

### 12.Relation Between Continuous Numerical Features and Labels

Explore how continuous numerical features vary with respect to the target variable (TermDeposit), using box plots or scatter plots.

### 13.Find Outliers in Numerical Features

Detected and handled outliers in numerical features, using box plots or statistical methods like Z-score or IQR.

o,age and day are considered to be normal distribution

- there are no outliers in day,no need to worry about it

- there are outliers in the age coloum but there is no age limit to take a deposit.

### 14.Explore Correlation Between Numerical Features

Compute pairwise correlations between numerical features, visualizing them using a correlation matrix or heatmap.

### 15.Find Pair Plot

Created a pair plot (scatterplot matrix) of numerical features to visualize relationships between them.

no insights are found by using scatter plot

### 16.Check if the Dataset is Balanced

Determine the distribution of the target variable (TermDeposit). It is a classification problem, checked the balance between classes. Visualized this with a bar plot.

# THE DATASET IS IMBALANCED(91%-09%)

## I choose to do over-sampling

``` PYTHON

oversampler = RandomOverSampler(random_state=42)

X_resampled, y_resampled = oversampler.fit_resample(df.drop(columns=['y']), df['y'])

df_resampler = pd.concat([X_resampled, y_resampled], axis=1)

print("Class Distribution after Oversampling:")

print(df_resampler['y'].value_counts() / df_resampler.shape[0])

```

after the data set is partfectly balanced.Then,prepare the dataset to train and test machine learning models.

# Data Preparation

- splitted the data into numerical_columns and catagorical_columns

- Standard Scaling and one_hot encoder applied,Mapping function applied on the catagorical_data and produced

- The numerical data is standardized using "Standard Scalar"

# THE DATA READY FOR MACJINE LEARNING MODEL SELECTION

## Train_data=75% AND Test_Data=25%

The machine learning models used are:-

## Logistic Regression:-

This model predicted the data accuracy as 84%

### Accuracy=84%

## Decision Tree Classifier:-

Cross-Validatio=15,which splits the data into equal folds

{'max_depth': [2, 3, 4,5,6,7,8,9,10],
              'criterion': ['gini', 'entropy'],
             'max_leaf_nodes':[4,6,8,10,12,14]}

### Accuracy around 80%

## Random Forest Classifier:-

{'max_depth': [2, 3, 4,6,8], the maximium depth taken is up to 8 nodes.

'max_features': ['auto', 'sqrt'], Featues taken for splitting the node
'criterion': ['gini', 'entropy'], measuring quality of the split
'oob_score' : [True, False], Out-of-bag consideration
'n_estimators':[10,15,20,25,30]} no.of trees taken for consideration

33#

### ACCURACY OF 86%(Highest)

## Gausian NB:-

- gausian NB classification report says that

- model performance in the training data is good with 76% and test data Average with 65%

### Accuracy=65%

## KNN:-

- K-value=30

- Performed equally on train data and teat data

### Model Performance=84%

# Conclusion

the model produces accuracy with 75%,82%,85%,87% ranges in both testing and training and testing parts respectively.

so the model has good performance and good accuracy rate.

# 87% Accuracy produced by Random Forest
