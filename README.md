# House-Price Prediction
### Question: What problem are we trying to solve or answer?
### Answer: Predict the price of a house using Machine Learning

# Table of Contents
   1. [Data Details](/)
   2. [EDA](/)
   3. [Feature Selection](/)
   4. [Modeling](/)
   5. [Cross Validation of Model Results](/)
   6. [Conclusion](/)
   7. [Business Insights](/)
       
---
### Data Details 
Source: https://www.kaggle.com/datasets/harlfoxem/housesalesprediction?resource=download

About the data: This dataset contains house sale prices for King County for homes sold between May 2014 a nd May 2015

- Number of Rows   :
- Number of Columns: 21

### Features Description

| #  | Column Name      | Data Type | Description                                       |
|----|------------------|-----------|---------------------------------------------------|
| 1  | id               | integer   | Unique ID for each home sold                      |
| 2  | date             | string    | Date the house was sold                           |
| 3  | yr_built         | integer   | Year built                                        |
| 4  | bedrooms         | integer   | Number of bedrooms                                |
| 5  | bathrooms        | float     | Number of bathrooms                               |
| 6  | sqft_living      | integer   | Living area in square feet                        |
| 7  | sqft_lot         | integer   | Lot size in square feet                           |
| 8  | floors           | float     | Number of floors in the house                     |
| 9  | waterfront       | integer   | Has waterfront view (0 or 1)                      |
| 10 | view             | integer   | Quality of the view                               |
| 11 | condition        | integer   | Condition of the house                            |
| 12 | grade            | integer   | Overall grade (workmanship & design)              |
| 13 | sqft_above       | integer   | Square feet above ground                          |
| 14 | sqft_basement    | integer   | Square feet in the basement                       |
| 15 | price            | float     | Target variable: house price                      |
| 16 | yr_renovated     | integer   | Year renovated (0 if never renovated)             |
| 17 | zipcode          | integer   | Zip code area                                     |
| 18 | lat              | float     | Latitude                                          |
| 19 | long             | float     | Longitude                                         |
| 20 | sqft_living15    | integer   | Avg living area of nearest 15 neighbors           |
| 21 | sqft_lot15       | integer   | Avg lot size of nearest 15 neighbors              |


### Data Snapshot
Below is the snapshot of sample data
![Data Snapshots](https://github.com/viksaraw/House-Prices/blob/main/Pics/Data%201.png)


### Data Quality

File: 

It is important to validate the quality of data before doing Machine learning Modeling.
Following steps have been taken to assure the quality of data

#### Steps taken to validate Data Quality

	
   1. Missing Data : Verified Misssing data- No missing data found in the dataframe
   2. Duplicate data : Verified Duplicate records - No duplicate records found
   3. Wrong data values - NaN, Wrong Price Values - Validated unique value for each column
   4. Incorrect format/ Data Types - Date field is in String format, it's data type needs to be changed to Int. TO make it consistent with year renovated, it makes sense to 
      remove the month and date values and just keep the year value
   5. Drop Columns which doesn't make any sense in modeling based upon Functional knowledge - Dropped Id
       
		
### EDA = Exploratory Data Analysis
File: 
Exploratory Data Analysis (EDA) in machine learning is a crucial preliminary step that involves examining datasets to understand their characteristics, patterns, and relationships before building models. It helps uncover insights, identify anomalies, and test hypotheses, leading to a deeper understanding of the data and informed decisions about model selection and preprocessing. 


Following Steps have been Taken on EDA

1. Univariate Distributions
2. Bivariate Distributions
3. Multivariate Distribution
4. Outlier Analysis
5. Corelation Matrix


# Univariate Distribution 
Histogram for all the numerical columns- check skewness, check values which can be ignored

#### 1. Distribution for Number of Bedrooms
![Histogram for No of Bedrooms](https://github.com/viksaraw/House-Prices/blob/main/Pics/EDA%201.png)

#### 2. Distribution for Number of Bathrooms
![Number of Bathrooms](https://github.com/viksaraw/House-Prices/blob/main/Pics/EDA%202.png)

#### 3. Distribution for Sqft Living
![Sauare Foot Living](https://github.com/viksaraw/House-Prices/blob/main/Pics/EDA%203.png)

#### 4. Distribution for Grade
![Square Foot Lot](https://github.com/viksaraw/House-Prices/blob/main/Pics/EDA%204.png)

#### 5. Distribution for number of floors
![Number of floors](https://github.com/viksaraw/House-Prices/blob/main/Pics/EDA%205.png)

#### 6. Distribution for sqft_basement
![Distribution](https://github.com/viksaraw/House-Prices/blob/main/Pics/EDA%206.png)

#### 7. Distribution for year built
![View](https://github.com/viksaraw/House-Prices/blob/main/Pics/EDA%207.png)

#### 8. Distribution for square ft living 15
![View](https://github.com/viksaraw/House-Prices/blob/main/Pics/EDA%208.png)

#### Analysis from Univariate EDA
### Findings
1. Histogram is right skewed for Sqft_living - we can ignore the sqft living after 4000
2. Histogram is right skewed for sqft_above - so sqft_above > 6000 can be ignored
3. Histogram is very hight for sqft_basement with 0 value- meaning many houses don't have basement
4. histogram is right skewed for sqft_basement - meaning data set has less records for houses having more than 2000 sqft_basement

----------------------------------------------------------------------------------------------------------------------------------
### Bivariate Analysis

Scatter plot of price against various other columns to see the pattern of positive or negativer relationship with price

#### 1. Distribution for price against number of Bedrooms
![View](https://github.com/viksaraw/House-Prices/blob/main/Pics/EDA%209.png)
-------------------------------------------------------------------------------------
#### 2. Distribution for price against number of bathrooms
![View](https://github.com/viksaraw/House-Prices/blob/main/Pics/EDA%2010.png)

#### 3. Distribution for price against sqft_living
![View](https://github.com/viksaraw/House-Prices/blob/main/Pics/EDA%2011.png)

#### 4. Distribution for price against sqft_lot
![View](https://github.com/viksaraw/House-Prices/blob/main/Pics/EDA%2012.png)

#### 5. Distribution for price against grade
![View](https://github.com/viksaraw/House-Prices/blob/main/Pics/EDA%2013.png)

#### 6. Distribution for price against sqft_above
![View](https://github.com/viksaraw/House-Prices/blob/main/Pics/EDA%2014.png)

#### 7. Distribution for price against sqft_basement
![View](https://github.com/viksaraw/House-Prices/blob/main/Pics/EDA%2015.png)

#### 8. Distribution for price against year built
![View](https://github.com/viksaraw/House-Prices/blob/main/Pics/EDA%2016.png)

#### 9. Distribution for year renovated
![View](https://github.com/viksaraw/House-Prices/blob/main/Pics/EDA%2017.png)

#### 10. Distribution for sqft living 15
![View](https://github.com/viksaraw/House-Prices/blob/main/Pics/EDA%2018.png)

#### 11. Distribution for sqft lot 15
![View](https://github.com/viksaraw/House-Prices/blob/main/Pics/EDA%2019.png)

### Analysis of Bivariate Distribution
1. Analysis 1: The price increases as number of bedroom increases from 0 to 5 but decreases from 5 to 10
2. Analysis 2: Price increases with increase in bathroom 
3. Analysis 3: Price increases with increase in sqft_living
4. Analysis 4: Price for all the views are similar - may be view is a candidate column to drop
5. Analysis 5: Price increases as grade increases
6. Analysis 6: Price increases with sqft_above
-----------------------------------------------------------------------------------------------------------------------------

### Correlation Matrix 
A correlation matrix in data modeling is a table displaying correlation coefficients between variables. It quantifies the strength and direction of the relationship between variables, helping to identify patterns and improve model building by revealing which variables are most closely related. 


#### Correlation Matrix displaying columns having strong and weak coefficients
Coefficient between the columns is stronger if they show as red in the graph - the deeper red the stronger coefficients
Coefficient between the columns is weaker if they show as blue in the graph -  the deeper blue the weaker coefficients

![View](https://github.com/viksaraw/House-Prices/blob/main/Pics/EDA%2020.png)

### Analysis from Coorelation Matrix

 1. sqft_living, sqft_above, and sqft_living15 are highly correlated with each other.
 2. bedrooms, bathrooms, and grade also show strong positive correlations with many other features.
 3. zipcode has low or no correlation with most variables, suggesting it might be better treated as a categorical feature.
--------------------------------------------------------------------------------------------------------------------------------------------
### Outlier Analysis
  #### Using Box plot to see how is the distribution of Price above and beyond the 1st and 3rd Quartile
![View](https://github.com/viksaraw/House-Prices/blob/main/Pics/EDA%2021.png)

### Analysis from Outlier Analysis
1. Clearly there are outliers which needs to be removed
2. Clearly there are few outliers above 8000 sqft but those numbers are very less and can be ignored


### Overall Analysis from EDA

1. sqft_living, sqft_above, and sqft_living15 are highly correlated with each other
2. sqft_living amd grade are linarly related to Price
3. There are outliers for sqft_living and price which needs to be taken care while preprocessing
4. Clearly the feature values are not normalized and need normalization, scaling, additional columns might need to be created using logs
5. zipcode has low or no correlation with most variables, suggesting it might be better treated as a categorical feature.

-------------------------------------------------------------------------------------------------------------------------------------------

#### Feature Selection

Feature selection in machine learning is the process of selecting a subset of relevant features from a larger set to improve model performance, reduce overfitting, and enhance interpretability. It involves identifying and eliminating redundant or irrelevant features to build more efficient and accurate predictive models

Following models have been used to know the best features from the data set
1. Decision Tree Classifier
2. XBoost Classifier
3. Variance Threshold


#### Feature Importance By Decision Tree Classifier

![Using Decision Tree](https://github.com/viksaraw/House-Prices/blob/main/Pics/FS1-%20DT.png)

#### Feature importance by XBoost Regressor

![Using XBoost](https://github.com/viksaraw/House-Prices/blob/main/Pics/FS2-Xb.png)

#### Selected Features by XBoost Regressor
![Selected Features by XBoost](https://github.com/viksaraw/House-Prices/blob/main/Pics/FS3-XB2.png)

#### Feature importance by Variance Trheshold
![Using Variance Threshold](https://github.com/viksaraw/House-Prices/blob/main/Pics/FS4-Var.png)

#### Final Selected Features

|# | Column Name    |   Reason                                            |
|--|----------------|---------------------------------------------------  |
|1 |  grade         |                                                     |
|2 |  sqft_living   |                                                     |
|3 |  lat           |                                                     |
|4 |  waterfront    |                                                     |
|5 |  view          |                                                     |
|6 |  long          |                                                     | 
|7 |  yr_built      |                                                     |  
|8 |  zipcode       |                                                     | 
|9 |  condition     |                                                     | 
|10|  sqft_above    |                                                     | 
|11|  yr_renovated  |                                                     |  
|12|  bathrooms     |                                                     |
|13|  sqft_lot15    |                                                     | 
|14|  sqft_lot      |                                                     |
|15|  sqft_basement |                                                     |  
|16|  floors        |                                                     |
|17|  bedrooms      |                                                     |
|18|  sqft_living   |                                                     |



       
