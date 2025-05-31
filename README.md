# House-Price Prediction
### Question: What problem are we trying to solve or answer?
### Answer: Predict the price of a house using Machine Learning

# Table of Contents
   1. [Data Details](/)
   2. [EDA](/)
   3. [Feature Selection](/)
   4. [Data Cleanup](/)
   5. [Modeling](/)
   6. [Cross Validation of Model Results](/)
   7. [Feature Engineering + Tuning](/)
   8. [Conclusion](/)
   9. [Business Insights](/)
       

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

#### Distribution for Number of Bedrooms
![Histogram for No of Bedrooms](https://github.com/viksaraw/House-Prices/blob/main/Pics/EDA%201.png)

#### Distribution for Number of Bathrooms
![Number of Bathrooms](https://github.com/viksaraw/House-Prices/blob/main/Pics/EDA%202.png)

#### Distribution for Sqft Living
![Sauare Foot Living](https://github.com/viksaraw/House-Prices/blob/main/Pics/EDA%203.png)

#### Distribution for Sqft Lot
![Square Foot Lot](https://github.com/viksaraw/House-Prices/blob/main/Pics/EDA%204.png)

#### Distribution for number of floors
![Number of floors](https://github.com/viksaraw/House-Prices/blob/main/Pics/EDA%205.png)

#### Distribution for 
![Distribution](https://github.com/viksaraw/House-Prices/blob/main/Pics/EDA%206.png)

#### Distribution for 
![View](https://github.com/viksaraw/House-Prices/blob/main/Pics/EDA%207.png)

#### Distribution for
![View](https://github.com/viksaraw/House-Prices/blob/main/Pics/EDA%208.png)

#### Distribution for
![View](https://github.com/viksaraw/House-Prices/blob/main/Pics/EDA%209.png)
-------------------------------------------------------------------------------------
#### Distribution for
![View](https://github.com/viksaraw/House-Prices/blob/main/Pics/EDA%2010.png)

#### Distribution for
![View](https://github.com/viksaraw/House-Prices/blob/main/Pics/EDA%2011.png)


#### Distribution for
![View](https://github.com/viksaraw/House-Prices/blob/main/Pics/EDA%2012.png)

#### Distribution for
![View](https://github.com/viksaraw/House-Prices/blob/main/Pics/EDA%2013.png)

#### Distribution for
![View](https://github.com/viksaraw/House-Prices/blob/main/Pics/EDA%2014.png)

#### Distribution for
![View](https://github.com/viksaraw/House-Prices/blob/main/Pics/EDA%2015.png)

#### Distribution for
![View](https://github.com/viksaraw/House-Prices/blob/main/Pics/EDA%2016.png)

#### Distribution for
![View](https://github.com/viksaraw/House-Prices/blob/main/Pics/EDA%2017.png)

#### Distribution for
![View](https://github.com/viksaraw/House-Prices/blob/main/Pics/EDA%2018.png)

#### Distribution for
![View](https://github.com/viksaraw/House-Prices/blob/main/Pics/EDA%2019.png)

#### Distribution for
![View](https://github.com/viksaraw/House-Prices/blob/main/Pics/EDA%2020.png)

#### Distribution for
![View](https://github.com/viksaraw/House-Prices/blob/main/Pics/EDA%2021.png)
