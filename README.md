# House-Price Prediction
### Question: What problem are we trying to solve or answer?
### Answer: Predict the price of a house using Machine Learning

# Table of Contents
   1. Data Details
   2. EDA
   3. Feature Selection
   4. Modeling
   5. Cross Validation of Model Results
   6. Conclusion
   7. Business Insights
       
---
### Data Details 
Source: https://www.kaggle.com/datasets/harlfoxem/housesalesprediction?resource=download

About the data: This dataset contains house sale prices for King County for homes sold between May 2014 a nd May 2015

- Number of Rows   : 21613
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

File: https://github.com/viksaraw/House-Prices/blob/main/Scripts/EDA.ipynb

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
1. Analysis 1: Histogram is right skewed for Sqft_living - we can ignore the sqft living after 4000
2. Analysis 2: Histogram is right skewed for sqft_above - so sqft_above > 6000 can be ignored
3. Analysis 3: Histogram is very hight for sqft_basement with 0 value- meaning many houses don't have basement
4. Analysis 4: Histogram is right skewed for sqft_basement - meaning data set has less records for houses having more than 2000 sqft_basement

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

 1. Analysis 1: sqft_living, sqft_above, and sqft_living15 are highly correlated with each other.
 2. Analysis 2: bedrooms, bathrooms, and grade also show strong positive correlations with many other features.
 3. Analysis 3: zipcode has low or no correlation with most variables, suggesting it might be better treated as a categorical feature.
--------------------------------------------------------------------------------------------------------------------------------------------
### Outlier Analysis
  #### Using Box plot to see how is the distribution of Price above and beyond the 1st and 3rd Quartile
![View](https://github.com/viksaraw/House-Prices/blob/main/Pics/EDA%2021.png)

### Analysis from Outlier Analysis
1. Analysis 1: Clearly there are outliers which needs to be removed
2. Analysis 2: Clearly there are few outliers above 8000 sqft but those numbers are very less and can be ignored


### Overall Analysis from EDA

1. sqft_living, sqft_above, and sqft_living15 are highly correlated with each other
2. sqft_living amd grade are linarly related to Price
3. There are outliers for sqft_living and price which needs to be taken care while preprocessing
4. Clearly the feature values are not normalized and need normalization, scaling, additional columns might need to be created using logs
5. zipcode has low or no correlation with most variables, suggesting it might be better treated as a categorical feature.

-------------------------------------------------------------------------------------------------------------------------------------------

#### Feature Selection
**Files**

By Decision Tree: https://github.com/viksaraw/House-Prices/blob/main/Scripts/FeatureSelectionByDecisionTree.ipynb  <br>
By Other Models: https://github.com/viksaraw/House-Prices/blob/main/Scripts/Feature%20Selection.ipynb

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

|# | Column Name    |  
|--|----------------|
|1 |  grade         |                                                     
|2 |  sqft_living   |                                                     
|3 |  lat           |                                                     
|4 |  waterfront    |                                                     
|5 |  view          |                                                     
|6 |  long          |                                                      
|7 |  yr_built      |                                                       
|8 |  zipcode       |                                                      
|9 |  condition     |                                                      
|10|  sqft_above    |                                                      
|11|  yr_renovated  |                                                       
|12|  bathrooms     |                                                     
|13|  sqft_lot15    |                                                      
|14|  sqft_lot      |                                                     
|15|  sqft_basement |                                                       
|16|  floors        |                                                     
|17|  bedrooms      |                                                     
|18|  sqft_living   |                                                     

### Modeling
Now that the Data is cleaned up and Features are decided, it is time to add different models to this and see their performance

I will apply the following models and measure their metrics

1. Decision Tree
2. XBoost Regression
3. Random Forest
4. Logistic Regression
5. Linear Regression
6. Ridge Regression
7. Lasso Regression


#### 1. Decision Tree
File : https://github.com/viksaraw/House-Prices/blob/main/Scripts/Decision%20Tree.ipynb

First of all the Decision Tree Regressor was used. Since this easy to understand and visualize, making them valuable for decision-making. 

**Advantages**:
1. Interpretability: Decision trees are easy to understand and visualize, making them valuable for decision-making. 
2. No data preprocessing: They can handle data without normalization or scaling. 
3. Can handle both categorical and numerical data: They are versatile for different problem types. 
4. Feature selection: They can automatically select important features.

       
**Implementation**
1. For better performance Date Sold was converted to year sold
2. Data was split in X_train, X_test, y_train, y_test using standard train test split with trainning set having 80% data
3. DecisionTreeRegressor model was run, fitted and predicted
4. Mean Squared Error, Mean Absolute Error, R2 Score were calculated
5. Depth of Tree, number of leaves were saved
6. Recall Score, precision Score and F1 score were saved

#### MAE and MSE Score Metrices
![MAE and MSE](https://github.com/viksaraw/House-Prices/blob/main/Pics/Modeling1%20DT1.png)

#### Precision Score, recall Score and other metrices
![Precision and Recall](https://github.com/viksaraw/House-Prices/blob/main/Pics/Modeling%202%20DT2.png)

**Conclusion from Decision Tree Regressor**

**1. Overfitting is Evident**
	Training Accuracy: 1.00 vs. Test Accuracy: 0.72
	The model fits the training data perfectly but generalizes poorly to unseen data, indicating overfitting. This is also supported by the very deep tree (depth = 34) 	and large number of leaves (16,705).<br><br>
**2.  Moderate Predictive Power**
	R² Score: 0.72
	The model explains about 72% of the variance in the test data, which is decent but leaves room for improvement. This aligns with the test accuracy.<br><br>
**3. High Prediction Error**
	MAE: $104,988, MSE: $42.2 billion
	These are large error values, suggesting that the model's predictions can be significantly off from actual prices. MAE gives a more interpretable average error per 	prediction <br><br>
**4.  Poor Classification Metrics**
	Recall: 0.0022, Precision: 0.0016, F1 Score: 0.0018
	These metrics are extremely low, indicating that if you're using classification metrics on a regression task (perhaps for a thresholded price category), the model 	is not effective in identifying the correct class <br><br>
**5. Model Complexity is Too High**
	Tree Depth: 34, Leaves: 16,705
	Such a complex tree is likely memorizing the training data rather than learning general patterns. Pruning or limiting depth could help reduce overfitting and 		improve generalization <br><br>

 --------------------------------------------------------------------------------------------------------------------------------------------------------------------

 #### 2. Random Forest
It uses an ensemble of decision trees to make predictions. It's an ensemble method, meaning it combines the predictions of multiple models (in this case, decision trees) to arrive at a more accurate and robust result

File : https://github.com/viksaraw/House-Prices/blob/main/Scripts/Random%20Forest.ipynb

**Advantages of Random Forest**

1. Reduces Overfitting :By averaging the results of many decision trees, Random Forest smooths out predictions and reduces the risk of overfitting that plagues single trees.
2. Improves Accuracy :Typically achieves higher accuracy than a single decision tree due to the ensemble approach, which captures more complex patterns in the data.
3. Handles Non-Linearity and Interactions Well:Can model non-linear relationships and feature interactions without needing explicit specification.
4. Robust to Noise and Outliers: Since it aggregates predictions from multiple trees, it’s less sensitive to noisy data and outliers.
5. Works Well with Missing Data: Can handle missing values better than many other models, especially if imputation is used.

**Implementation**

1. For better performance Date Sold was converted to year sold
2. Data was split in X_train, X_test, y_train, y_test using standard train test split with trainning set having 80% data
3. RandomForestRegressor model was run, fitted and predicted
4. Accuracy, Recall and Precision Score were calculated for Training and Test Data set

**Metrics with Random Forest**

![Random Forest](https://github.com/viksaraw/House-Prices/blob/main/Pics/Modeling%203%20RF.png)

**Conclusion from Random Forest Model**<br>
**1. Strong Generalization Performance**
	Training R²: 0.9835, Test R²: 0.8530
	The model generalizes well to unseen data, with only a modest drop from training to test accuracy—indicating low overfitting and high predictive power.<br>

**2. High Recall and Precision**
	Test Recall: 0.9258, Test Precision: 0.9086
	The model is both sensitive (captures most relevant instances) and precise (makes few false predictions), making it reliable for tasks where both false positives 	and false negatives matter.<br>

**3. Well-Balanced Model**
	The close alignment between training and test metrics across R², recall, and precision suggests a well-tuned and stable model<br>

 ---------------------------------------------------------------------------------------------------------------------------------------------------------------

  #### 3. XBoost Model

  File: https://github.com/viksaraw/House-Prices/blob/main/Scripts/XBoost.ipynb
  
  XGBoost is a highly efficient and accurate machine learning algorithm, especially well-suited for supervised learning tasks like classification and regression. It's a 
  powerful ensemble method that builds on top of Gradient Boosting, using decision trees as its base learners.

  **Advantages of XBoost Model**
  
  1. Efficiently handles missing values and outliers 
  2. Includes built-in regularization to prevent overfitting
  3. Scales well to large datasets 
  4. Offers flexibility in tuning and optimization

  **Implementation**

 1. For better performance Date Sold was converted to year sold
 2. Data was split in X_train, X_test, y_train, y_test using standard train test split with trainning set having 80% data
 3. RandomForestRegressor model was instanced with n_estimators=100,learning_rate=0.1,max_depth=6
 4. Random Forest Regressor model was fitted and predicted
 5. Root Mean Squared Error, Accuracy, Recall and Precision Scores were calculated for Training and Test Sets

**Feature Importance with XBoost Model**
![XBoost1](https://github.com/viksaraw/House-Prices/blob/main/Pics/Modeling%204-XB.png)<br>

**Metrics with XBoost Model**
![Xboost2](https://github.com/viksaraw/House-Prices/blob/main/Pics/Modeling%205%20XB%202.png)

**Conclusion from XBoost Model**

1. Excellent Predictive Accuracy
Test R²: 0.8796 shows that the model explains nearly 88% of the variance in house prices, outperforming both Decision Tree and Random Forest models<br>

2. Moderate Prediction Error
RMSE: $134,909 indicates the average prediction error is relatively low for a regression task involving house prices, especially considering the complexity of real estate data <br>

3. Balanced Generalization
Training R²: 0.9578 vs. Test R²: 0.8796 shows a small generalization gap, suggesting the model is well-regularized and not overfitting <br>

4. High Recall and Precision
Test Recall: 0.9244, Test Precision: 0.9098 — the model is both accurate and consistent in identifying relevant predictions, making it reliable for downstream decision-making<br>

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

  #### 3. Logistic Regression

  File : https://github.com/viksaraw/House-Prices/blob/main/Scripts/Logistic%20Regression.ipynb
  
  Logistic regression is a supervised machine learning algorithm in data science. It is a type of classification algorithm that predicts a discrete or categorical outcome. 

  **Advantages of Logistic Regression Model**

 1. **Interpretability**: Logistic regression models are relatively easy to interpret compared to more complex models
 2. **Ease of Implementation:** It's a relatively simple and widely used model, with many statistical packages and libraries supporting it
 3. **Quantifies Impact:** It can quantify the magnitude of each independent variable's impact on the binary outcome

**Implementation**

 1. For better performance Date Sold was converted to year sold
 2. Scaling was done using Standard Scaler
 3. Data was split in X_train, X_test, y_train, y_test using standard train test split with trainning set having 80% data
 4. Logistic Regression model was instanced,fitted and predicted
 5. Accuracy score was calculated
 6. Explained Variance Ratio by PCA component was calculated
 7. Classification Report was printed
 8. Confusion matrix was printed
 9. PCA Visualization was done with first 2 components

**Metrics with Logistic Regression Model**<br>
![Metrics](https://github.com/viksaraw/House-Prices/blob/main/Pics/Modeling%206%20Logistic%20Metrics.png)

**Confusion Matrix**<br>

Confusion matrix evaluates the performance of the classifier:

True Positives (TP): 1829
True Negatives (TN): 1851
False Positives (FP): 285
False Negatives (FN): 358

![Confusion Matrix](https://github.com/viksaraw/House-Prices/blob/main/Pics/Modeling%207%20Logistic%20Confusion.png)

**PCA Projection with first 2 components**<br><br>
The PCA transformation enables a two-dimensional view of the dataset after reducing its dimensions. Logistic regression is then applied to distinguish between price categories.

Red: Houses in the high-price segment
Blue: Houses in the low-price segment
Although some overlap is visible, the combination of PCA and logistic regression demonstrates a degree of linear class separation<br>

![PCA Projection](https://github.com/viksaraw/House-Prices/blob/main/Pics/Modeling%208%20Logistic%20Variance.png)

**Conclusion from Logistic Regression Model**<br>

**1. Solid Classification Performance**
The model achieves 85% accuracy, with balanced precision, recall, and F1-scores across both classes (0 and 1), indicating reliable and consistent classification<br><br>
**2. Dimensionality Reduction Insight**
The first few PCA components explain a significant portion of the variance (Component 1 alone explains 26.15%). This suggests that dimensionality reduction is effective, and the data has underlying structure that Logistic Regression can leverage<br><br>
**3. Well-Balanced Model**
The macro and weighted averages for precision, recall, and F1-score are all 0.85, showing that the model performs equally well across both classes, with no major bias toward one class<br><br>

-----------------------------------------------------------------------------------------------------------------------------------

### Linear Regression Model

File: https://github.com/viksaraw/House-Prices/blob/main/Scripts/Linear%20Regression.ipynb

Linear regression models the relationship between a dependent variable (target) and one or more independent variables (features) by fitting a straight line (in 2D) or a hyperplane (in higher dimensions) through the data

**Mathematical Form**

![Mathematical Form](https://github.com/viksaraw/House-Prices/blob/main/Pics/Linear%20Rg%20Math.png)

**Advantages of Linear Regression Model**

**1. Simple and Easy to Interpret**	
	Linear regression is straightforward to understand and implement.
	The model’s coefficients clearly show the impact of each feature on the target variable<br><br>
**2. Fast and Efficient**	
	It’s computationally inexpensive, making it ideal for large datasets and real-time applications<br><br>
**3. Good for Linearly Separable Data**
	Performs well when the relationship between features and the target is approximately linear<br><br>
**4. Useful for Inference**
	Helps in understanding relationships between variables, not just prediction.
	You can test hypotheses about the data using statistical metrics like p-values and confidence intervals<br><br>

**Implementation**

 1. For better performance Date Sold was converted to year sold
 2. Scaling was done using Standard Scaler
 3. Data was split in X_train, X_test, y_train, y_test using standard train test split with trainning set having 80% data
 4. Pipeline was created with standard scaler and linear regression
 5. Linear Regression model was instanced,fitted and predicted
 6. Cross Validation was done with params ranging from 2 to 11
 7. Best param was found using Cross Validation
 8. Metrics like Meas Square Error, Root Mean Squared Error were calculated
    
**Metrics with Linear Regression Model**<br><br>
![Metrics with Linear Regression](https://github.com/viksaraw/House-Prices/blob/main/Pics/Modeling%209%20Linear%20Regression.png)


**Conclusion from Linear Regression Model**<br>

1. Consistent Performance Across Train and Test Sets Train R²: 0.701, Test R²: 0.703 — the model generalizes well, with nearly identical performance on both sets, indicating low overfitting

2. Moderate Prediction Error Train RMSE: $197,757, Test RMSE: $212,041 — the model has a reasonable error margin, though the absolute values suggest room for improvement, especially for high-value predictions

3. Stable Cross-Validation Results 10-fold cross-validation confirms the model’s stability, with the best test MSE at ~$44.96 billion, reinforcing that the model performs consistently across different data splits

4. Simple Yet Effective Configuration The best parameter found was fit_intercept=True, showing that even a basic linear regression setup can yield solid results when the data has a linear trend

 ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

 ### Ridge Regression

 File: https://github.com/viksaraw/House-Prices/blob/main/Scripts/Ridge%20Regression.ipynb

 Ridge Regression is a type of regularized linear regression that adds a penalty to the size of the coefficients to prevent overfitting. It’s especially useful when your 
 data has multicollinearity (i.e., highly correlated features) or when the number of features is large compared to the number of observations<br>
 
 ![Ridge Regression ](https://github.com/viksaraw/House-Prices/blob/main/Pics/Ridge%20Mathematical%20Form.png)

**Key Benefits**
1. Reduces overfitting by shrinking coefficients
2. Handles multicollinearity better than standard linear regression
3. Improves generalization on unseen data
4. Keeps all features in the model (unlike Lasso, which can zero some out)

**Implementation**

1. For better performance Date Sold was converted to year sold
2. Scaling was done using Standard Scaler
3. Data was split in X_train, X_test, y_train, y_test using standard train test split with trainning set having 80% data
4. Pipeline was created with standard scaler and linear regression
5. Ridge Regression model was instanced,fitted and predicted
6. Cross Validation was done with params ranging from 2 to 11
7. Best param was found using Cross Validation
8. Metrics like Meas Square Error, Root Mean Squared Error were calculated

**Metrics with Ridge Regression Model**<br>

Best hyper parameter Value obtained by Cross Validation Technique

![Best CV value](https://github.com/viksaraw/House-Prices/blob/main/Pics/Modeling%2011%20Ridge%201.png)

Metrics from Ridge Regression

![Metrics from Ridge](https://github.com/viksaraw/House-Prices/blob/main/Pics/Modeling%2012%20Ridge%202.png)

**Conclusion from using Ridge Regression**<br><br>

1. **Consistent and Balanced Performance**
	Training R²: 0.701, Test R²: 0.703 — the model performs consistently across both sets, indicating good generalization and minimal overfitting.
2. **Moderate Prediction Error**
	Test RMSE: $212,053, Test MAE: $126,902 — while the model is stable, the error values suggest that predictions can still be off by a significant margin, especially 	for high-priced properties.
3. **Cross-Validation Stability**
	Best CV value: 2, with a Test MSE of ~$44.97 billion, shows that the model maintains reliable performance across folds, though a higher CV value might offer more 	robust validation.
4. **Effective Regularization**
	Ridge regression helps control coefficient magnitudes, improving model stability without sacrificing accuracy, especially in the presence of multicollinearity

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

 ### Lasso Regression

File: 

Lasso Regression (short for Least Absolute Shrinkage and Selection Operator) is a type of linear regression with L1 regularization. It not only helps prevent overfitting but also performs feature selection by shrinking some coefficients exactly to zero.

![LassoRegression](https://github.com/viksaraw/House-Prices/blob/main/Pics/Lasso%20Regression%20Mathematical.png)

**Key Advantages**

1. Reduces overfitting by penalizing large coefficients.
2. Performs automatic feature selection by setting some coefficients to zero.
3. Useful when you suspect only a subset of features are truly important.
4. Helps in building simpler, more interpretable models.

**Implementation**

1. For better performance Date Sold was converted to year sold
2. Scaling was done using Standard Scaler
3. Data was split in X_train, X_test, y_train, y_test using standard train test split with trainning set having 80% data
4. Pipeline was created with standard scaler and linear regression
5. Lasso Regression model was instanced,fitted and predicted
6. Metrics like Meas Square Error, Root Mean Squared Error were calculated

**Metrics with Lasso Regression Model**<br>

![ModelingLasso](https://github.com/viksaraw/House-Prices/blob/main/Pics/Modeling%2010-%20Lasso.png)

**Conclusion**

1. Consistent Model Performance
	Training R²: 0.701, Test R²: 0.703 — the model performs nearly identically on both sets, indicating strong generalization and minimal overfitting.
2. Moderate Prediction Error
	Test RMSE: $212,052, Test MAE: $126,906 — the model maintains a reasonable error margin, similar to Ridge and Linear Regression, suggesting stable predictive 		capability.
3. Potential Feature Selection
	Since Lasso applies L1 regularization, it likely reduced the influence of less important features, possibly setting some coefficients to zero, which helps in 		simplifying the model.
4. Comparable to Ridge and Linear Regression
	The performance metrics are very close to Ridge and standard Linear Regression, indicating that Lasso is equally effective for this dataset, with the added benefit 	of automatic feature selection.
---------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Conclusion

1. I used this excercise to create one working predictive model to predict the housing price of the given data set
2. I did basic Machine learning excercises following CRISP and run few models
3. I compared the performance of the models to find the best one
4. While doing this excercise I found some insights which I am sharing at the end

#### Metrics of Model Comparison
![Model Comparison](https://github.com/viksaraw/House-Prices/blob/main/Pics/Comparision%201.png)

**Clearly Best Model: XGBoost**
	Highest R²: 0.88752 → explains the most variance in the target variable
	Lowest RMSE: 122,888.77 → lowest average prediction error
	Lowest Std Dev: Indicates stable performance across folds
    
**Runner-Up: Random Forest**
	Very close in performance to XGBoost, but slightly lower R² and higher RMSE
    
**Summary:**
	XGBoost is the best overall model in this comparison
	It's both accurate and consistent, making it a strong choice for predicting house prices

 #### Model Comparison Graph
 ![Model Comparison Graph](https://github.com/viksaraw/House-Prices/blob/main/Pics/Comparision%203.png)

 ### Insights

 

 



   

