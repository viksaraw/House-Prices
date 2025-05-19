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
| 15 |price             | float     | Target variable: house price                      |
| 16 | yr_renovated     | integer   | Year renovated (0 if never renovated)             |
| 17 | zipcode          | integer   | Zip code area                                     |
| 18 | lat              | float     | Latitude                                          |
| 19 | long             | float     | Longitude                                         |
| 20 | sqft_living15    | integer   | Avg living area of nearest 15 neighbors           |
| 21 | sqft_lot15       | integer   | Avg lot size of nearest 15 neighbors              |
 

### Data Snapshot
Belo is the snapshot of sample data

[!Data Snapshot](https://onedrive.live.com/?cid=F021F9F9BDA04EE7&sb=name&sd=1&id=F021F9F9BDA04EE7%21scf61ef90080043aab04f05ae487d7069&parId=F021F9F9BDA04EE7%21s1841d9d4f6d443ffbfd91056fd06c4f7&o=OneUp)



