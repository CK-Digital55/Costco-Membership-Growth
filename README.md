# Costco Membership Growth Prediction Model



## Description

**This model is an analysis and recommendation report focused on Costco Wholesale Incorporated's store #1201. The aim is to use advanced data analytics and machine 
 learning methodologies to align with Costco's vision for 
 expansion and to enhance customer service from 2023 to 2033. The report provides historical context and discusses the store's membership trajectory since its launch in 
 2016.**

## Languages and Utilities used

- Python
- Jupyter Notebook

## Environments Used

- Windows 11
- Server: 2023 version

## 1. Project walk-through

 *Uploaded several datasets and a visualization image:*

- Costco Global Membership revenue dataset.csv
- statistic_id952763_costco_-paid-membership-worldwide-2014-2022.xlsx
- statistic_id718406_costco_-memberships-worldwide-2014-2022.xlsx
- Costco-2022-Annual-Report.csv
- lafayette_parish_population_visualizations.png
- lafayette_parish_population_2023.csv

**Let's start by examining each dataset to understand what information is available. Then we can decide on the best way to proceed with building a predictive model.**


## 2. Initialize the list to hold the dataset names and previews


## 3. Read the first few rows of each dataset to understand its structure


## 4. Here's a brief overview:

- Costco Global Membership revenue dataset.csv: This dataset seems to contain textual descriptions related to Costco, such as the number of memberships, global revenue, and average sales per warehouse. It doesn't appear 
  to contain numerical data for analysis.

- statistic_id952763_costco_-paid-membership-worldwide-2014-2022.xlsx: The first few rows of this Excel file don't seem to contain actual data; they may be metadata or introductory content.

- statistic_id718406_costco_-memberships-worldwide-2014-2022.xlsx: Similar to the previous Excel file, the first few rows don't seem to contain the actual data.

- Costco-2022-Annual-Report.csv: This file also seems to contain textual data, possibly an introduction or metadata related to Costco's annual report for 2022.

- lafayette_parish_population_2023.csv: This dataset contains population data for areas in Lafayette Parish for the year 2023.

## 5. Before proceeding, I'll check if the Excel files contain more sheets with data. I'll also skip the initial rows in the Excel files to see if they contain any relevant numerical data for analysis.

**Input Function to read Excel sheets and skip initial rows to find data**


## 6. Read Excel files and check for additional sheets containing data

 **We have found some numerical data in the Excel files:**

- statistic_id952763_costco_-paid-membership-worldwide-2014-2022.xlsx: This file contains the number of paid Costco members worldwide from 2014 to 2019. The data starts from 42 million in 2014 and goes up to 53.9 million 
  in 2019.

- statistic_id718406_costco_-memberships-worldwide-2014-2022.xlsx: This file contains the number of Costco card holders worldwide from 2014 to 2019. The data starts from 76.4 million in 2014 and goes up to 98.5 million 
  in 2019.

- lafayette_parish_population_2023.csv: This dataset contains population data for areas in Lafayette Parish for the year 2023.

 ## *We used the Membership Growth Prediction: We will predict the future growth of Costco memberships globally*

## 7. Predicting membership growth can provide valuable insights for business strategy and planning. 
 **To build a predictive model for global membership growth, I'll consider the following steps:**

- Data Preparation: Clean and preprocess the data for analysis.
- Exploratory Data Analysis (EDA): Analyze the data to identify trends and patterns.
- Model Building: Use machine learning techniques to build the predictive model.
- Evaluation: Evaluate the model's performance.
- Prediction: Use the model to make future predictions.
  
## 8. Let's start with the Data Preparation step. I'll combine the data from the two relevant Excel files— one containing the number of paid Costco members worldwide and the other containing the total number of Costco 
   card holders worldwide. This will give us a more comprehensive view of membership growth


*It appears that the columns are not properly labeled, likely due to the way the Excel sheets were read. I'll clean up these datasets by appropriately renaming the columns and dropping any unnecessary ones. Then, we can 
 proceed with the data merging and analysis.*

## 9. Now we have a cleaned and merged dataset that contains the following information:

- Year: The year of the data
- Paid_Memberships_Millions: The number of paid Costco memberships worldwide, in millions
- Total_Memberships_Millions: The total number of Costco card holders worldwide, in millions

## 10. With this dataset in hand, we can proceed to the Exploratory Data Analysis (EDA) step. Here, we'll visualize the data to identify trends and patterns that can inform our predictive model. 


## 11. Set the style for the visualizations

## 12. Create subplots

## 13. Plot for Paid Memberships


## 14. Plot for Total Memberships



## 15. *The line plots above showcase the growth trends for both paid and total memberships at Costco from 2015 to 2022:*

- Growth of Paid Memberships Worldwide: There is a steady upward trend in the number of paid memberships. The growth seems to be fairly consistent over the years.

- Growth of Total Memberships Worldwide: Similarly, the total memberships also show a steady upward trend. The growth pattern is quite consistent, similar to that of paid memberships.

- These trends suggest that a linear or polynomial model might be a good fit for making future predictions.

## 16. Next, we'll move on to Model Building. I'll start by preparing the data for training and testing, and then proceed to fit a model to it.



## 17. I started with a simple linear regression model for its interpretability and ease of use. If the linear model performs reasonably well, it may serve our purpose. If not, we can explore more complex models like 
   polynomial regression or even time series models.
 
- Data Preparation: Separate the features (Year) and labels (Paid_Memberships_Millions and Total_Memberships_Millions).
- Train the Model: Fit a linear regression model to the data.
- Evaluation: Evaluate the model's performance using metrics like R-squared and Mean Squared Error (MSE).


## 18. Let's start with Data Preparation.


## 19.  Prepare the features and labels for Paid Memberships

## 20. Train a linear regression model for Paid Memberships


## 21. Train a linear regression model for Total Memberships


## 22. Make predictions


## 23. Calculate evaluation metrics


## 24. The evaluation metrics for the linear regression models are as follows:

**For Paid Memberships:**
Mean Squared Error (MSE): 
                0.9372
                0.9372
R-squared (R2): 0.9798 
                0.9798

**For Total Memberships:**
Mean Squared Error (MSE): 
                2.1632
                2.1632
R-squared (R2): 0.9850
                0.9850

Both models have high R2 values, which indicates a good fit to the observed data. The MSE values are also relatively low, suggesting that the model makes predictions that are close to the actual values.



## 25. Create subplots

## 26. Plot for Paid Memberships with Predictions


## 27. Plot for Total Memberships with Predictions



## 28. The plots above compare the actual data points (in blue) with the predictions made by the linear regression models (in red) for both paid and total memberships.

- Paid Memberships: Actual vs Predicted: The red line closely follows the actual data points, indicating a good fit for the model.

- Total Memberships: Actual vs Predicted: Similarly, the model for total memberships also shows a good fit, with the red line closely following the actual data points.

*These visualizations confirm that the linear models are reasonably accurate for predicting future membership growth.*


## 29. Now that we have validated models, we can use them to make future predictions for both paid and total memberships at Costco for the next few years.

**Let's forecast the membership numbers for the years 2023 to 2033.**

**I'll use the trained models to predict the memberships for these years.**


## 30. Create an array for the future years

## 31. Use the trained models to make predictions for future years

## 32. Create a DataFrame to store the future predictions



## 33. According to these predictions:

- The number of Paid Memberships is expected to grow from approximately 67.34 million in 2023 to around 96.78 million by 2033.

- The Total Memberships are expected to increase from approximately 121.83 million in 2023 to around 173.91 million by 2033.


 
## 34. Now we can visualize these predictions

## 35. Create subplots

## 36. Plot for Predicted Paid Memberships


## 37. Plot for Predicted Total Memberships


# 38. Here are the visualizations for the predicted growth of Costco memberships worldwide from 2023 to 2033:

- **Predicted Growth of Paid Memberships Worldwide (2023-2033): The green line indicates a steady growth in paid memberships, increasing from about 67 million in 2023 to approximately 97 million by 2033.**

- **Predicted Growth of Total Memberships Worldwide (2023-2033): The purple line also suggests a steady growth in total memberships, increasing from about 122 million in 2023 to around 174 million by 2033.**

**These predictions can be valuable for Costco's long-term planning and strategic decision-making.**

