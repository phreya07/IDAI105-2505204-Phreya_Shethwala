# IDAI105-2505204-Phreya_Shethwala
This project analyzes Black Friday sales data to understand customer purchasing patterns and behavior. It uses data mining techniques and an interactive Streamlit dashboard to identify customer segments, product associations, and key business insights.

# Project Overview

This project analyzes Black Friday retail sales data to understand customer purchasing behavior and product demand patterns. Using data mining techniques such as exploratory data analysis (EDA), customer clustering, association rule mining, and anomaly detection, the project uncovers meaningful insights from customer transactions. The findings are presented through an interactive Streamlit dashboard that allows users to explore trends, customer segments, product relationships, and unusual spending patterns.

The objective is to demonstrate how data analytics can support business decision-making by identifying customer segments, improving marketing strategies, and discovering opportunities for cross-selling and targeted promotions.

# Purpose of the Project

Retail companies experience extremely high sales volumes during large promotional events such as Black Friday. However, understanding customer preferences and spending behavior during these events can be challenging without proper data analysis.

This project aims to apply data mining techniques to analyze sales data and generate insights that can help retailers:

Understand customer purchase behavior

Identify high-value customer segments

Discover frequently purchased product combinations

Detect unusual spending patterns

Support data-driven marketing strategies

# Project Goals

The primary goals of the project are:

Analyze Black Friday retail data to understand purchasing patterns.

Perform data preprocessing and cleaning to prepare the dataset for analysis.

Use exploratory data analysis to identify trends and relationships within the dataset.

Apply clustering techniques to segment customers based on their purchasing behavior.

Use association rule mining to discover relationships between product categories.

Detect anomalies such as unusually high spending customers.

Present insights through an interactive Streamlit dashboard.

# Problem Statement

Retailers often collect large volumes of transaction data during major sales events, but this data is rarely analyzed effectively. Without proper analysis, businesses may miss valuable insights about customer behavior, product demand, and purchasing patterns.

The main problem addressed in this project is how to transform raw retail transaction data into meaningful insights that can support strategic business decisions. By applying data mining techniques, the project aims to reveal patterns that help retailers better understand their customers and optimize their sales strategies.

# User Motivation

Retail companies need tools that help them understand customer behavior quickly and effectively. Traditional reports often fail to highlight hidden patterns within large datasets.

This project was developed to provide an interactive dashboard that enables users to:

Explore customer purchase patterns visually

Identify different customer segments

Understand relationships between product categories

Detect unusual purchasing behavior

Support marketing and promotional decision-making

The motivation behind the project is to demonstrate how data analytics and visualization can transform raw data into actionable business intelligence.

# Data Preprocessing

Before performing the analysis, the dataset was cleaned and prepared to ensure accuracy and reliability.

Preprocessing Steps:

Handling missing values in Product_Category_2 and Product_Category_3

Encoding categorical variables such as Gender and Age

Converting categorical attributes into numerical values

Removing duplicate entries

Normalizing numerical variables for clustering analysis

These preprocessing steps ensure that the dataset is consistent and suitable for machine learning techniques.

# Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to identify patterns and relationships within the dataset.

Key Visualizations

Purchase distribution histogram

Purchase comparison by gender

Product category popularity analysis

Average purchase by age group

Correlation heatmap of key features

These visualizations help reveal important trends such as which customer groups spend the most and which product categories generate the highest revenue.

# Advanced Analysis Techniques
# Customer Segmentation (Clustering)

Customer segmentation was performed using the K-Means clustering algorithm.

The clustering model groups customers based on features such as:

Age

Occupation

Marital status

Purchase amount

The elbow method was used to determine the optimal number of clusters. Each cluster represents a group of customers with similar purchasing behaviors.

Example customer segments:

Budget shoppers

Frequent buyers

Premium spenders

# Association Rule Mining

Association rule mining was used to discover relationships between product categories.

The Apriori algorithm was applied to identify frequent product combinations.

The analysis calculates:

Support

Confidence

Lift

These metrics help determine how strongly two product categories are related. Retailers can use this information to create product bundles or cross-selling strategies.

# Anomaly Detection

Anomaly detection techniques were used to identify unusual spending behavior in the dataset.

Statistical methods such as Interquartile Range (IQR) and Z-score were used to detect outliers.

These anomalies may represent:

Extremely high spending customers

Bulk purchase transactions

Unusual purchasing behavior

Identifying such patterns can help retailers recognize high-value customers or investigate abnormal transactions.

# Project Limitations

Although the project provides useful insights, several limitations should be considered:

The dataset only represents a single sales event.

Customer income and personal preferences are not included.

External factors such as promotions or advertising are not analyzed.

Product categories are generalized and may not reflect detailed product information.

Future improvements could include additional datasets and more advanced machine learning models.

# Insights & Reporting

After completing the data analysis, the key findings were summarized to highlight important patterns in customer purchasing behavior during the Black Friday sale.

Age Group Spending Patterns
The analysis shows that the 26–35 age group contributes the highest purchase amounts, indicating that young working professionals are the most active shoppers during the sale event. This group tends to have higher purchasing power and shows strong engagement with promotional offers.

Product Preferences by Gender
Product category analysis reveals differences in purchasing preferences between male and female customers. Male customers tend to purchase more items from electronics and technology-related categories, while female customers show stronger interest in fashion, lifestyle, and accessory categories. These insights can help retailers design targeted marketing campaigns for different customer groups.

High-Spending Customer Segments
Clustering and anomaly detection techniques identified a group of premium buyers who spend significantly more than the average customer. These high-value customers represent an important segment for retailers and could be targeted with personalized promotions, loyalty programs, and exclusive offers to encourage repeat purchases.

Overall, the analysis provides valuable insights into customer behavior, helping retailers better understand their audience and optimize sales strategies during large promotional events.

# Conclusion

This project demonstrates how data mining techniques can be applied to retail sales data to extract meaningful insights. By combining data preprocessing, exploratory analysis, clustering, association rule mining, and anomaly detection, the project reveals patterns that help retailers understand customer behavior and improve business strategies. The interactive Streamlit dashboard makes these insights accessible and easy to explore.

Streamlit Link: 
