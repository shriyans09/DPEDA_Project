import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset (modify the path as needed)
df = pd.read_csv("updated_dataset.csv")

# Page Title
st.title("E-commerce Customer Behavior Analysis")

# Overview Section
st.header("Project Overview")
st.write("""
This project analyzes customer behavior data from an e-commerce platform. 
Through this dashboard, you can explore various insights on demographics, 
purchase patterns, product preferences, and customer sentiment.
""")

df = df.drop(columns=['Customer Age'], errors='ignore')  # to avoid errors if column doesn't exist
#change column names for preprocessing
df.columns = [
    'CustID',
    'PurchaseDate',
    'Category',
    'ProdPrice',
    'Quantity',
    'TotalAmt',
    'PaymentMethod',
    'Returns',
    'CustName',
    'Age',
    'Gender',
    'Churn',
    'Reviews'
]
df.head()

# Sidebar Home Button
st.sidebar.title("Navigation")
if st.sidebar.button("Home"):
    st.rerun()  # Refreshes the page, effectively scrolling to the top

# Display Sample Data
st.header("Sample Data")
st.write("Here is a sample of the dataset used for analysis:")
st.dataframe(df.head(5))  # Display the first 10 rows as a sample

# Button to Show Table Statistics
if st.button("Describe Table Statistics"):
    st.header("Dataset Statistics")
    st.write("Summary statistics for each column in the dataset:")
    st.dataframe(df.describe(include='all'))


# Sidebar Filters
# st.sidebar.header("Filters")
# age_group = st.sidebar.selectbox("Select Age", df['Age'].dropna().unique())
# category = st.sidebar.selectbox("Select Product Category", df['Category'].dropna().unique())
# gender = st.sidebar.selectbox("Select Gender", df['Gender'].dropna().unique())


# Apply Filters to Data
# filtered_df = df[(df['Age'] == age_group) &
#                  (df['Category'] == category) &
#                  (df['Gender'] == gender)]

# *****************************

#change no review to NaN in review column
df['Reviews'] = df['Reviews'].replace('No Review', np.nan)
df = df.dropna()
#total null values in sorted order
df.isnull().sum().sort_values(ascending=False)

#seed the random numbers

np.random.seed(0)

df['Reviews'] = df['Reviews'].replace({
    'Worst': np.random.uniform(0, 1.5, size=len(df[df['Reviews'] == 'Worst'])),
    'Bad': np.random.uniform(1.5, 3, size=len(df[df['Reviews'] == 'Bad'])),
    'Good': np.random.uniform(3, 4, size=len(df[df['Reviews'] == 'Good'])),
    'Excellent': np.random.uniform(4, 5, size=len(df[df['Reviews'] == 'Excellent']))
})

df['Reviews'] = df['Reviews'].round(1)

# ******************************

# Demographic Analysis
st.header("Demographic Analysis")

# Age Distribution
st.subheader("Age Distribution")
st.write("""
This analysis provides insights into the age distribution of our customer base. 
By understanding the age groups that are most active on our platform, 
we can better tailor our marketing and product recommendations.
""")
fig, ax = plt.subplots()
sns.histplot(df['Age'], bins=20, kde=True, ax=ax)
st.pyplot(fig)

# Product Category Analysis
st.header("Product Category Preferences")
st.write("""
This analysis explores customer preferences across various product categories. 
By identifying the most popular categories and analyzing them by gender, 
we gain insights into which product types resonate best with different demographics. 
These insights can inform our inventory and promotional strategies.
""")
fig, ax = plt.subplots()
sns.countplot(x='Category', data=df, hue='Gender', ax=ax)
ax.set_title("Product Category by Gender")
ax.set_xlabel("Product Category")
ax.set_ylabel("Number of Purchases")
st.pyplot(fig)


# Purchase Patterns
st.header("Purchase Patterns")

# Payment Method Distribution
st.subheader("Payment Method Distribution")
st.write("""
This analysis examines the distribution of payment methods used by customers. 
Understanding preferred payment options can help improve the checkout experience 
and highlight any potential needs for expanding or enhancing payment options.
""")
fig, ax = plt.subplots()
sns.countplot(x='PaymentMethod', data=df, ax=ax)
ax.set_title("Payment Method Distribution")
ax.set_xlabel("Payment Method")
ax.set_ylabel("Number of Purchases")
st.pyplot(fig)


# Correlation Analysis
st.header("Correlation Analysis")
st.write("""
This heatmap visualizes the correlation among key numeric variables in our dataset, including product price, 
quantity, total purchase amount, and customer age. By examining the strength and direction of these relationships, 
we can identify potential patterns or dependencies, which may inform pricing strategies, product recommendations, 
and targeted marketing efforts.
""")
fig, ax = plt.subplots(figsize=(10, 6))
corr = df[['ProdPrice', 'Quantity', 'TotalAmt', 'Age']].corr()
sns.heatmap(corr, annot=True, cmap='viridis', linewidths=0.5, ax=ax)
ax.set_title("Correlation Heatmap of Product Price, Quantity, Total Purchase Amount, and Age")
st.pyplot(fig)

# Churn Rate by Age Group
st.subheader("Churn Rate by Age Group")
st.write("""
This analysis focuses on the churn rate across different age groups, offering insights into which age segments have higher or lower customer retention. 
By identifying age groups with higher churn rates, we can explore targeted retention strategies to improve customer loyalty and reduce churn.
""")
fig, ax = plt.subplots()

df['Age Group'] = pd.cut(df['Age'], bins=[18, 30, 40, 50, 60, 70, 80], labels=['18-30', '31-40', '41-50', '51-60', '61-70', '71-80'])
plt.figure(figsize=(10, 6))

sns.countplot(x='Age Group', hue='Churn', data=df, ax=ax)
ax.set_title("Churn Rate by Age")
ax.set_xlabel("Age Group")
ax.set_ylabel("Number of Customers")
st.pyplot(fig)

# Add a header for the section
st.header("KDE Plot of Total Purchase Amount by Gender")

# Description of the plot
st.write("""
This plot shows the Kernel Density Estimate (KDE) for the total purchase amount 
by gender. The KDE is used to estimate the probability density function of a 
continuous random variable. It provides a smooth curve that shows how the 
purchase amounts are distributed for male and female customers.
""")

# Set the size of the plot
plt.figure(figsize=(10, 6))

# Plot KDE for Male and Female Total Purchase Amount
sns.kdeplot(df[df['Gender'] == 'Male']['TotalAmt'], label='Male', fill=True)
sns.kdeplot(df[df['Gender'] == 'Female']['TotalAmt'], label='Female', fill=True)

# Add title and labels
plt.title('KDE Plot of Total Purchase Amount by Gender')
plt.xlabel('Total Purchase Amount')
plt.ylabel('Density')

# Display the legend
plt.legend()

# Display the plot in Streamlit
st.pyplot(plt)



# Sentiment Analysis
st.header("Customer Sentiment Analysis")

# Review Sentiment Distribution
st.subheader("Review Sentiment Distribution")
st.write("""
This analysis explores the distribution of customer sentiments based on their reviews. 
By categorizing reviews into different sentiment levels, we can gauge overall customer satisfaction 
and identify areas for improvement. Understanding sentiment trends helps in enhancing 
customer experience and tailoring services to meet customer expectations.
""")
fig, ax = plt.subplots()
bins = [0,1.5,3,4,5]
labels = ['Worst','Bad','Good','Excellent']
df['ReviewsCat'] = pd.cut(df['Reviews'], bins=bins, labels=labels)
plt.figure(figsize=(10, 6))
sns.countplot(x='ReviewsCat', data=df, ax=ax)
ax.set_title("Customer Review Sentiment")
ax.set_xlabel("Review Sentiment")
ax.set_ylabel("Count")
st.pyplot(fig)
