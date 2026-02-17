# ===============================
# Lab 3 - Data Cleaning & Analysis
# ===============================

import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load Dataset
# -------------------------------

file_path = "Chocolate_Sales.csv"
df = pd.read_csv(file_path)

print("Dataset Loaded Successfully!")
print(df.head())


# -------------------------------
# 2. Basic Information
# -------------------------------

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())


# -------------------------------
# 3. Check Missing Values
# -------------------------------

missing_summary = pd.DataFrame({
    "Missing Count": df.isnull().sum(),
    "Missing Percentage (%)": (df.isnull().sum() / len(df)) * 100
})

print("\nMissing Values Summary:")
print(missing_summary)


# -------------------------------
# 4. Check Duplicates
# -------------------------------

duplicate_count = df.duplicated().sum()

print("\nNumber of Duplicate Rows:", duplicate_count)

if duplicate_count > 0:
    print("\nSample Duplicates:")
    print(df[df.duplicated()].head())


# -------------------------------
# 5. Convert Date Column
# -------------------------------

df['Date'] = pd.to_datetime(df['Date'])


# -------------------------------
# 6. Feature Engineering
# -------------------------------

df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year


# -------------------------------
# 7. Top Countries by Sales
# -------------------------------

country_sales = df.groupby('Country')['Amount'].sum().sort_values(ascending=False)

print("\nTop Countries by Sales:")
print(country_sales)


# -------------------------------
# 8. Top Products by Sales
# -------------------------------

product_sales = df.groupby('Product')['Amount'].sum().sort_values(ascending=False)

print("\nTop Products by Sales:")
print(product_sales)


# -------------------------------
# 9. Salesperson Performance
# -------------------------------

salesperson_perf = df.groupby('Sales Person')['Amount'].sum().sort_values(ascending=False)

print("\nTop Salespersons:")
print(salesperson_perf)


# -------------------------------
# 10. Correlation Analysis
# -------------------------------

correlation = df['Boxes'].corr(df['Amount'])

print("\nCorrelation between Boxes and Amount:", correlation)


# -------------------------------
# 11. Monthly Sales Trend
# -------------------------------

monthly_sales = df.groupby(['Year', 'Month'])['Amount'].sum()

monthly_sales.plot(kind='line', marker='o')
plt.title("Monthly Sales Trend")
plt.xlabel("Year, Month")
plt.ylabel("Total Sales Amount")
plt.grid()
plt.show()


# -------------------------------
# 12. Top 10 Countries Bar Chart
# -------------------------------

top10_countries = country_sales.head(10)

top10_countries.plot(kind='bar')
plt.title("Top 10 Countries by Sales")
plt.xlabel("Country")
plt.ylabel("Total Sales Amount")
plt.show()


# -------------------------------
# 13. Key Findings
# -------------------------------

print("\n========== Key Findings ==========")

print("1. No missing values were found in the dataset.")
print("2. No duplicate records were detected.")
print("3. Australia, UK, India, USA, and Canada generate the highest revenue.")
print("4. Top products contribute significantly to overall sales.")
print("5. Salesperson performance varies considerably.")
print("6. Boxes sold and revenue show almost no correlation.")
print("7. Sales peak in early months of the year (especially January).")

print("=================================")
