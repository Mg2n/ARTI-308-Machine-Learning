import pandas as pd

# =========================
# 1. Load Dataset
# =========================
df = pd.read_csv("Chocolate_Sales.csv")

print("First 5 rows:")
print(df.head())

# =========================
# 2. Data Quality Assessment
# =========================
print("\nDataset Info BEFORE cleaning:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# =========================
# 3. Data Cleaning
# =========================

# Convert Date to datetime
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

# Clean Amount column
df["Amount"] = df["Amount"].replace('[\$,]', '', regex=True).astype(float)

# Drop duplicates (if any)
df = df.drop_duplicates()

# =========================
# 4. After Cleaning
# =========================
print("\nDataset Info AFTER cleaning:")
print(df.info())

# =========================
# 5. Basic Analysis
# =========================

print("\nTotal Sales:")
print(df["Amount"].sum())

print("\nSales by Country:")
print(df.groupby("Country")["Amount"].sum())

print("\nSales by Product:")
print(df.groupby("Product")["Amount"].sum())