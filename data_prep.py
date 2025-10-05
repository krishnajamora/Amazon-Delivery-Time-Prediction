import pandas as pd
from sklearn.preprocessing import LabelEncoder
from geopy.distance import geodesic

df = pd.read_csv('amazon_delivery.csv')
print("Original Columns:", df.columns.tolist())
df.columns = df.columns.str.strip()
print("Stripped Columns:", df.columns.tolist())

df.fillna(df.median(numeric_only=True), inplace=True)
df.drop_duplicates(inplace=True)

categorical_cols = ['Weather', 'Traffic', 'Vehicle', 'Area', 'Category']
for col in categorical_cols:
    if col in df.columns:
        df[col] = LabelEncoder().fit_transform(df[col])
    else:
        print(f"Warning: Column '{col}' not found in dataset")

# Drop rows missing date or time
df = df.dropna(subset=['Order_Date', 'Order_Time'])
df['OrderDateOrderTime'] = pd.to_datetime(df['Order_Date'] + ' ' + df['Order_Time'], errors='coerce')
df = df.dropna(subset=['OrderDateOrderTime'])

df['PickupTime'] = pd.to_datetime(df['Pickup_Time'], errors='coerce')
df = df.dropna(subset=['PickupTime'])

def compute_distance(row):
    store = (row['Store_Latitude'], row['Store_Longitude'])
    drop = (row['Drop_Latitude'], row['Drop_Longitude'])
    return geodesic(store, drop).km

df['Distance'] = df.apply(compute_distance, axis=1)

df['OrderHour'] = df['OrderDateOrderTime'].dt.hour
df['OrderDayOfWeek'] = df['OrderDateOrderTime'].dt.dayofweek

df.to_csv('amazondelivery_processed.csv', index=False)

print("Data prep done, saved as amazondelivery_processed.csv")
