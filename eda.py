import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load processed dataset
df = pd.read_csv('amazondelivery_processed.csv')

# Delivery time distribution plot
plt.figure(figsize=(8, 5))
sns.histplot(df['Delivery_Time'], bins=30, kde=True)
plt.title('Delivery Time Distribution')
plt.xlabel('Delivery Time (hours)')
plt.savefig('visuals/delivery_time_dist.png')
plt.show()

# Scatter plot: Distance vs Delivery Time
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Distance', y='Delivery_Time', data=df)
plt.title('Distance vs Delivery Time')
plt.xlabel('Distance (km)')
plt.ylabel('Delivery Time (hours)')
plt.savefig('visuals/distance_vs_time.png')
plt.show()

# Correlation heatmap
# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=['number'])

plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.savefig('visuals/correlation_heatmap.png')
plt.close()
