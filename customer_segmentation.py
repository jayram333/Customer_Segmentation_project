import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = {
    'CustomerID':[1,2,3,4,5,6,7,8,9,10],
    'AnnualIncome':[15,16,17,40,42,43,80,82,85,90],
    'SpendingScore':[39,81,6,77,40,76,6,94,3,72]
}

df = pd.DataFrame(data)

X = df[['AnnualIncome','SpendingScore']]

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X)

print(df)

plt.figure(figsize=(6,4))
plt.scatter(df['AnnualIncome'], df['SpendingScore'], c=df['Cluster'])
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Customer Segmentation')
plt.show()
