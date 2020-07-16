from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from itertools import cycle, islice
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
df = pd.read_csv('C:/Users/15877/PycharmProjects/ِWorld indicators/minute_weather.csv', sep=',')
sampled_df = df[(df['rowID'] % 10) == 0]

print(sampled_df[sampled_df['rain_accumulation'] == 0].shape)
print(sampled_df[sampled_df['rain_duration'] == 0].shape)
# Drop all the Rows with Empty rain_duration and rain_accumulation
del sampled_df['rain_accumulation']
del sampled_df['rain_duration']

sampled_df = sampled_df.dropna()

features = ['air_pressure', 'air_temp', 'avg_wind_direction', 'avg_wind_speed', 'max_wind_direction',
            'max_wind_speed', 'relative_humidity']
select_df = sampled_df[features]
print(select_df.shape)
X = StandardScaler().fit_transform(select_df)
print(X.shape)

kmeans = KMeans(n_clusters=12)
model = kmeans.fit(X)
# the centers of 12 clusters
centers = model.cluster_centers_

# creating a function that creates a DataFrame and a column for the Cluster Number
def pd_centers(featuresUsed, centers):
    colNames = list(featuresUsed)
    colNames.append('prediction')

    # Zip with a column called 'prediction' (index)
    Z = [np.append(A, index) for index, A in enumerate(centers)]

    # Convert to pandas data frame for plotting
    P = pd.DataFrame(Z, columns=colNames)
    P['prediction'] = P['prediction'].astype(int)
    return P

P = pd_centers(features, centers)
print(P)

def parallel_plot(data):
    my_colors = list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None, len(data)))
    plt.figure(figsize=(15, 10)).gca().axes.set_ylim([-3, +3])
    parallel_coordinates(data, 'prediction', color=my_colors, marker='o')
    plt.show()

# Dry Days

parallel_plot(P[P['relative_humidity'] < -0.1])
print(P[P['relative_humidity'] < -0.1])
