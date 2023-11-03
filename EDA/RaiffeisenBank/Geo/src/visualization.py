import pandas as pd
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
import numpy as np

def plot_scatter_persons(*args):
    """
    Plot scatter points for multiple persons.
    """
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    markers = ['x', 'o', '^', 's', 'd']
    for index, df in enumerate(args):
        lat, lon = df['latitude'].values, df['longitude'].values
        plt.scatter(lon, lat, c=colors[index % len(colors)], label=f'Person {index + 1}', alpha=0.5, s=50, marker=markers[index % len(markers)])
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Locations Visited by Persons')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

def create_heatmap(*args):
    """
    Create a heatmap for multiple persons.
    """
    m = folium.Map(location=[np.mean([df['latitude'].mean() for df in args]), np.mean([df['longitude'].mean() for df in args])], zoom_start=13)
    all_locations = [loc for df in args for loc in df[['latitude', 'longitude']].values.tolist()]
    HeatMap(all_locations).add_to(m)
    return m

def plot_daywise_activity(df, title):
    """
    Plot the activity distribution by day of the week.
    """
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daywise_counts = df['day_of_week'].value_counts().sort_index()
    daywise_counts.index = days

    plt.figure(figsize=(10, 6))
    daywise_counts.plot(kind='bar', color='skyblue')
    plt.title(title)
    plt.ylabel('Number of Activities')
    plt.xlabel('Day of the Week')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(axis='y')
    plt.show()

def extract_temporal_features(df):
    """
    Extract temporal features from a dataframe.
    """
    timestamp_col = 'start_time(YYYYMMddHHmmZ)'
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
    df['day_of_week'] = df[timestamp_col].dt.dayofweek
    df['hour_of_day'] = df[timestamp_col].dt.hour
    return df

def plot_daywise_activity_with_mean(df, person_name):
    """
    Plot the activity distribution with mean for a person.
    """
    df = extract_temporal_features(df)
    plot_daywise_activity(df, f'Activity Distribution by Day of the Week for {person_name}')
