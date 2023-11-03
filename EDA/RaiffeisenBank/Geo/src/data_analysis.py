import time
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from rtree import index
import geopandas as gpd


# Decorator to measure execution time
def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Execution time for {func.__name__}: {end - start:.2f} seconds")
        return result
    return wrapper

def process_dataframes(dataframes_dict: dict, precision: int = 3) -> dict:
    """
    Process each DataFrame in the dictionary: Convert the timestamp column to datetime format,
    and round the latitude and longitude values.
    
    Args:
    - dataframes_dict (dict): Dictionary containing data for each user.
    - precision (int): Rounding precision for latitude and longitude (default is 3).
    
    Returns:
    - Dictionary containing processed data for each user.
    """
    timestamp_col = 'start_time(YYYYMMddHHmmZ)'
    # Convert the 'start_time(YYYYMMddHHmmZ)' column to datetime format for each DataFrame in the dictionary
    for key, df in dataframes_dict.items():
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
    
    # Round the latitude and longitude values to the specified precision for each DataFrame in the dictionary
    for key, df in dataframes_dict.items():
        df[['latitude', 'longitude']] = df[['latitude', 'longitude']].round(precision)
    
    return dataframes_dict

def compare_locations(dataframes_dict: dict) -> dict:
    """
    Compare locations between users in the provided dictionary to find common locations.
    
    Args:
    - dataframes_dict (dict): Dictionary containing processed data for each user.
    
    Returns:
    - Dictionary of common location counts between each pair of users.
    """
    
    # Create sets of tuples (latitude, longitude) for each person in the dictionary
    location_sets = {key: set(tuple(x) for x in df[['latitude', 'longitude']].values) for key, df in dataframes_dict.items()}
    
    # Find common locations between the users
    common_locations = {}
    keys = list(location_sets.keys())
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            key_pair = f"{keys[i]}_{keys[j]}"
            common_locations[key_pair] = location_sets[keys[i]].intersection(location_sets[keys[j]])
    
    common_location_counts = {key: len(value) for key, value in common_locations.items()}
    
    return common_location_counts


@timing_decorator
def location_and_time_overlap_vectorized(df1: pd.DataFrame,
                                         df2: pd.DataFrame, 
                                         location_threshold: float = 0.001,
                                         time_threshold: pd.Timedelta = pd.Timedelta(minutes=15)) -> pd.DataFrame:
    """
    Checks for overlaps in location and time between two dataframes using vectorized operations.
    
    Args:
    - df1, df2 (pd.DataFrame): Dataframes to compare.
    - location_threshold (float): Maximum allowed difference in location coordinates to consider them as the same.
    - time_threshold (pd.Timedelta): Maximum allowed time difference to consider two visits as overlapping.
    
    Returns:
    - pd.DataFrame: Overlapping entries from both dataframes.
    """
    timestamp_col = 'start_time(YYYYMMddHHmmZ)'
    # Calculate the start and end times for each row in both dataframes
    df1['start_time'] = df1[timestamp_col]
    df1['end_time'] = df1['start_time'] + pd.to_timedelta(df1['duration(ms)'], unit='ms')
    
    df2['start_time'] = df2[timestamp_col]
    df2['end_time'] = df2['start_time'] + pd.to_timedelta(df2['duration(ms)'], unit='ms')
    
    # Cross join the dataframes to compare every location of one user against every location of another user
    df1['key'] = 1
    df2['key'] = 1
    combined = pd.merge(df1, df2, on='key').drop('key', axis=1)
    
    # Filter rows where locations overlap and times overlap
    location_overlap = (combined['latitude_x'].sub(combined['latitude_y']).abs().le(location_threshold)) & \
                       (combined['longitude_x'].sub(combined['longitude_y']).abs().le(location_threshold))
    
    time_overlap = (combined['start_time_x'].le(combined['end_time_y'])) & \
                   (combined['start_time_y'].le(combined['end_time_x']))
    
    overlap_rows = combined[location_overlap & time_overlap].copy()
    
    # Filter only the necessary columns and rename them for clarity
    overlap_df = overlap_rows[['latitude_x', 'longitude_x', 'start_time_x', 'latitude_y', 'longitude_y', 'start_time_y']]
    overlap_df.columns = ['Person_1_Lat', 'Person_1_Lon', 'Person_1_Time', 'Person_2_Lat', 'Person_2_Lon', 'Person_2_Time']
    
    return overlap_df


def average_distance_between_places(df:pd.DataFrame) -> int:
    """Calculate the average distance between consecutive places visited by the user."""
    distances = []
    for i in range(1, len(df)):
        lat_lon_1 = (df.iloc[i-1]['latitude'], df.iloc[i-1]['longitude'])
        lat_lon_2 = (df.iloc[i]['latitude'], df.iloc[i]['longitude'])
        distances.append(geodesic(lat_lon_1, lat_lon_2).kilometers)
    return sum(distances) / len(distances) if distances else 0


@timing_decorator
def calculate_social_activity_metric_v1(dfs_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Calculate the original social activity metric for each DataFrame in the provided dictionary, 
    but format the output as a DataFrame.
    
    Args:
    - dfs_dict (Dict[str, pd.DataFrame]): Dictionary of DataFrames containing location data for different users.
    
    Returns:
    - pd.DataFrame: DataFrame containing the calculated metrics for each user.
    """
    metrics = []
    for person_name, df in dfs_dict.items():
        NPV = len(set(tuple(x) for x in df[['latitude', 'longitude']].values))
        ADBP = average_distance_between_places(df)
        ADAP = df['duration(ms)'].mean() / (1000 * 60)  # Convert from ms to minutes
        SAM = NPV + ADBP + ADAP
        metrics.append({
            "Person": person_name,
            "NPV": NPV,
            "ADBP": ADBP,
            "ADAP": ADAP,
            "SAM": SAM
        })
    
    return pd.DataFrame(metrics)

def harmonic_mean(x: float, y: float) -> float:
    """Calculate the harmonic mean of two numbers."""
    if x == 0 or y == 0:
        return 0
    return (2 * x * y) / (x + y)


@timing_decorator
def calculate_social_activity_metric_v2(dfs_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Calculate the updated social activity metric for each DataFrame in the provided dictionary.
    
    Args:
    - dfs_dict (Dict[str, pd.DataFrame]): Dictionary of DataFrames containing location data for different users.
    
    Returns:
    - pd.DataFrame: DataFrame containing the calculated metrics for each user.
    """
    metrics = []
    for person_name, df in dfs_dict.items():
        NPV = len(set(tuple(x) for x in df[['latitude', 'longitude']].values))
        ADBP = average_distance_between_places(df)
        ADAP = df['duration(ms)'].mean() / (1000 * 60)  # Convert from ms to minutes
        weight = harmonic_mean(ADBP, ADAP)
        SAM = NPV * weight
        metrics.append({
            "Person": person_name,
            "NPV": NPV,
            "ADBP": ADBP,
            "ADAP": ADAP,
            "Weight": weight,
            "SAM": SAM
        })
    
    return pd.DataFrame(metrics)

def calculate_weighted_sam(df: pd.DataFrame) -> float:
    """
    Calculate the Social Activity Metric (SAM) based on the weighted importance of different parts of the day.
    
    Args:
    - df (pd.DataFrame): DataFrame containing the temporal features of a user's activities.
    
    Returns:
    - float: Weighted SAM for the user.
    """
    weighted_sam = 0
    total_weight = sum([weight for _, _, weight in PARTS_OF_DAY.values()])
    
    for part, (start_hour, end_hour, weight) in PARTS_OF_DAY.items():
        part_df = df[(df['hour_of_day'] >= start_hour) & (df['hour_of_day'] < end_hour)]
        average_activities = part_df.shape[0] / len(df['day_of_week'].unique())
        weighted_sam += average_activities * weight
    
    return weighted_sam / total_weight

# Calculate the weighted SAM for each user and store the results in a dictionary
weighted_sam_dict = {}
for person, df in processed_dataframes.items():
    weighted_sam_dict[person] = calculate_weighted_sam(df)

weighted_sam_dict


def weighted_adap(df: pd.DataFrame, weights: dict) -> float:
    """
    Compute the weighted Average Duration At Places (ADAP) based on parts of the day.
    
    Args:
    - df (pd.DataFrame): DataFrame containing user data with 'hour_of_day' and 'duration(ms)' columns.
    - weights (dict): A dictionary mapping parts of the day to their weights.
    
    Returns:
    - float: The weighted ADAP.
    """
    total_weighted_duration = 0
    total_weight = 0
    
    for part, (start, end) in PARTS_OF_DAY.items():
        mask = (df['hour_of_day'] >= start) & (df['hour_of_day'] < end)
        segment_duration = df[mask]['duration(ms)'].sum() / (1000 * 60)  # Convert to minutes
        total_weighted_duration += segment_duration * weights[part]
        total_weight += weights[part]
    
    return total_weighted_duration / total_weight


def calculate_social_activity_metric_weighted(df: pd.DataFrame, weights: dict) -> dict:
    """
    Calculate the Social Activity Metric (SAM) with weighted ADAP.
    
    Args:
    - df (pd.DataFrame): DataFrame containing user data.
    - weights (dict): A dictionary mapping parts of the day to their weights.
    
    Returns:
    - dict: A dictionary containing NPV, ADBP, weighted ADAP, and SAM values.
    """
    NPV = len(set(tuple(x) for x in df[['latitude', 'longitude']].values))
    ADBP = average_distance_between_places(df)
    ADAP_weighted = weighted_adap(df, weights)
    SAM = NPV + ADBP + ADAP_weighted
    
    return {
        "NPV": NPV,
        "ADBP": ADBP,
        "Weighted ADAP": ADAP_weighted,
        "SAM": SAM
    }

def calculate_social_activity_metric_harmonic_v2(dfs, weights):
    """
    Calculate SAM using harmonic mean and part-of-day weights for each DataFrame.
    Args:
    - dfs (dict): Dictionary of DataFrames containing location data for different users.
    - weights (dict): Weights assigned to different parts of the day.
    
    Returns:
    - pd.DataFrame: DataFrame containing the calculated metrics for each user.
    """
    metrics = {}
    for person, df in dfs.items():
        NPV = len(set(tuple(x) for x in df[['latitude', 'longitude']].values))
        ADBP = average_distance_between_places(df)
        weighted_ADAP = weighted_adap(df, weights)
        weight = harmonic_mean(ADBP, weighted_ADAP)
        SAM = NPV * weight
        metrics[person] = {
            "NPV": NPV,
            "ADBP": ADBP,
            "Weighted ADAP": weighted_ADAP,
            "Weight": weight,
            "SAM": SAM
        }
    
    return pd.DataFrame(metrics).T


@timing_decorator
def location_and_time_overlap_count_rtree(df1: pd.DataFrame, df2: pd.DataFrame, location_threshold: float = 0.00001, 
                                              time_threshold: pd.Timedelta = pd.Timedelta(minutes=1)) -> int:
    overlaps_count = 0
    
    # Step 1: Index df2 locations using R-tree
    idx = index.Index()
    for i, row in df2.iterrows():
        idx.insert(i, (row['longitude'], row['latitude'], row['longitude'], row['latitude']))
    
    # Step 2: Query R-tree using df1 locations
    for _, row1 in df1.iterrows():
        left, bottom = row1['longitude'] - location_threshold, row1['latitude'] - location_threshold
        right, top = row1['longitude'] + location_threshold, row1['latitude'] + location_threshold
        
        for j in idx.intersection((left, bottom, right, top)):
            row2 = df2.iloc[j]
            
            # Step 3: Temporal Filtering
            start_time_1 = row1['start_time(YYYYMMddHHmmZ)']
            end_time_1 = start_time_1 + pd.Timedelta(milliseconds=row1['duration(ms)'])
            start_time_2 = row2['start_time(YYYYMMddHHmmZ)']
            end_time_2 = start_time_2 + pd.Timedelta(milliseconds=row2['duration(ms)'])
            
            if max(start_time_1, start_time_2) <= min(end_time_1, end_time_2) + time_threshold:
                overlaps_count += 1
                
    return overlaps_count


def check_visits_to_amenities(person_gdf, pois_gdf, amenities_of_interest, distance_threshold=0.0001):
    """
    Check if a person visited specific amenities.
    
    Args:
    - person_gdf (GeoDataFrame): The person's movement data.
    - pois_gdf (GeoDataFrame): The POIs data.
    - amenities_of_interest (list): List of amenities to check.
    - distance_threshold (float): The distance threshold to consider a visit.
    
    Returns:
    - dict: Dictionary with amenity as the key and the count of visits as the value.
    """
    
    # Filter the POIs to only include the amenities of interest
    filtered_pois = pois_gdf[pois_gdf['amenity'].isin(amenities_of_interest)]
    
    visit_counts = {amenity: 0 for amenity in amenities_of_interest}
    
    # For each record in the person's data, check if they were near any of the filtered POIs
    for _, person_row in person_gdf.iterrows():
        for _, poi_row in filtered_pois.iterrows():
            if person_row['geometry'].distance(poi_row['geometry']) <= distance_threshold:
                visit_counts[poi_row['amenity']] += 1
                
    return visit_counts


@timing_decorator
def calculate_enhanced_sam_optimized(user_gdf: gpd.GeoDataFrame, 
                                     pois_gdf: gpd.GeoDataFrame, distance_threshold: float = 0.001) -> float:
    """
    Calculate the enhanced SAM score by incorporating visits to social amenities using optimized approach.
    
    Args:
    - user_gdf (gpd.GeoDataFrame): User location data.
    - pois_gdf (gpd.GeoDataFrame): Points of interest data.
    - distance_threshold (float): Threshold for considering a user's location to be at a POI.
    
    Returns:
    - float: Enhanced SAM score.
    """
    social_amenities = ['bar', 'cafe', 'club', 'restaurant', 'theater', 'cinema']
    social_pois = pois_gdf[pois_gdf['amenity'].isin(social_amenities)]
    
    # Create a buffer around each user's point based on the distance threshold
    user_gdf['geometry'] = user_gdf.buffer(distance_threshold)
    
    # Use spatial join to find overlaps between user locations and POIs
    joined = gpd.sjoin(user_gdf, social_pois, how="inner", op="intersects")
    
    # The length of the joined GeoDataFrame gives the number of overlaps
    visits_count = len(joined)
    
    # For demonstration, we're using a basic SAM calculation: Number of POIs visited.
    # This can be enhanced with more sophisticated metrics like duration, frequency, etc.
    enhanced_sam = visits_count
    return enhanced_sam


def calculate_enhanced_sam(user_gdf: gpd.GeoDataFrame,
                           pois_gdf: gpd.GeoDataFrame, distance_threshold: float = 0.0001) -> float:
    """
    Calculate the enhanced SAM score by incorporating visits to social amenities.

    Args:
    - user_gdf (gpd.GeoDataFrame): User location data.
    - pois_gdf (gpd.GeoDataFrame): Points of interest data.
    - distance_threshold (float): Threshold for considering a user's location to be at a POI.

    Returns:
    - float: Enhanced SAM score.
    """
    social_amenities = ['bar', 'cafe', 'club', 'restaurant', 'theater', 'cinema']
    social_pois = pois_gdf[pois_gdf['amenity'].isin(social_amenities)]

    # Check for visits to these social amenities
    visits_count = 0
    for user_row in user_gdf.itertuples():
        user_point = user_row.geometry
        # Use spatial index to speed up the process
        possible_matches_index = list(social_pois.sindex.intersection(user_point.bounds))
        possible_matches = social_pois.iloc[possible_matches_index]
        precise_matches = possible_matches[possible_matches.distance(user_point) <= distance_threshold]
        visits_count += len(precise_matches)

    # Calculate the SAM components
    number_of_locations = len(user_gdf['geometry'].unique())
    total_distance = user_gdf['geometry'].distance(user_gdf['geometry'].shift()).sum()
    total_time = user_gdf['duration(ms)'].sum()

    # Use harmonic mean to combine the SAM components
    sam_components = [number_of_locations, total_distance, total_time, visits_count]
    sam_components = [x if x != 0 else 1 for x in sam_components]  # Replace 0 with 1 to avoid division by zero
    harmonic_mean_sam = len(sam_components) / np.sum(1.0 / np.array(sam_components))

    return harmonic_mean_sam
