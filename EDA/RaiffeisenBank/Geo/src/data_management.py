import sqlite3
import pandas as pd
import geopandas as gpd
import requests
from shapely.geometry import Point
import os

def fetch_and_save_amsterdam_pois(output_folder="data/processed"):
    """
    Fetches points of interest (POIs) from Amsterdam via the Overpass API and saves them as a GeoJSON file.

    Args:
    - output_folder (str): The folder where the GeoJSON file will be saved. Defaults to "data/processed".
    """
    # Define the Overpass API endpoint
    OVERPASS_API = "https://overpass-api.de/api/interpreter"

    # Define the query to fetch all POIs in Amsterdam
    query = """
    [out:json];
    area["name"="Amsterdam"]["admin_level"="8"];
    node(area)[amenity];
    out center;
    """

    # Make the API request
    response = requests.get(OVERPASS_API, params={'data': query})
    data = response.json()

    # Extract nodes from the response
    nodes = data['elements']

    # Create a GeoDataFrame from nodes
    geometry = [Point(node['lon'], node['lat']) for node in nodes]
    pois_gdf = gpd.GeoDataFrame(nodes, geometry=geometry)

    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Save the GeoDataFrame to a file
    pois_gdf.to_file(f"{output_folder}/amsterdam_pois.geojson", driver='GeoJSON')

def get_data_from_db_or_csv(person_id: int, conn: sqlite3.Connection, csv_filename: str) -> pd.DataFrame:
    """
    Attempt to retrieve data for a user using an SQL query from the database. 
    If there's an error, fall back to reading the data from the CSV file.

    Args:
    - person_id (int): ID of the person (1, 2, or 3).
    - conn (sqlite3.Connection): Connection to the SQLite database.
    - csv_filename (str): Name of the CSV file to fall back to.

    Returns:
    - pd.DataFrame: Data for the user.
    """
    try:
        query = f"SELECT * FROM all_persons WHERE person_id = {person_id}"
        df = pd.read_sql_query(query, conn)
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        df = pd.read_csv(csv_filename, delimiter=';')
    return df

def create_geodataframe(df, lat_col='latitude', lon_col='longitude') -> gpd.GeoDataFrame:
    """
    Convert a DataFrame with latitude and longitude columns to a GeoDataFrame.

    Args:
    - df (pd.DataFrame): The original DataFrame.
    - lat_col (str): The name of the column containing latitude values.
    - lon_col (str): The name of the column containing longitude values.

    Returns:
    - gpd.GeoDataFrame: A GeoDataFrame with the points created from latitude and longitude.
    """
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]))
    return gdf

# You can add more functions or code below as needed.
