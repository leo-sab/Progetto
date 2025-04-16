import polars as pl
import geopandas as gpd

def get_data():
    """
    Load and preprocess data 
    """
    # Load data
    data  = pl.read_csv("hotel_bookings.csv", null_values= ["Undefined"])
    




