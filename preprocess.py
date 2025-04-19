import polars as pl
import geopandas as gpd
import streamlit as st

@st.cache_data
def get_data():
    """
    Load and preprocess data 
    """
    # Load data
    data  = pl.read_csv("hotel_bookings.csv", null_values= ["Undefined","NA"])
         
    # Drop columns that are not needed
    data.drop_in_place("agent")
    data.drop_in_place("company")
    
    # remove outliers and errors
    data = data.with_columns(
        pl.when(
            (data["adr"]<0)|
            (data["adr"] > 5000)|
            (((data["adr"] == 0)&(data["market_segment"] != "Complementary"))))
        .then(None).otherwise(data["adr"]).alias("adr")
    )

    data = data.with_columns(
        pl.when(data["meal"].is_null()).then(pl.lit("SC")).otherwise(data["meal"]).alias("meal")
    )
    #fix error in country names
    data = data.with_columns(
        pl.when(data["country"] == "CN").
        then(pl.lit("CAN")).
        otherwise(data["country"]).alias("country")
    )
    # remove null values
    data = data.drop_nulls()

    # create arrival date
    d = {"January": "01", "February": "02", "March": "03", "April": "04",
     "May": "05", "June": "06", "July": "07", "August": "08",
     "September": "09", "October": "10", "November": "11", "December": "12"}
    data = data.with_columns(
        pl.col("arrival_date_month").map_elements(lambda x: d[x]).alias("arrival_date_month_n") # Conversione in numero
    )
# metto uno zero davanti al giorno se è minore di 10 (esempio: 1-> 01):
    data = data.with_columns(
        pl.when(pl.col("arrival_date_day_of_month") < 10)
        .then(pl.concat_str([pl.lit("0"), pl.col("arrival_date_day_of_month").cast(str)]))
        .otherwise(pl.col("arrival_date_day_of_month").cast(str)).alias("arrival_date_day_of_month")
    )
    data = data.with_columns(
        pl.concat_str([
            pl.col("arrival_date_year").cast(str),      
            pl.lit("/"),
            pl.col("arrival_date_month_n").cast(str), 
            pl.lit("/"),
            pl.col("arrival_date_day_of_month").cast(str) 
        ]).alias("arrival_date") 
    )   
    data = data.with_columns(
    pl.col("arrival_date").str.to_date().dt.strftime("%Y-%m-%d").alias("arrival_date")
    )
    data = data.with_columns(
        pl.col("arrival_date").str.strptime(pl.Date, "%Y-%m-%d")
    )
    return data


@st.cache_data
def get_mapdata(data = None) -> gpd.GeoDataFrame:
    """
    prepare data for maps
    """
    url = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip"
    world = gpd.read_file(url)
    if data is None:
        return world
    else:
        aggr = data.group_by("country").agg(pl.col("country").count().alias("count"))
        data_pd = aggr.to_pandas()
        map_data = world.merge(data_pd, left_on="ADM0_A3_US", right_on="country")
        return map_data


def add_map(chart):
    """Salva la mappa come file html e la carica in streamlit"""
    chart.save("map.html")
    with open("map.html") as fp:
        st.components.v1.html(fp.read(), width=600, height=600)




