import polars as pl
import geopandas as gpd
import streamlit as st
import altair as alt
import joblib
from scipy.stats import chi2_contingency

@st.cache_resource
def get_all():
    data = get_data()
    world = get_mapdata()
    joined = get_mapdata(data)
    return data, world, joined

@st.cache_resource
def get_model():
# Load pre-trained model, metrics and label encoder from file
    model = joblib.load("random_forest_model_0.pkl")
    metrics = joblib.load("model_RF0_metrics.pkl")
    label_encoder = joblib.load("label_encoders_RF0.pkl")
    return model, metrics, label_encoder

def get_data(preprocess = True) -> pl.DataFrame:
    #Load and preprocess data.
   
    # Load data
    data  = pl.read_csv("hotel_bookings.csv", null_values= ["Undefined","NA"])
    if preprocess:   
        # Drop columns that are not needed
        data.drop_in_place("agent")
        data.drop_in_place("company")
        
        # remove outliers and errors in adr
        data = data.with_columns(
            pl.when(
                (data["adr"]<0)|
                (data["adr"] > 5000)|
                (((data["adr"] == 0)&(data["market_segment"] != "Complementary"))))
            .then(None).otherwise(data["adr"]).alias("adr")
        )
        # impute missing values in meal
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

        # create arrival date (type date)
        ## convert arrival_date_month to number
    
        d = {"January": "01", "February": "02", "March": "03", "April": "04",
        "May": "05", "June": "06", "July": "07", "August": "08",
        "September": "09", "October": "10", "November": "11", "December": "12"}
        data = data.with_columns(
            pl.col("arrival_date_month").replace_strict(d).alias("arrival_date_month_n")        )
        ## put the day of the month in two digits
        data = data.with_columns(
            pl.when(pl.col("arrival_date_day_of_month") < 10)
            .then(pl.concat_str([pl.lit("0"), pl.col("arrival_date_day_of_month").cast(str)]))
            .otherwise(pl.col("arrival_date_day_of_month").cast(str)).alias("arrival_date_day_of_month")
        )
        ## concatenate year, month and day to create a date string
        data = data.with_columns(
            pl.concat_str([
                pl.col("arrival_date_year").cast(str),      
                pl.lit("/"),
                pl.col("arrival_date_month_n").cast(str), 
                pl.lit("/"),
                pl.col("arrival_date_day_of_month").cast(str) 
            ]).alias("arrival_date") 
        )   
        ## convert the date string to a date type
        data = data.with_columns(
        pl.col("arrival_date").str.to_date().dt.strftime("%Y-%m-%d").alias("arrival_date")
        )
        data = data.with_columns(
            pl.col("arrival_date").str.strptime(pl.Date, "%Y-%m-%d")
        )
    return data


def get_mapdata(data = None) -> gpd.GeoDataFrame:

    #Load the world map data and merge it with the data provided.
    #If no data is provided, return the world map data only.
   
    path = "ne_10m_admin_0_countries.zip"
    world = gpd.read_file(path)
    if data is None:
        return world
    else:
        aggr = data.group_by("country").agg(pl.col("country").count().alias("count"),pl.col("is_canceled").mean().alias("rate_cancelled"))
        data_pd = aggr.to_pandas()
        map_data = world.merge(data_pd, left_on="ADM0_A3_US", right_on="country")
        return map_data


### Functions:

def add_map(chart):
 #Salva la mappa come file html e la carica in streamlit
    chart.save("map.html")
    with open("map.html") as fp:
        st.components.v1.html(fp.read(), width=600, height=600)

def chi2(observed):
#    Perform a chi-squared test on the data provided.
    chi2, pvalue, df, exp =  chi2_contingency(observed)
    return round(chi2,2), pvalue

def bar_chart(data, x, y, color = None, palette = None):

#    Create an Altair bar chart
#    data:    Polars DataFrame
#    x:       Categorical column 
#    y:       Numerical column 
#    color:   Categorical column for coloring bars; if None, returns a simple bar chart
#    palette: Altair color scheme for the 'color' column

    chart = alt.Chart(data).mark_bar().encode(
        alt.X(x+":N", title=x),
        alt.Y(y, title=y))
    if color is None:
        return chart
    else:
        return chart.encode(alt.Color(color+":N", title=color,
            scale=alt.Scale(scheme=palette)))
