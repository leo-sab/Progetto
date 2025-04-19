import streamlit as st
import polars as pl
import altair as alt
from preprocess import *

# global variables
cat_color = "category20" 

# Load data
data = get_data()
world = get_mapdata()
joined = get_mapdata(data)

# functions
def add_map(chart):
    """Salva la mappa come file html e la carica in streamlit"""
    chart.save("map.html")
    with open("map.html") as fp:
        st.components.v1.html(fp.read(), width=600, height=600)

#### MAIN CODE ####
st.set_page_config(page_title="Hotel Bookings", page_icon="🏨", layout="centered")

st.title("Dashboard analisi prenotazioni hotel")
"""
L'obiettivo di questo progetto è quello di sviluppare un'applicazione per visualizzare variabili del dataset hotel_bookings relative a delle prenotazioni di clienti di
 hotel portoghesi nel periodo che va dall'agosto 2015 all'agosto 2017, indagando le principali cause di cancellazione e relazioni tra variabili con grafici e metodi di machine learning.
# Presentazione del dataset:
 """
st.write("Il dataset contiene", data.shape[0], "prenotazioni e ", data.shape[1], "variabili, l'analisi si concentrerà solo su alcune di esse.")

st.write(data.describe())
"""In particolare ci chiediamo: 
- Quante persone cancellano la prenotazione?
- Quali variabili influenzano la cancellazione?"""


# bar chart is_canceled
base = alt.Chart(data).transform_aggregate(
    count='count()',
    groupby=["is_canceled"]
).transform_calculate(
    proportion="datum.count / " + str(data.shape[0])
)

chart = base.mark_bar().encode(
    alt.X("is_canceled:N", title="Cancellazione"),
    alt.Y("count:Q", title="Numero Prenotazioni"),
    alt.Color("is_canceled:N", title="Cancellazione", 
            scale=alt.Scale(scheme=cat_color))
)

text = base.mark_text(
    align='center',
    baseline='bottom',
    dy=-2  
    ,size = 15, color = "white" 
).encode(
    alt.X("is_canceled:N"),
    alt.Y("count:Q"),
    alt.Text("proportion:Q", format=".0%")  
)

st.altair_chart(chart+text, use_container_width=True)
"""
Il 37% delle prenotazioni viene cancellato, proviamo a vedere se il tipo di Hotel
ha un'influenza sulla cancellazione.
"""
# bar chart hotel and is_canceled

chart = alt.Chart(data).mark_bar().encode(
    alt.Y("is_canceled:N", title=""),
    alt.X("count()", title="Numero Prenotazioni"),
    alt.Facet("hotel:N", title=""),
    alt.Color("is_canceled:N", title="Cancellazione")
    ).properties(title="Cancellazione per tipo di hotel")
st.altair_chart(chart, use_container_width=True)

rate_city = round(data.filter(
    (data["hotel"] == "City Hotel") & (data["is_canceled"] == 1)
).shape[0] / data.filter(
    data["hotel"] == "City Hotel"
).shape[0] , 2)
rate_resort = round(data.filter(
    (data["hotel"] == "Resort Hotel") & (data["is_canceled"] == 1)
).shape[0] / data.filter(
    data["hotel"] == "Resort Hotel"
).shape[0] , 2)

st.write("Come possiamo vedere dal grafico, nel dataset ci " \
"sono più prenotazioni su hotel di città e sempre quelli di città hanno un tasso di " \
"cancellazione più alto, ossia", rate_city, " contro quello degli hotel resort che " \
"è pari a ", rate_resort)
"""
Questa differenza è chiaramente significativa e potrebbe essere dovuta a vari fattori,
come ad esempio un tipo diverso di clientela, prezzi, prenotazioni fatte mediamente in tempi diversi o alla stagionalità.

Proveremo in seguito ad indagare più a fondo.
"""


"""
Concentriamoci ora sulla variabile 'adr' (Average Daily Rate), 
che rappresenta il prezzo medio per notte.
"""
############################################################
# Calcoliamo le mediane utilizzando solo Polars
median_adr_hotel = data.group_by("hotel").agg(
    pl.col("adr").median().alias("median_adr")
)

# Prepariamo i dati per l'istogramma speculare
# Creiamo un dataframe per l'istogramma con bin predefiniti
hist_data = data.select(
    pl.col("hotel"),
    pl.col("adr")
)

# Definiamo i colori per ciascun hotel
hotel_colors = {
    "City Hotel": "#1f77b4",  # Blu
    "Resort Hotel": "#aec7e8"  # Azzurro
}

# Creiamo l'istogramma speculare
hist = alt.Chart(hist_data.to_pandas()).transform_bin(
    "adr_bin",
    "adr",
    bin={"maxbins": 60}
).transform_aggregate(
    count="count()",
    groupby=["adr_bin", "hotel"]
).transform_calculate(
    # Usiamo ancora i valori negativi per Resort Hotel, ma aggiungiamo un valore assoluto per il tooltip
    count_signed="datum.hotel == 'City Hotel' ? datum.count : -datum.count",
    count_abs="Math.abs(datum.count)"
).mark_bar().encode(
    x=alt.X("adr_bin:Q", title="Prezzo medio per notte (€)", axis=alt.Axis(format=",.0f")),
    y=alt.Y("count_signed:Q", title="Frequenza", 
           # Formattiamo l'asse y per mostrare i valori assoluti invece di negativi
           axis=alt.Axis(format="|~s", labelExpr="Math.abs(datum.value)")),
    color=alt.Color("hotel:N", scale=alt.Scale(domain=list(hotel_colors.keys()), range=list(hotel_colors.values()))),
    tooltip=[
        alt.Tooltip("hotel:N", title="Tipo Hotel"),
        alt.Tooltip("adr_bin:Q", title="Prezzo", format=",.0f"),
        alt.Tooltip("count_abs:Q", title="Frequenza", format=",")
    ]
)

# Aggiungiamo etichette agli assi per chiarire quale hotel è sopra e quale sotto
top_label = alt.Chart({"values": [{"text": "City Hotel"}]}).mark_text(
    align="left",
    baseline="top",
    fontSize=12,
    dy=-180
).encode(
    x=alt.value(0),
    y=alt.value(0),
    text="text:N"
)

bottom_label = alt.Chart({"values": [{"text": "Resort Hotel"}]}).mark_text(
    align="left",
    baseline="bottom",
    fontSize=12,
    dy=180
).encode(
    x=alt.value(0),
    y=alt.value(0),
    text="text:N"
)

# Linee delle mediane per City Hotel
city_median_line = alt.Chart(
    median_adr_hotel.filter(pl.col("hotel") == "City Hotel")
).mark_rule(
    strokeDash=[4, 2],
    color="red",
    size=2
).encode(
    x="median_adr:Q",
    tooltip=alt.Tooltip("median_adr:Q", title="Mediana City Hotel", format=",.2f")
)

# Aggiunta testo per la mediana City Hotel
city_median_text = alt.Chart(
    median_adr_hotel.filter(pl.col("hotel") == "City Hotel")
).mark_text(
    align="left",
    baseline="middle",
    dx=5,
    dy=-40,
    fontSize=12,
    fontWeight="bold",
    color="red"
).encode(
    x="median_adr:Q",
    text=alt.Text("median_adr:Q", format=",.2f €")
)

# Linee delle mediane per Resort Hotel
resort_median_line = alt.Chart(
    median_adr_hotel.filter(pl.col("hotel") == "Resort Hotel")
).mark_rule(
    strokeDash=[4, 2],
    color="darkred",
    size=2
).encode(
    x="median_adr:Q",
    tooltip=alt.Tooltip("median_adr:Q", title="Mediana Resort Hotel", format=",.2f")
)

# Aggiunta testo per la mediana Resort Hotel
resort_median_text = alt.Chart(
    median_adr_hotel.filter(pl.col("hotel") == "Resort Hotel")
).mark_text(
    align="left",
    baseline="middle",
    dx=5,
    dy=40,
    fontSize=12,
    fontWeight="bold",
    color="darkred"
).encode(
    x="median_adr:Q",
    text=alt.Text("median_adr:Q", format=",.2f €")
)

# Combiniamo tutto in un grafico finale
final_chart = (
    hist 
)

# Visualizziamo il grafico
st.altair_chart(final_chart, use_container_width=True)
"""
Vediamo che il prezzo degli hotel di città hanno un prezzo mediano più a
"""

# prepare data for chart
bins = [0.2,0.4,0.6,0.8]
labels = ["(0, 65]", "(65, 85.5]","(85.5, 105.8]",
           "(105.8, 135]", "(135, 510]"]
temp = data.with_columns(
    pl.col("adr").qcut(quantiles=bins,labels=labels).alias("adr_bin")
)

bin = temp.group_by("hotel","adr_bin").agg(
    pl.col("is_canceled").mean().alias("cancellation_rate"),
    pl.col("hotel").count().alias("count")
    )
# buble chart adr_bin and hotel
chart = alt.Chart(bin).mark_circle(size=100).encode(
    alt.X("adr_bin:N", title="Prezzo medio per notte", sort = labels),
    alt.Y("cancellation_rate:Q", title="tasso di cancellazione"),
    alt.Size("count:Q", title="Numero Prenotazioni"),
    alt.Color("hotel:N", title="Tipo di hotel",
            scale=alt.Scale(scheme=cat_color)),
    tooltip=["hotel:N","adr_bin:N","cancellation_rate:Q"]
).properties(title="Tasso di cancellazione per hotel e prezzo medio per notte (diviso in 5 intervalli)")
st.altair_chart(chart, use_container_width=True)
"""
si
"""
chart = alt.Chart(data).mark_line(opacity = 0.4).encode(
    alt.X("arrival_date:T", title="Data di arrivo"),
    alt.Y("mean(adr):Q", title="Prezzo medio per notte"),
    alt.Color("hotel:N", title="Tipo di hotel",
            scale=alt.Scale(scheme=cat_color)),
).properties(
    title="Andamento prezzo medio (con smoothing)"
)
chart1 = alt.Chart(data.to_pandas()).transform_loess(
    "arrival_date", "adr", groupby=["hotel"], bandwidth=0.04
).mark_line().encode(
    alt.X("arrival_date:T"),
    alt.Y("mean(adr):Q"),
    alt.Color("hotel:N", title="Tipo di hotel")
    )
st.altair_chart(chart + chart1, use_container_width=True)

"""
Il prezzo medio per notte per entrambi gli holel sembra in leggero aumento,
evidente stagionalità per i resort hotel, che hanno sempre il picco dei prezzi nel mese di Agosto (precisamente a Ferragosto).
Da notare gli spike nel prezzo dei resort non catturati dallo smoothing verso fine Dicembre di entrambi gli anni che possiamo spiegare con 
il capodanno.
Il prezzo degli hotel di città invece risultano più stabili e mediamente maggiori dei resort. 
"""
chart = alt.Chart(data.with_columns(
    (pl.col("arrival_date").dt.strftime("%Y-%m")).alias("month")
)).mark_line().encode(
    alt.X("month:T",title="Data di arrivo"),
    alt.Y("count()", title="Numero Prenotazioni"),
    alt.Color("hotel:N", title="Tipo di hotel",
            scale=alt.Scale(scheme=cat_color)),
)
st.altair_chart(chart, use_container_width=True)
"""
Numero di prenotazioni nei Resort è quasi costante mentre quella degli hotel di città hanno picchi maggiori 
sopratutto in Ottobre e primi mesi estivi.
"""

chart = alt.Chart(data.with_columns(
    (pl.col("arrival_date").dt.strftime("%Y-%m")).alias("month")
)).mark_line().encode(
    alt.X("month:T",title="Data di arrivo"),
    alt.Y("count()", title="Numero Prenotazioni",),
    alt.Color("is_canceled:N", title="Cancellazione",
            scale=alt.Scale(scheme=cat_color)),
)
st.altair_chart(chart, use_container_width=True)

# chart lead time and type of hotel
chart = alt.Chart(data).mark_area().encode(
    alt.X("lead_time:Q", title="Lead time", scale  = alt.Scale(domain=[0, 630])),
    alt.Y("count()", title="Numero Prenotazioni"),
    alt.Color("is_canceled:N", title="Cancellazione",
            scale=alt.Scale(scheme=cat_color)),
    alt.Facet("hotel:N")
).properties(title="Lead time per tipo di hotel")
st.altair_chart(chart, use_container_width=True)
"""
Più aumenta il tempo tra prenotazione e arrivo, più aumenta il tasso di cancellazione.
City hotel hanno una distribuzione di lead time più "distesa" rispetto a quella dei Resort hotel, che tendono ad avere più clienti che prenotano
a ridosso della data di arrivo.
"""



if st.selectbox("Seleziona mondo o europa", [ "Europa","Mondo"]) == "Europa":
    map_type = "azimuthalEqualArea"
    center = (10, 48)
    scale = 800
else:
    map_type = "equalEarth" 
    center = (0, 0)
    scale = 150 

if st.checkbox("Escludendo il Portogallo? ", [False, True]) == False:
    maxdomain = 47651
else:
    maxdomain = 12073

chart = alt.Chart(world).mark_geoshape(color= "lightgray").properties(width=600, height=600).project(
    type= map_type,
    scale = scale,
    center=center
)

mappa = alt.Chart(joined).mark_geoshape().encode(
    color=alt.Color("count:Q", scale=alt.Scale(scheme="plasma", domain = [0,maxdomain]),legend=alt.Legend(title="Prenotazioni", orient="top-left")),
    tooltip=["country:N", "count:Q"]
).project(
    type= map_type,
    scale=scale,
    center=center
).properties(
    width=600,
    height=600,
    title="Mappa prenotazioni"
)
add_map(chart + mappa)
"""
La mappa mostra la distribuzione delle prenotazioni in tutto il mondo,
vediamo che la maggior parte delle prenotazioni proviene da paesi europei, in particolare dal Portogallo (puoi visualizzare le prenotazioni su una scala 
che va da 0 alla seconda più numerosa spuntando la tick su escludi Portogallo).
"""
