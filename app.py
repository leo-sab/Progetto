import streamlit as st
import polars as pl
import altair as alt
from preprocess import *
from scipy.stats import chi2_contingency
# global variables
cat_color = "category20" 
sequential_color = "viridis"

# Load data
data = get_data()
world = get_mapdata()
joined = get_mapdata(data)


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
chart = alt.Chart(data).mark_boxplot().encode(
    alt.X('adr:Q', title='Prezzo medio per notte (€)'),
    alt.Facet("hotel:N", title = ""),
    alt.Color("hotel:N", title="Tipo di hotel")

).properties(
    title="Distribuzione del prezzo medio per notte per tipo di hotel")

st.altair_chart(chart, use_container_width=True)
"""
Vediamo che il prezzo degli hotel di città hanno un prezzo mediano più a ...

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
    tooltip=["hotel:N","adr_bin:N","cancellation_rate:Q","count:Q"]
).properties(title="Tasso di cancellazione per hotel e prezzo medio per notte (diviso in 5 intervalli)")
st.altair_chart(chart, use_container_width=True)
"""
Il grafico rappresenta la variazione del tasso di cancellazione per fascie di prezzo (sono stati utilizzati i quantili 0.2,0.4,0.6,0.8 di adr) 
dei 2 tipi di hotel, utilizzando la grandezza del cerchio per mostrare quante prenotazioni ci sono in ogni fascia.

Notiamo che il tasso di cancellazione dei 2 tipi di hotel per la fascia di prezzo più bassa ha un comportamento opposto: 
i resort hanno il tasso più basso mentre i city hotel il più alto. 
Aumentando il prezzo quello dei resort tende a
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



if  st.selectbox("Vuoi visualizzare la distribuzione delle prenotazioni in tutto il mondo o solo in Europa", [ "Europa","Mondo"]) == "Europa":
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
    color=alt.Color("count:Q", scale=alt.Scale(scheme=sequential_color, domain = [0,maxdomain]),legend=alt.Legend(title="Prenotazioni", orient="top-left")),
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
che va da 0 alla seconda più numerosa spuntando la casella "Escludendo il Portogallo").
"""
if st.selectbox("Seleziona mondo o europa", [ "Europa","Mondo"]) == "Europa":
    map_type = "azimuthalEqualArea"
    center = (10, 48)
    scale = 800
else:
    map_type = "equalEarth" 
    center = (0, 0)
    scale = 150 
base = alt.Chart(world).mark_geoshape(color="lightgray").properties(width=600, height=600).project(
    type=map_type,
    scale=scale,
    center=center
)

mappa = alt.Chart(joined).mark_geoshape().encode(
    color=alt.Color("rate_cancelled:Q", scale=alt.Scale(scheme=sequential_color),
                legend=alt.Legend(title="Tasso di Cancellazione", format=".2%",orient="top-left")),
    tooltip=["country:N", alt.Tooltip("rate_cancelled:Q", format=".2%"), "count:Q"]
).project(
    type=map_type,
    scale=scale,
    center=center
).properties(
    width=600,
    height=600,
    title="Tasso di Cancellazione"
)

if st.checkbox("Vuoi un grafico che includa anche la numerosità delle prenotazioni (in scala logaritmica) per una migliore visualizzazione? ", [False, True]) == True:
    mappa = mappa.encode(opacity=alt.Opacity('count:Q', scale=alt.Scale(type = "log",range=[0,1]),
                 legend=alt.Legend(title="Numero Prenotazioni",orient="top-left", format=".0f")))

add_map(base + mappa)

"""
L'analisi del grafico rivela che il Portogallo presenta uno dei tassi di cancellazione più elevati tra i Paesi visualizzati, 
suggerendo una potenziale correlazione tra la localizzazione degli hotel e la frequenza di cancellazione. 
Un'ipotesi plausibile è che la vicinanza geografica possa incentivare una maggiore propensione alla cancellazione da parte
 dei clienti che prenotano in Portogallo. 
 La ridotta incidenza di costi o complicazioni logistiche associate alla cancellazione per i clienti locali potrebbe contribuire a questo fenomeno.

Questa osservazione potrebbe fornire spunti per interpretare le differenze nei tassi di cancellazione riscontrate negli hotel di 
città appartenenti a fasce di prezzo inferiori. Se una porzione considerevole della clientela di tali strutture è costituita da residenti portoghesi, 
la loro ipotizzata maggiore tendenza alla cancellazione potrebbe essere un fattore determinante nel tasso complessivo di cancellazione per questa categoria di hotel.

Parallelamente, si potrebbe ipotizzare un effetto di auto-selezione di tipo economico tra i clienti internazionali che viaggiano in Portogallo.
I viaggiatori stranieri, affrontando costi e impegni maggiori legati al viaggio, potrebbero dimostrare una minore propensione alla cancellazione, 
specialmente se orientati verso hotel di fascia di prezzo più alta. Questa dinamica suggerisce una potenziale relazione tra la nazionalità del cliente, 
la fascia di prezzo dell'hotel e la probabilità di cancellazione, meritevole di ulteriori indagini.

Tabella hotel di città con adr <= 65:
"""
temp = data.filter((pl.col("adr")<=65) & (pl.col("hotel") == "City Hotel")).with_columns(
    pl.when(pl.col("country") == "PRT").then(pl.lit("Portogallo")).otherwise(pl.lit("Altri Paesi")).alias("country")
).group_by("country").agg(
    pl.count().alias("numero di prenotazioni"), pl.col("is_canceled").cast(pl.Float64).mean().alias("tasso di cancellazione"))
st.write( temp )
"""
Tabella restanti prenotazioni:"""
temp1 = data.filter((pl.col("adr")>65) | (pl.col("hotel") != "City Hotel")).with_columns(
    pl.when(pl.col("country") == "PRT").then(pl.lit("Portogallo")).otherwise(pl.lit("Altri Paesi")).alias("country")
).group_by("country").agg(
    pl.count().alias("numero di prenotazioni"),pl.col("is_canceled").mean().alias("tasso di cancellazione"))
st.write( temp1 )

observed = pl.concat([temp,temp1]).select(pl.col("numero di prenotazioni")).to_numpy().reshape(2,2)

chi ,p  = chi2(observed)
"""
Già senza ulteriori analisi si osserva che i 2 gruppi sono molto disomigenei, però volendo verificare ciò in maniera più rigorosa
possiamo utilizzare un test statistico per verificare l'assunzione di indipendenza (chi-quadro):
"""
st.write(f"La statistica test è pari a {chi}",f" con un P-value di {p}")
""" 
Quindi abbiamo un ulteriore indizio a favore dell'ipotesi fatta in precedenza.
"""

#book changes 
#special requests
# deposit type

