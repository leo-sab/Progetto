import streamlit as st
import polars as pl
import altair as alt
from preprocess import *
from scipy.stats import chi2_contingency


st.set_page_config(page_title="Hotel Bookings", page_icon="🏨", layout="centered")

# global variables
cat_color = "category20" 
sequential_color = "viridis"
divergent_color = "redblue"


# Load data
data, world, joined = get_all()


#### MAIN CODE ####

st.title("EDA Prenotazioni Hotel")
"""
Questo progetto ha come obiettivo l'analisi di un dataset di prenotazioni di hotel, si divide in 2 parti:
- Analisi esplorativa dei dati, con visualizzazioni e grafici interattivi.
- Creazione di un modello di previsione
# Presentazione del dataset:
 """
st.write("Il dataset contiene", data.shape[0], "prenotazioni e ", data.shape[1], "variabili, l'analisi si concentrerà solo su alcune di esse.")

with  st.expander("Mostra summary dei dati ") : st.write(data.drop(pl.col("index")).describe())

"""
Per maggiori informazioni sulle variabili del dataset consultare il README
In particolare ci chiediamo: 
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
Questa differenza è potrebbe essere dovuta a vari fattori, 
come ad esempio un tipo diverso di clientela, prezzi, prenotazioni fatte mediamente in tempi diversi o alla stagionalità;
proveremo in seguito ad indagare più a fondo.
"""


"""
Vediamo ora come si distribuisce la variabile 'adr' (Average Daily Rate), 
che rappresenta il prezzo medio per notte.
"""
# boxplot adr

chart = alt.Chart(data).mark_boxplot().encode(
    alt.X('adr:Q', title='Prezzo medio per notte (€)'),
    alt.Facet("hotel:N", title = ""),
    alt.Color("hotel:N", title="Tipo di hotel")
).properties(
    title="Distribuzione del prezzo medio per notte per tipo di hotel",
    height = 50
)

st.altair_chart(chart, use_container_width=True)
"""
Questo boxplot rappresenta la distribuzione del prezzo medio per notte per tuoi di hotel.
La parte centrale rappresenta il 50% dei dati (tra il primo e il terzo quartile), la linea
rappresenta la mediana, i punti fuori dalla parte centrale sono fli outliers.
Si vede chiaramente che questa distribuzione è fortemente asimmetrica, si può vedere che gli hotel di città 
hanno un prezzo mediano maggiore rispetto ai resort ma con hanno una distribuzione più "schiacciata" con code più corte 
rispetto ai resort. 
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

Si può notare che il tasso di cancellazione dei 2 tipi di hotel per la fascia di prezzo più bassa ha un comportamento opposto: 
i resort hanno il tasso più basso mentre i city hotel il più alto. 
Aumentando il prezzo quello dei resort tende a salire mentre quello degli hotel di città tende a scendere.

"""
# chart adr and arrival date, opaco
chart = alt.Chart(data).mark_line(opacity = 0.4).encode(
    alt.X("arrival_date:T", title="Data di arrivo"),
    alt.Y("median(adr):Q", title="Prezzo medio per notte"),
    alt.Color("hotel:N", title="Tipo di hotel",
            scale=alt.Scale(scheme=cat_color)),
).properties(
    title="Andamento prezzo mediano (con smoothing)"
)
# add smoothing 
chart1 = alt.Chart(data.to_pandas()).transform_loess(
    "arrival_date", "adr", groupby=["hotel"], bandwidth=0.04
).mark_line().encode(
    alt.X("arrival_date:T"),
    alt.Y("median(adr):Q"),
    alt.Color("hotel:N", title="Tipo di hotel")
    )
st.altair_chart(chart + chart1, use_container_width=True)

"""
Il seguente grafico rappresenta l'andamento del prezzo mediano durante preso in esame,per semplificare la visualizzazione
 e rendere più evidente la stagionalità e il trend e rimuovere il rumore aggiunto lo smoothing.
Il prezzo mediano per notte per entrambi ti tipi di hotel sembra in aumento,
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

chart =alt.Chart(data).mark_area().encode(
    y = alt.Y( "count()").stack("normalize"),
    x = alt.X("arrival_date:T"),
    color= alt.Color("hotel:N")
)
st.altair_chart(chart, use_container_width=True)

chart = alt.Chart(data).mark_circle().encode(
    y='arrival_date_month_n:O',
    x='arrival_date_year:O',
    size='count():Q',
    facet = "hotel:N",
    color = 'hotel:N'
)
st.altair_chart(chart, use_container_width= True)
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
).interactive().properties(title="Lead time per tipo di hotel")
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
L'analisi del grafico rivela che il Portogallo, nazione con più prenotazioni, presenta uno dei tassi di cancellazione più elevati tra i Paesi visualizzati, 
questo ci suggerisce una potenziale separazione in 2 cluster (clienti locali e clienti stranieri). 
Un'ipotesi plausibile è che la vicinanza geografica possa incentivare una maggiore propensione alla cancellazione da parte
 dei clienti che prenotano in Portogallo. 
 La ridotta incidenza di costi o complicazioni logistiche associate alla cancellazione per i clienti locali potrebbero contribuire 
 a fare prenotazioni con più leggerezza rispetto a chi provviene dall'estero.

Questa osservazione potrebbe fornire spunti per interpretare le differenze nei tassi di cancellazione riscontrate negli hotel di 
città appartenenti a fasce di prezzo inferiori. Se una porzione considerevole della clientela di tali strutture è costituita da residenti portoghesi, 
la loro ipotizzata maggiore tendenza alla cancellazione potrebbe essere un fattore determinante nel tasso complessivo di cancellazione per questa categoria di hotel.

Parallelamente, si potrebbe ipotizzare un effetto di auto-selezione di tipo economico tra i clienti internazionali che viaggiano in Portogallo.
I viaggiatori stranieri, affrontando costi e impegni maggiori legati al viaggio, potrebbero dimostrare una minore propensione alla cancellazione, 
specialmente se orientati verso hotel di fascia di prezzo più alta. 

Tabella di tutte le prenotazioni:
"""
temp = data.with_columns(
    pl.when(pl.col("country") == "PRT").then(pl.lit("Portogallo")).otherwise(pl.lit("Altri Paesi")).alias("country")
).group_by("country").agg(
    pl.len().alias("numero di prenotazioni"),pl.col("is_canceled").mean().alias("tasso di cancellazione"))
st.write( temp )
"""
Tabella hotel di città con adr <= 65:
"""
temp1 = data.filter((pl.col("adr")<=65) & (pl.col("hotel") == "City Hotel")).with_columns(
    pl.when(pl.col("country") == "PRT").then(pl.lit("Portogallo")).otherwise(pl.lit("Altri Paesi")).alias("country")
).group_by("country").agg(
    pl.len().alias("numero di prenotazioni"), pl.col("is_canceled").cast(pl.Float64).mean().alias("tasso di cancellazione"))
st.write( temp1 )
"""
Tabella Resort Hotel con adr <= 65:
"""
temp2 = data.filter((pl.col("adr")<=65) & (pl.col("hotel") == "Resort Hotel")).with_columns(
    pl.when(pl.col("country") == "PRT").then(pl.lit("Portogallo")).otherwise(pl.lit("Altri Paesi")).alias("country")
).group_by("country").agg(
    pl.len().alias("numero di prenotazioni"), pl.col("is_canceled").cast(pl.Float64).mean().alias("tasso di cancellazione"))
st.write( temp2 )

observed = pl.concat([temp1,temp2]).select(pl.col("numero di prenotazioni")).to_numpy().reshape(2,2)

chi ,p  = chi2(observed)
"""
Già senza ulteriori analisi si osserva che i 2 gruppi sono molto disomigenei, però volendo verificare ciò in maniera più rigorosa
possiamo utilizzare un test statistico per verificare l'assunzione di indipendenza (chi-quadro):
"""
st.write(f"La statistica test è pari a {chi}",f" con un P-value di {p}")
""" 
Oltre a questo vediamo anche che i tassi di cancellazione sono molto diversi tra loro 
con una quasi separazione netta tra clienti provenienti dal Portogallo e quelli provenienti da altri paesi.

Quindi si può concludere dicendo che c'è un ulteriore indizio a favore dell'ipotesi fatta in precedenza.

Ora analizzeremo se esistono altre variabili utili per aiutare l'hotel a prevedere la cancellazione della prenotazione 
e definire politiche mirate per ridurne il tasso.

Se un cliente è incline a fare molti cambiamenti alla prenotazione, tenderà a cancellarla più facilmente?
Se un cliente in passato ha cancellato prenotazioni, è più probabile che lo faccia di nuovo?
Se la prenotazione è di un cliente abituale, è più probabile che non venga cancellata?
Se un cliente fa richieste speciali, è più probabile che non cancelli la prenotazione?
Come può l'hotel ridurre le prenotazioni cancellate? Può essere utile l'utilizzo di cauzioni?
"""


# book changes
chart = bar_chart(data.with_columns(pl.when(pl.col("booking_changes")>1).then(pl.lit("2+")).otherwise("booking_changes").alias("booking_changes")), 
 "booking_changes","count()", "is_canceled", cat_color)

st.altair_chart(chart, use_container_width=True)
"""
Dal grafico si osserva che aumentando il numero di cambiamenti fatti, la prenotazione tende ad essere cancellata di meno, infatti abbiamo un rate di 
0.41, poi di 0.14 ed infine di 0.19.
Guardando questa associazione tra le 2 variabili si potrebbe ipotizzare che un cliente che tende a fare più cambiamenti
sia un cliente più attento o propenso a soggiornare e quindi sia meno incline a disdire.

Ora proviamo a verificare se c'è una qualche associazione tra il numero di richieste speciali e le cancellazioni.
"""

# total_of_special_requests
chart = bar_chart(data.with_columns(pl.when((pl.col("total_of_special_requests"))>1).then(pl.lit("2+")).otherwise("total_of_special_requests").alias("total_of_special_requests"))
    , "total_of_special_requests","count()","is_canceled", cat_color)
st.altair_chart(chart, use_container_width=True)
"""
Anche in questo caso, il grafico ci suggerisce che più richieste si associano a meno cancellazioni, infatti i clienti con 0 richieste speciali 
hanno un tasso di cancellazione del 48%. Questo rafforza l'ipotesi che un cliente 
più propenso a intervenire sulla prenotazione sia un cliente che cancellerà meno frequentemente.

Osserviamo ora se il comportamento dei clienti abituali possa essere un informazione utile agli hotel.
Si precisa che il numero di prenotazioni fatte da clienti abituali nel dataset è di soli 3497.
"""
# is_repeated_guest
chart = alt.Chart(data).mark_arc().encode(
    theta=alt.Theta("count()"),
    color=alt.Color(
        "is_canceled:N",
        scale=alt.Scale(scheme = cat_color),
        title="cancellazione"
    ),
    facet=alt.Facet(
        "is_repeated_guest:N",
        title="",
        header=alt.Header(
            titleOrient="bottom",
            labelOrient="bottom",
            labelExpr='datum.value ? "Cliente Abituale" : "Nuovo Cliente"'
        )
    )
).properties(
    title="Grafico a torta delle cancellazioni per clienti ripetuti e non"
)
st.altair_chart(chart, use_container_width=True)
"""
Come si poteva immaginare i clienti abituali cancellano meno rispetto ai nuovi clienti.

Ora ci soffermiamo sulle variabili previous_cancellations, previous_bookings_not_canceled che rappresentano il numero di cancellazioni
fatte in precedenza e il numero di prenotazioni non cancellate, questo dato non si riferisce allo stesso hotel ma a tutti gli hotel 
in un periodo precedente. 
Per l'analisi sono state divise le prenotazioni in 4 gruppi ossia:
- Clienti che non avevano ne cancellazioni ne prenotazioni passate (andate a buon fine).
- Clienti con solo cancellazioni passate.
- Clienti con solo prenotazioni senza cancellazioni.
- Clienti con sia cancellazioni passate che non, essendo poche osservazioni non è stato
  considerato quale delle 2 è maggiore.
"""
# preparation data
temp = data.filter(pl.col("is_canceled") == 1).with_columns(
    pl.when(pl.col("previous_cancellations") > 0).then(pl.lit("CANCELLAZIONI PASSATE")).otherwise(pl.lit("NO CANCELLAZIONI PASSATE")).alias("previous_cancellations"),
    pl.when(pl.col("previous_bookings_not_canceled") > 0).then(pl.lit("PRENOTAZIONI PASSATE")).otherwise(pl.lit("NO PRENOTAZIONI PASSATE")).alias("previous_bookings_not_canceled")
).group_by(
    pl.col("previous_cancellations"),
    pl.col("previous_bookings_not_canceled")
).agg(
    pl.len().alias("count"))
temp_total = data.with_columns(
    pl.when(pl.col("previous_cancellations") > 0).then(pl.lit("CANCELLAZIONI PASSATE")).otherwise(pl.lit("NO CANCELLAZIONI PASSATE")).alias("previous_cancellations"),
    pl.when(pl.col("previous_bookings_not_canceled") > 0).then(pl.lit("PRENOTAZIONI PASSATE")).otherwise(pl.lit("NO PRENOTAZIONI PASSATE")).alias("previous_bookings_not_canceled")
).group_by(
    pl.col("previous_cancellations"),
    pl.col("previous_bookings_not_canceled")
).agg(
    pl.len().alias("count"))

temp_join = temp_total.join(temp,left_on=["previous_cancellations","previous_bookings_not_canceled"],
    right_on=["previous_cancellations","previous_bookings_not_canceled"])
temp_join = temp_join.with_columns(
    (pl.col("count_right")/pl.col("count")).alias("rate")
)

temp_join.drop_in_place("count_right")

# Heatmap and text
chart = alt.Chart(temp_join).mark_rect().encode(
    x=alt.X('previous_bookings_not_canceled:N', title='Prenotazioni Passate Non Cancellate'),
    y=alt.Y('previous_cancellations:N', title='Cancellazioni Passate'),
    color=alt.Color('rate:Q', scale=alt.Scale(domainMax= 1, domainMin=0,
    domainMid=0.37,scheme=divergent_color), title='Tasso di cancellazione'),
    tooltip=['previous_cancellations', 'previous_bookings_not_canceled', alt.Tooltip('rate:Q', format='.2f'), 'count']
).properties(
    title='Tasso di Cancellazione per Comportamento Passato',
    height = 500
)

text = alt.Chart(temp_join).mark_text(size = 30).encode(
    x=alt.X('previous_bookings_not_canceled:N'),
    y=alt.Y('previous_cancellations:N'),
    text = alt.Text("count")
    )
"""
Qui sotto è rappresentata una heatmap dove il colore rappresenta il tasso di cancellazione di quel determinato gruppo di clienti
Con indicato sopra il numero di prenotazioni presenti nel dataset.
La scala dei colori è di tipo divergente con punto centrale il tasso di cancellazione di tutto il dataset (0.37).
"""
st.altair_chart(chart+text, use_container_width=True)
"""
I clienti che non avevano ne cancellazioni passate ne prenotazioni andate a buon fine sono il gruppo più numeroso
e hanno un tasso di cancellazione ovviamente simile a quello della totalità del dataset.
Osservando gli altri gruppi più piccoli e ricordando che correlazione non significa causalitàm, possiamo fare delle ipotesi utili, ad esempio si osserva
che i clienti con solo cancellazioni precedenti hanno un tasso estremamente alto pari a 0.84.
I clienti che invece hanno avuto prenotazioni passate andate a buon fine a prescindere dal fatto di aver cancellato o meno almeno una 
volta in precedenza, hanno tassi di cancellazione molto bassi pari a 0.14 (no canc. passate) e 0.19.

Questa informazione, insieme alle conclusioni trovate dai dati precedenti sui clienti abituali, 
suggerisce che un possibile approccio efficace da testare per ridurre le cancellazioni potrebbe essere quello di incentivare
il ritorno degli ospiti già soggiornati che almeno una volta non hanno annullato la prenotazione, attraverso promozioni mirate o 
altre strategie di fidelizzazione.

Inoltre potrebbe essere utile l'organizzazione di dataset in collaborazione con altri hotel
per tenere traccia dei clienti con tendenze a cancellare spesso.

Oltre a quanto detto un hotel potrebbe valutare come politica per la riduzione delle cancellazioni
l'inserimento o meno di una cauzione, rimborsabile o non.
Per valutare ciò osserviamo che informazioni può dare la variabile deposit_type:
"""
# deposity_type
count_chart = alt.Chart(data).mark_bar().encode(
    x=alt.X('deposit_type:N', title='Tipo di Cauzione'),
    y=alt.Y('count()', title='Numero di Prenotazioni'),
    color=alt.Color('is_canceled:O', title='Stato Prenotazione',
                    scale=alt.Scale(domain=[0,1], range=['#4C78A8','#F58518'])),
    tooltip=[
        alt.Tooltip('deposit_type:N', title='Tipo Cauzione'),
        alt.Tooltip('is_canceled:O', title='Cancellata?'),
        alt.Tooltip('count()', title='Conteggio')
    ]
).properties(
    title='Prenotazioni e Cancellazioni per Tipo di Cauzione',
    width=600
)

# 2) Tasso di cancellazione
rate_chart = alt.Chart(data).transform_aggregate(
    cancel_rate='mean(is_canceled)', 
    groupby=['deposit_type']
).mark_bar().encode(
    x=alt.X('deposit_type:N', title='Tipo di Cauzione'),
    y=alt.Y('cancel_rate:Q', title='Tasso di Cancellazione', axis=alt.Axis(format='%')),
    tooltip=[
        alt.Tooltip('deposit_type:N', title='Tipo Cauzione'),
        alt.Tooltip('cancel_rate:Q', title='Cancel Rate', format='.1%')
    ]
).properties(
    title='Tasso di Cancellazione per Tipo di Cauzione',
    width=600
)

st.subheader("Analisi di `deposit_type`")
st.altair_chart(count_chart & rate_chart, use_container_width=True)