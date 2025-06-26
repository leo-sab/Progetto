import streamlit as st
from preprocess import get_model
import altair as alt
import polars as pl
model, metrics, label_encoder = get_model()

st.title("Modello per Previsione")
"""
Utilizzando i dati mostrati nella pagina precedente, è stato addestrato un modello di classificazione con una Random Forest
per prevedere se una prenotazione sarà cancellata o meno.
Qui di seguito sono riportate le metriche del modello e un tool per visualizzare le previsioni su nuove prenotazioni.
"""
# MODEL METRICS 
""" ### Metriche test del modello
"""
classification_report = metrics["classification_report"]
weighted_avg = classification_report["weighted avg"]
 
"""
Le metriche qui riportate sono state calcolate su un campione (test set) pari al 20% del totale dei dati.
Per ulteriori informazioni guardare (?) affianco alla metrica.
""" 
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label=" Accuracy",
        value=f"{classification_report['accuracy']:.1%}",
        help=" (TP+FP) / (TP+FP+TN+FN) Proporzione di predizioni corrette sul totale delle predizioni"
    )

with col2:
    st.metric(
        label="Precision",
        value=f"{weighted_avg['precision']:.1%}",
        help="(TP/(TP+FP)) Proporzione di predizioni positive che erano effettivamente corrette "
    )

with col3:
    st.metric(
        label="Recall",
        value=f"{weighted_avg['recall']:.1%}",
        help=" (TP/(TP+FN)) Proporzione di casi positivi effettivi identificati correttamente "
    )

with col4:
    st.metric(
        label="F1-Score",
        value=f"{weighted_avg['f1-score']:.1%}",
        help="Media armonica tra precision e recall"
    )

with col5:
    st.metric(
        label="AUC Score",
        value=f"{metrics['auc_score']:.1%}",
        help="Area sotto la curva ROC, misura la capacità di distinguere tra le classi"
    )

# Metrics by class
"""
Il modello sembrerebbe avere delle buone prestazioni, se l'interesse è quello di prevedere il maggior numero di cancellazioni possibili,
allora è di interesse il recall di "Cancellato". 
Qui vengono riportate le metriche per ogni classe del target, in questo caso "Cancellato" e "Non Cancellato":
"""
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🟢 Non Cancellato")
    not_canceled = classification_report["Not Canceled"]
    
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric("Precision", f"{not_canceled['precision']:.1%}")
    with metric_col2:
        st.metric("Recall", f"{not_canceled['recall']:.1%}")
    with metric_col3:
        st.metric("F1-Score", f"{not_canceled['f1-score']:.1%}")
    
    st.info(f" Campione: {not_canceled['support']}")

with col2:
    st.markdown("### 🔴 Cancellato")
    canceled = classification_report["Canceled"]
    
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric("Precision", f"{canceled['precision']:.1%}")
    with metric_col2:
        st.metric("Recall", f"{canceled['recall']:.1%}")
    with metric_col3:
        st.metric("F1-Score", f"{canceled['f1-score']:.1%}")
    
    st.info(f" Campione: {canceled['support']}")

"""
Queste sono le metriche per classe, c'è da sottolineare il fatto che recall di "Cancellato" è abbastanza più basso
rispetto a quello di "Non Cancellato", questo potrebbe essere un punto di miglioramento per il modello se si è interessati
a prevedere il maggior numero di cancellazioni possibili.
"""

# ROC Curve
roc_data = metrics['roc_curve']
roc_df = pl.DataFrame({
    'FPR': roc_data['fpr'],
    'TPR': roc_data['tpr'],
    'Thresholds': roc_data['thresholds']
})
roc_chart = alt.Chart(roc_df).mark_line().encode(
    x=alt.X('FPR:Q', title='FPR'),
    y=alt.Y('TPR:Q', title='TPR'),
    tooltip=['FPR:Q', 'TPR:Q', 'Thresholds:Q']
).properties(
    title='Curva ROC',
    width=600,
    height=400
)

line_df = pl.DataFrame({
    'x': [0, 1],
    'y': [0, 1]
})

# Creazione del grafico per la linea di casualità
line_chart = alt.Chart(line_df).mark_line(color='gray', strokeDash=[3, 3]).encode(
    x='x:Q',
    y='y:Q'
)

st.altair_chart(roc_chart+line_chart, use_container_width=True)
"""
Qui sopra viene riportato la curva ROC, ossia la curva che mostra il trade-off tra il tasso di veri positivi (TPR) e
 il tasso di falsi positivi (FPR) per diversi valori di soglia di classificazione. 
 Maggiore è l'area sotto la curva (AUC = 96.2%), migliore è la capacità del modello di distinguere tra le classi.
 """
# Feature Importance
feature_importance = metrics['feature_importance']

# data for chart
feature_df = pl.DataFrame({
    "feature": list(feature_importance.keys()),
    "importance": list(feature_importance.values())
})

# create a bar chart for feature importance
feature_chart = alt.Chart(feature_df, title='Importanza delle Variabili'
    ).mark_bar().encode(
        x=alt.X('importance:Q', title='Importanza'),
        y=alt.Y('feature:N', title='Caratteristica', sort='-x'),
        color=alt.Color('importance:Q', scale=alt.Scale(scheme='blues')),
        tooltip=['feature:N', 'importance:Q']
    ).properties(title='Importanza delle Variabili')

st.altair_chart(feature_chart, use_container_width=True)
"""
Questa sezione illustra quanto ciascuna variabile influenzi le previsioni del modello.
Le variabili con un valore più elevato sono quelle che contribuiscono maggiormente al modello. 
Notiamo che molte delle variabili qui identificate come rilevanti erano già state evidenziate 
nella fase di esplorazione iniziale dei dati.


"""

confusion_matrix = metrics['confusion_matrix']
confusion_df = pl.DataFrame({
    'Reale': ["Non cancellato", "Non cancellato", "Cancellato", "Cancellato"],
    'Previsione': ["Non cancellato","Cancellato", "Non cancellato","Cancellato"],
    'Count': [
        confusion_matrix[0, 0], # True Negative
        confusion_matrix[0, 1], # False Positive
        confusion_matrix[1, 0], # False Negative
        confusion_matrix[1, 1]  # True Positive
    ],
    'Flag': ["Corretto",  "Errore", "Errore", "Corretto"] 
})


confusion_chart = alt.Chart(confusion_df).mark_rect().encode(
    x=alt.X('Previsione', title='Predetto'),
    y=alt.Y('Reale', title='Reale'),
    color=alt.Color('Flag:N', 
                   title='Tipo Predizione',
                   scale=alt.Scale(domain=["Corretto",  "Errore"], 
                        range=["#53D48B", "#AF233F"])),
    tooltip=['Previsione', 'Reale', 'Count:Q']
).properties(
    title='Matrice di Confusione',     height = 500
)
text = alt.Chart(confusion_df).mark_text(
    align='center',
    baseline='middle',
    fontSize=16,
    fontWeight='bold',
    color='black'
).encode(
    x='Previsione:N',
    y='Reale:N',
    text='Count:Q'
)
st.altair_chart(confusion_chart + text, use_container_width=True)
"""
La matrice di confusione mostra il numero di predizioni corrette e errate del modello.
Le celle verdi indicano le predizioni corrette, mentre quelle rosse indicano gli errori di classificazione.

La matrice di confusione mostra il numero di predizioni corrette e errate del modello.
Le celle verdi indicano le predizioni corrette, mentre quelle rosse indicano gli errori di classificazione.


### Cross Validation

Ora vediamo la cross-validation del modello, che ci permette di valutare le prestazioni del modello in maniera più
robusta.

"""
cv_metrics = metrics['cv_scores']
# Cross Validation Section


col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label="CV Accuracy",
        value=f"{cv_metrics['accuracy']:.1%}",
        help="Media dell'accuracy su tutti i fold della cross-validation"
    )

with col2:
    st.metric(
        label="CV Precision", 
        value=f"{cv_metrics['precision']:.1%}",
        help="Media della precision su tutti i fold della cross-validation"
    )

with col3:
    st.metric(
        label="CV Recall",
        value=f"{cv_metrics['recall']:.1%}",
        help="Media del recall su tutti i fold della cross-validation"
    )

with col4:
    st.metric(
        label="CV F1-Score",
        value=f"{cv_metrics['f1']:.1%}",
        help="Media dell'F1-score su tutti i fold della cross-validation"
    )

with col5:
    st.metric(
        label="CV AUC",
        value=f"{cv_metrics['roc_auc']:.1%}",
        help="Media dell'AUC su tutti i fold della cross-validation"
    )
"""
Le metriche sono state calcolate su 5 fold di cross-validation,
possiamo notare che quasi tutte le metriche sono molto simili a quelle del test set, tranne per recall che 
è leggermente più bassa.

###  Tool per Previsione Cancellazione Prenotazioni

InseriRE i dati di una nuova prenotazione:
"""

# Create prediction form
with st.form("prediction_form"):
    st.subheader("Inserisci i dati della prenotazione")
    
    # Create columns FOR layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Dati Generali**")
        lead_time = st.number_input("Lead Time (giorni)", min_value=0, max_value=737, value=50, 
                                   help="Giorni tra prenotazione e arrivo")
                
        arrival_date_week_number = st.number_input("Settimana dell'Anno", min_value=1, max_value=53, value=27)
        
        stays_in_weekend_nights = st.number_input("Notti Weekend", min_value=0, max_value=19, value=1)
        
        stays_in_week_nights = st.number_input("Notti Settimana", min_value=0, max_value=50, value=2)
        
        adults = st.number_input("Adulti", min_value=0, max_value=55, value=2)
        
        children = st.number_input("Bambini", min_value=0, max_value=10, value=0)
        
        babies = st.number_input("Neonati", min_value=0, max_value=10, value=0)
    
    with col2:
        st.markdown("**Dati Hotel**")
        hotel = st.selectbox("Tipo Hotel", list(label_encoder['hotel'].classes_))
        
        meal = st.selectbox("Tipo Pasto", list(label_encoder['meal'].classes_))
        
        country = st.selectbox("Paese", list(label_encoder['country'].classes_))
        
        market_segment = st.selectbox("Segmento Mercato", list(label_encoder['market_segment'].classes_))
        
        distribution_channel = st.selectbox("Canale Distribuzione", list(label_encoder['distribution_channel'].classes_))
        
        reserved_room_type = st.selectbox("Tipo Camera Prenotata", list(label_encoder['reserved_room_type'].classes_))
        
        assigned_room_type = st.selectbox("Tipo Camera Assegnata", list(label_encoder['assigned_room_type'].classes_))
    
    with col3:
        st.markdown("**Dati Prenotazione**")
        booking_changes = st.number_input("Modifiche Prenotazione", min_value=0, max_value=21, value=0)
        
        deposit_type = st.selectbox("Tipo Deposito", list(label_encoder['deposit_type'].classes_))
        
        days_in_waiting_list = st.number_input("Giorni in Lista d'Attesa", min_value=0, max_value=391, value=0)
        
        customer_type = st.selectbox("Tipo Cliente", list(label_encoder['customer_type'].classes_))
        
        adr = st.number_input("ADR", min_value=0.0, max_value=5400.0, value=100.0, step=0.5,
                             help="Average Daily Rate (Tariffa Media)")
        
        required_car_parking_spaces = st.number_input("Posti auto richiesti", min_value=0, max_value=8, value=0)
        
        total_of_special_requests = st.number_input("Richieste speciali", min_value=0, max_value=5, value=0)
        
        is_repeated_guest = st.selectbox("Cliente abituale", [0, 1], format_func = lambda x: "Sì" if x == 1 else "No")
        
        previous_cancellations = st.number_input("Cancellazioni precedenti", min_value=0, max_value= 50, value=0)
        
        previous_bookings_not_canceled = st.number_input("Prenotazioni precedenti non cancellate", min_value=0, max_value=50, value=0)
    
    # Submit button
    submitted = st.form_submit_button("🔍 Predici Cancellazione", type="primary")
    if submitted:
        # Prepare input data
        input_data = {
            'hotel': hotel,
            'lead_time': lead_time,
            'arrival_date_week_number': arrival_date_week_number,
            'stays_in_weekend_nights': stays_in_weekend_nights,
            'stays_in_week_nights': stays_in_week_nights,
            'adults': adults,
            'children': children,
            'babies': babies,
            'meal': meal,
            'country': country,
            'market_segment': market_segment,
            'distribution_channel': distribution_channel,
            'is_repeated_guest': is_repeated_guest,
            'previous_cancellations': previous_cancellations,
            'previous_bookings_not_canceled': previous_bookings_not_canceled,
            'reserved_room_type': reserved_room_type,
            'assigned_room_type': assigned_room_type,
            'booking_changes': booking_changes,
            'deposit_type': deposit_type,
            'days_in_waiting_list': days_in_waiting_list,
            'customer_type': customer_type,
            'adr': adr,
            'required_car_parking_spaces': required_car_parking_spaces,
            'total_of_special_requests': total_of_special_requests
        }
        
        # Create a DataFrame with the input
        input_df = pl.DataFrame([input_data])
        
        # CREATE THE MISSING FEATURE: same_room_type
        # This is the line that was missing!
        input_df = input_df.with_columns(
            pl.when(pl.col("reserved_room_type") == pl.col("assigned_room_type"))
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias("same_room_type")
        )
        
        # Apply label encoding for categorical variables
        for col in input_df.columns:
            if input_df[col].dtype == pl.Utf8:
                if col in label_encoder:

                    encoder = label_encoder[col]
                    original_value = input_df[col].item()
                    encoded_value = encoder.transform([original_value])[0]
                    
                    input_df = input_df.with_columns(
                        pl.lit(encoded_value).alias(col)
                    )
        

        # Convert to numpy array for prediction
        X_input = input_df.to_numpy()
        
        # Make prediction
        prediction = model.predict(X_input)[0]
        prediction_proba = model.predict_proba(X_input)[0]
        
        # Display results
        st.markdown("---")
        st.subheader(" Risultato Previsione")
        
        # Create columns for results
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.error("🔴 **PRENOTAZIONE A RISCHIO CANCELLAZIONE**")
                st.warning("La prenotazione ha un'alta probabilità di cancellazione")
            else:
                st.success("🟢 **PRENOTAZIONE PROBABILMENTE CONFERMATA**")
                st.info("La prenotazione ha una bassa probabilità di cancellazione")
        
        with col2:
            st.metric(
                label="Probabilità di Cancellazione", 
                value=f"{prediction_proba[1]:.1%}",
            )
            st.metric(
                label="Probabilità di Conferma", 
                value=f"{prediction_proba[0]:.1%}"
            )
        
        prob_df = pl.DataFrame({
            'Esito': ['Non Cancellato', 'Cancellato'],
            'Probabilità': [prediction_proba[0], prediction_proba[1]]
        })
        
        prob_chart = alt.Chart(prob_df).mark_bar().encode(
            x=alt.X('Probabilità:Q', scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format='%')),
            color=alt.Color('Esito:N', 
                           scale=alt.Scale(domain=['Non Cancellato', 'Cancellato'], 
                                         range=["#2DD475", "#C41B3D"]))
        ).properties(height=100)
        
        st.altair_chart(prob_chart, use_container_width=True)
        