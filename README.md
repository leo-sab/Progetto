# Hotel Booking Cancellation


## Obiettivo del progetto
Creare una dashboard per visualizzare l'analisi esplorativa del dataset, per approfondire le variabili presenti e fare possibili ipotesi dei motivi della cancellazione e di politiche per la prevenzione.
Creazione di un tool utilizzabile dai gestori di hotel per prevenire/prepararsi a possibili cancellazioni.
## Come avviare l'applicazione?
Con il commando: 
`uv run streamlit run home.py`

## Struttura del progetto
PROGETTO_SE2/
│
├── .gitignore                     
├── .python-version                
├── pyproject.toml                 
├── uv.lock                        
│
├── hotel_bookings.csv            # Dataset principale usato per analisi e training
├── ne_10m_admin_0_countries.zip  # Dati per mappe
├── map.html                      # Mappa 
│
├── home.py                       # Entry-point Streamlit dell’applicazione
├── app.py                        # Pagina EDA
├── app_model.py                  # Pagina Modello
│
├── trainmodel.py                 # Script di training del modello 
├── preprocess.py                 # Funzioni utilizzate in app.py e preprocessing del dataset
│
├── label_encoders_RF0.pkl        # Label encoder salvato per la trasformazione delle variabili categoriche
├── model_RF0_metrics.pkl         # Metriche salvate del modello 
├── random_forest_model_0.pkl     # Modello Random Forest 
│
├── README.md                     
└── __pycache__/                  

## Librerie utilizzate
 Streamlit, Polars, Geopandas, Altair, Scipy, Joblib, Scikit-learn

## Informazioni sul dataset
Il dataset si trova su Kaggle (https://www.kaggle.com/datasets/thedevastator/hotel-bookings-analysis) contiene le prenotazioni fatte per hotel portoghesi, ha circa 119 mila righe e 33 variabili.

| Variabile                        | Tipo         | Descrizione                                                                                                           |
|------------------------------|--------------|----------------------------------------------------------------------------------------------------------------------|
| index                        | Int64        | Numero identificativo univoco della prenotazione                                                                     |
| hotel                        | String, Utf8 | Tipo di hotel (Resort Hotel o City Hotel)                                                                            |
| is_canceled                  | Int64        | Indica se la prenotazione è stata cancellata (1) o no (0)                                                                   |
| lead_time                    | Int64        | Numero di giorni tra la prenotazione e la data di arrivo                                                             |
| arrival_date_year            | Int64        | Anno di arrivo previsto                                                                                                      |
| arrival_date_month           | String       | Mese di arrivo                                                                                                       |
| arrival_date_week_number     | Int64        | Numero della settimana dell’anno per l’arrivo                                                                        |
| arrival_date_day_of_month    | Int64        | Giorno del mese per l’arrivo                                                                                          |
| stays_in_weekend_nights      | Int64        | Numero di notti nel weekend (sabato o domenica) del soggiorno                                                        |
| stays_in_week_nights         | Int64        | Numero di notti infrasettimanali del soggiorno                                                                       |
| adults                       | Int64        | Numero di adulti                                                                                                     |
| children                     | Float64      | Numero di bambini                                                                                                    |
| babies                       | Int64        | Numero di neonati                                                                                                    |
| meal                         | String       | Tipo di pasto prenotato (BB - Bed & Breakfast, HB - Mezza Pensione, FB - Pensione Completa, SC - Nessun pasto)       |
| country                      | String       | Paese di origine del cliente                                                                                         |
| market_segment               | String       | Segmento di mercato                                                                                                  |
| distribution_channel         | String       | Canale di distribuzione della prenotazione (es. TA/TO, Direct, Corporate)                                           |
| is_repeated_guest            | Int64        | Se il cliente è un ospite ripetuto (1) o no (0)                                                                      |
| previous_cancellations       | Int64        | Numero di prenotazioni precedenti cancellate dal cliente                                                             |
| previous_bookings_not_canceled | Int64      | Numero di prenotazioni precedenti non cancellate dal cliente                                                         |
| reserved_room_type           | String       | Codice del tipo di camera prenotata                                                                                  |
| assigned_room_type           | String       | Codice del tipo di camera effettivamente assegnata                                                                   |
| booking_changes              | Int64        | Numero di modifiche fatte alla prenotazione                                                                          |
| deposit_type                 | String       | Tipo di deposito/cauzione (No Deposit, Non Refund, Refundable)                                                    |
| agent                        | Float64      | ID dell’agenzia di viaggio che ha effettuato la prenotazione                                                         |
| company                      | Float64      | ID dell’azienda/organizzazione che ha effettuato la prenotazione                                                     |
| days_in_waiting_list         | Int64        | Giorni in lista d’attesa prima della conferma                                                                        |
| customer_type                | String       | Tipo di cliente / accordo                                                                                              |
| adr                          | Float64      | Average Daily Rate (tariffa giornaliera media)                                                                       |
| required_car_parking_spaces  | Int64        | Numero di posti auto richiesti                                                                                       |
| total_of_special_requests    | Int64        | Numero totale di richieste speciali                                                                                  |
| reservation_status           | String       | Stato finale della prenotazione (Check-Out, Canceled, No-Show)                                                       |
| reservation_status_date      | String       | Data dell’ultimo stato della prenotazione                                                                            |


## Preprocessing 
Il preprocessing dei dati viene eseguito all'interno della funzione get_data(), che carica e pulisce il dataset hotel_bookings.csv
Le operazioni effettuate sono:

    - Rimozione colonne non rilevanti: le colonne agent e company vengono eliminate, in quanto scarsamente informative e con alta percentuale di valori mancanti (segnala solo l'id dell'ospite se la prenotazione è fatta da agenzie/aziende)

    - Gestione valori nulli: i valori "Undefined" e "NA" sono considerati nulli, si assume che la mancanza del dato non sia informativa

    - Correzione outlier e anomalie su adr: vengono rimossi i valori anomali, come adr negativi, superiori a 5000, o pari a 0 con market_segment diverso da "Complementary" ( soggiorni gratuiti/offerti )

    - Imputazione meal: si assume che i valori "Undefined" senza pacchetto pasto.

    Correzione errori di codifica: il codice paese "CN" viene corretto in "CAN" per la correttezza delle mappe.

    Creazione di una colonna arrival_date in formato Date: viene generata a partire dalle colonne arrival_date_year, arrival_date_month e arrival_date_day_of_month. 

 ## Scelta colori:

 ## Modello:
 Il modello utilizzato è un Random Forest Classifier.
 Il target da predire è la variabile binaria is_canceled.

Il preprocessing comprende:

    - Rimozione di colonne ridondanti o non informative (index, reservation_status, date di arrivo, ecc.).

    - Creazione e modifica di feature: same_room_type indica se la stanza assegnata è uguale a quella prenotata, per quanto riguarda la variabile country tutti i paesi con meno di 500 prenotazioni sono accorpati nella categoria "Other".

    - Label Encoding delle variabili categoriche tramite LabelEncoder (codifica numerica per semplicità, si segnala il rischio che l'inserimento di ordine nelle variabili senza un'ordine possa influire negativamente, si assume che l'impatto sia ridotto), mantenendo traccia degli encoder per poterli riutilizzare.

Addestramento:

    Il dataset è diviso in training e test set con un rapporto 80/20.
    Il modello è addestrato con 100 alberi e random_state=42 per la riproducibilità.
    Viene eseguita una cross-validation stratificata a 5 fold per stimare le metriche medie: accuracy, precision, recall, f1-score e ROC AUC.


Il modello, le metriche e la "mappatura" degli encoders sono salvati in file .pkl per velocizzare l'esecuzione dell'applicazione.