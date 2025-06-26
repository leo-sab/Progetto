import streamlit as st

vis = st.Page("app.py", title="Esplorazione Dati")
model = st.Page("app_model.py", title="Modello")
pag = st.navigation([vis, model])
pag.run()