import streamlit as st

st.markdown("""# Prédiction de la durée  vie de batteries""")
st.markdown('# ')
st.markdown('# ')

st.info("""### I/ La durée de vie des batteries : un enjeu majeur""")
col1, col2 = st.columns(2)
col1.write("Le monde fait face à une augmentation exponentielle des ventes de batteries")
col2.write("Graphic : coming soon")

st.write('')
st.write('')
st.write('')
st.write('')
st.warning('La durée de vie des équipements est un enjeu majeur pour les utilisateurs et les fabricants')
col1, col2, col3 = st.columns(3)
col1.write("Enjeu environmental")
col2.write("Enjeu économique")
col3.write("Enjeu de fiabilité")


st.markdown("""## Le problème posé""")
st.markdown(""" Prédire si """)

st.markdown("""Présentation """)

st.markdown("""## Problem 1 : Classification """)
st.markdown("""
=> Definition of the problem \n
=> Running a model, testing a set \n \n

Return \n
=> Prediction \n
=> Display True data and True results \n
=> Display if the model was valid or not
\n\n\n
""")


st.markdown("""## Problem 2 : Prediction of number of life cycles left """)
st.markdown("""
=> Definition of the problem \n
=> Running a model, testing a set \n \n

Return \n
=> Prediction \n
=> Display True data and True results \n
=> Display if the model was valid or not
\n \n \n
""")

st.markdown("""## Conclusion""")
st.markdown("""
=> The mlodème is working \n \n

Possible applications \n
=> Innovation\n
=> Operation of a battery park
\n \n \n
""")


st.markdown("""## Model limitations""")
st.markdown("""
=> In real world batteries don't go full charge/full discharge, \
    the model will have to be adapted for real-world constraints \n
=> The model has been tested on a single battery type/brand. \
    The performance will probably be lower on batteries of different type. \
    New tests should be proceed to adapt. However the model architechture might remain valid.
""")
