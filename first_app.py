import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import pickle
import numpy as np
import pandas as pd

from lime import lime_text
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl','rb'))
# def predict(patent_text):

#     id_to_category = {0:'Not 705 or 706',1:'705',2:"706"}

#     final_features = tfidf.transform([patent_text])
#     prediction = model.predict(final_features)
#     predicted_class = id_to_category[prediction[0]]

#     return predicted_class

st.title('The Beatles or Taylor Swift') 
st.title('Streamlit Share And LIME Visualization')

txt = st.text_area('sjifagsorejas', 'jieoaj')
if st.button('Evaluate', key=None):
    c = make_pipeline(vectorizer, model)
    st.write(c.predict_proba(txt))
