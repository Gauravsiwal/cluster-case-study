
import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.cluster import KMeans

with open('model.pkl','rb') as file:
    model = pickle.load(file)

with open('scaler_instance.pkl','rb') as file:
    scaler = pickle.load(file)

with open('pca_instance.pkl','rb') as file:
    pca = pickle.load(file)


def prediction(input_data):

    scale_data = scaler.transform(input_data)
    pca_data = pca.transform(scale_data)
    pred = model.predict(pca_data)

    if pred==0:
        return 'Developed'
    elif pred==1:
        return 'Under Developed'
    else:
        return 'Developing'

def main():

    st.title('HELP FOUNDATION')
    st.subheader('This app give the status of the country based on socia-economic factors')
    ch_mort = st.text_input('Enter Child Mortality rate')
    exp = st.text_input('Enter Exports')
    imp = st.text_input('Enter Imports')
    hel = st.text_input('Enter Health.')
    inc = st.text_input('Enter Income')
    inf = st.text_input('Enter Inflation')
    exp = st.text_input('Enter Life expectancy')
    fer = st.text_input('Enter Total fertility.')
    gdp = st.text_input('Enter GDP')

    input_list = [[ch_mort,exp,hel,imp,inc,inf,exp,fer,gdp]]

    
    if st.button('Predict'):
        response = prediction(input_list)
        st.success(response)
    
if __name__ == '__main__':
    main()



