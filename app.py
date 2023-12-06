import streamlit as st
import sklearn
import numpy as np
import pickle
import re
import ast

st.title('Is it a Fraud!??')

loaded_model = pickle.load(open("SVM_MODEL.sav", 'rb'))
first_row = [
    [0.40964767, 0.85310975, 0.28816373, 0.8457041, 0.34371837, 0.5448024, 0.45495477, 0.70745996, 0.67525434,
     0.71146768,
     0.70209821, 0.45867999, 0.79363025, 0.66141385, 0.61468309, 0.63740079, 0.59131912, 0.67952185, 0.7051444,
     0.83884352,
     0.52924541, 0.40834939, 0.65421228, 0.61920848, 0.70841812, 0.61013333, 0.28867054, 0.78221927, 0.5615533,
     0.23260857]
]

labels = [
    "Normal", "Fraud"
]


def predict(array):
    prediction = loaded_model.predict(array)
    return labels[prediction[0]]


result = predict(first_row)
print(result)

array_string = st.text_area("Input some text here", first_row).strip()


def click_button():
    array = str2array(array_string)
    predicted_value = predict(array)
    st.write(f"Predicted Value: {predicted_value}")


def str2array(s):
    s = re.sub('\[ +', '[', s.strip())
    s = re.sub('[,\s]+', ', ', s)
    return np.array(ast.literal_eval(s))


st.button('Predict', on_click=click_button)
