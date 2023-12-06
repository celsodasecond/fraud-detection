import streamlit as st
import numpy as np
import sklearn
import pickle
import re
import ast

st.title('Is it a Fraud!??')

loaded_model = pickle.load(open("SVM_MODEL.sav", 'rb'))
first_row = [
    [0.97127286, 0.98438494, 0.28572327, 0.90040325, 0.23943409, 0.66830281, 0.44182885, 0.69029589, 0.6651602,
     0.88787761, 0.87921579, 0.25323654, 0.95231954, 0.45932044, 0.79104175, 0.4468321, 0.86138705, 0.8221349,
     0.82812259, 0.43824877, 0.32812773, 0.38812261, 0.66922031, 0.43903323, 0.87085091, 0.4653401, 0.54074136,
     0.75389233, 0.48611697, 0.0106317],

    [0.40964767, 0.85310975, 0.28816373, 0.8457041, 0.34371837, 0.5448024, 0.45495477, 0.70745996, 0.67525434,
     0.71146768, 0.70209821, 0.45867999, 0.79363025, 0.66141385, 0.61468309, 0.63740079, 0.59131912, 0.67952185,
     0.7051444, 0.83884352, 0.52924541, 0.40834939, 0.65421228, 0.61920848, 0.70841812, 0.61013333, 0.28867054,
     0.78221927, 0.5615533, 0.23260857],

    [0.97454038, 0.8608594, 0.26370778, 0.89027145, 0.26596577, 0.682023, 0.45731071, 0.77767511, 0.64042311,
     0.78446634, 0.94774627, 0.28485914, 0.87379123, 0.31430929, 0.87935346, 0.69949678, 0.81443757, 0.77490832,
     0.79341112, 0.54557497, 0.34066675, 0.37636183, 0.64853202, 0.48978419, 0.87114277, 0.44044633, 0.17530224,
     0.80827201, 0.46842562, 0.17887761],

    [0.17824082, 0.95051396, 0.26330732, 0.94274217, 0.09948252, 0.66267105, 0.50840654, 0.69108156, 0.66902358,
     0.91619092, 0.84069181, 0.26052707, 0.99227035, 0.67571038, 0.85248493, 1., 0.73350591, 0.83581248, 0.60191161,
     0.26906386, 0.34847759, 0.39098831, 0.68286303, 0.42557034, 0.25425361, 0.51151246, 0.40443655, 0.7651575,
     0.50343924, 0.04230118],

    [0.84239086, 0.98500628, 0.28734446, 0.92567467, 0.22974895, 0.6472706, 0.38660289, 0.69171929, 0.66221141,
     0.87768728, 0.89695894, 0.29140915, 0.9697239, 0.46257411, 0.86996375, 0.62550191, 0.84991046, 0.78484306,
     0.80025334, 0.38791323, 0.31582399, 0.38977495, 0.67706674, 0.45712791, 0.7954078, 0.42109577, 0.26863096,
     0.75903105, 0.48211278, 0.00281726],

    [0.97127286, 0.98438494, 0.28572327, 0.90040325, 0.23943409, 0.66830281, 0.44182885, 0.69029589, 0.6651602,
     0.88787761, 0.87921579, 0.25323654, 0.95231954, 0.45932044, 0.79104175, 0.4468321, 0.86138705, 0.8221349,
     0.82812259, 0.43824877, 0.32812773, 0.38812261, 0.66922031, 0.43903323, 0.87085091, 0.4653401, 0.54074136,
     0.75389233, 0.48611697, 0.0106317]
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
