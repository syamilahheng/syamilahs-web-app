from sklearn.utils.validation import column_or_1d
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.header("Penguin Species Prediction") 

st.title("Penguin")


st.write("""
Hi ! It's Syamilah! Let's predict which penguin species is this. """)

st.write("""
Reference:
 """)


st.sidebar.header('User Input Penguin Features')

def user_input_features():
    culmen_length_mm = st.sidebar.slider('culmen_length_mm', 32.1, 59.6, 40.0)
    culmen_depth_mm = st.sidebar.slider('culmen_depth_mm', 13.1, 21.5, 15.0)
    flipper_length_mm = st.sidebar.slider('flipper_length_mm', 172, 231, 200)
    body_mass_g = st.sidebar.slider('body mass (g)', 2700, 6300, 3000)
    data = {'culmen_length_mm': culmen_length_mm,
            'culmen_depth_mm': culmen_depth_mm,
            'flipper_length_mm': flipper_length_mm,
            'body_mass_g': body_mass_g}
    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features()
df.dropna(0,'any')

st.subheader('User Input Penguin Features')
st.write(df)

penguins = pd.read_csv("penguins.csv")
X = penguins.drop(['Species'], axis = 1)
Y = penguins.Species

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Species Prediction')
st.write(penguins.Species[prediction])
# st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)
