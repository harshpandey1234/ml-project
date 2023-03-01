import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_hub as hub

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """

st.markdown(hide_streamlit_style, unsafe_allow_html = True)

st.title('Performance Prediction of Student')


def main():
  add_bg_from_url()
  g1_marks = st.number_input("G1 marks: ")
  g2_marks = st.number_input("G2 marks: ")
  absent = st.number_input("absent: ")
  failure = st.number_input("failure [0-3]: ")
  walc = st.number_input("walc: ")
  dalc = st.number_input("dalc: ")
  medu = st.number_input("medu [0-4]: ")
  goout = st.number_input("goout: ")
  fedu = st.number_input("fedu [0-4]: ")
  schoolsup = st.selectbox('Schhol sup:', ['yes','no'])
  romantic = st.selectbox('romantic:', ['yes','no'])
  fjob = st.selectbox("Father's Occupation: ", ['teacher', 'services', 'at_home', 'health', 'other'])
  mjob = st.selectbox("Mother's Occupation: ", ['teacher', 'services', 'at_home', 'health', 'other'])
  reason = st.selectbox("Reason: ", ['course', 'other', 'home', 'reputation'])
  paid = st.selectbox("Paid: ", ['yes', 'no'])

  #fmap = {'services':1, 'at_home':2, 'teacher':3, 'health':4, 'other':5}

  if fjob == 'services':
    fjob = 1
  elif fjob == 'at_home':
    fjob = 2
  elif fjob == 'teacher':
    fjob = 3
  elif fjob == 'health':
    fjob = 4
  elif fjob == 'other':
    fjob = 5

  if mjob == 'services':
    mjob = 1
  elif mjob == 'at_home':
    mjob = 2
  elif mjob == 'teacher':
    mjob = 3
  elif mjob == 'health':
    mjob = 4
  elif mjob == 'other':
    mjob = 5

  #paid
  if paid == 'yes':
    paid = 1
  else:
    paid = 0
  
  #reason
  if reason == 'course':
    reason = 1
  elif reason == 'home':
    reason = 2
  elif reason == 'reputation':
    reason = 3
  else:
    reason = 4

  #romantic
  if romantic == 'yes':
    romantic = 1
  else:
    romantic = 0

  #schoolsup
  if schoolsup == 'yes':
    schoolsup = 1
  else:
    schoolsup = 0


  input_features = {
    'absences':[absent],	
    'G2':[g2_marks],	
    'G1':[g1_marks],	
    'failures':[failure],	
    'Walc':[walc],	
    'schoolsup':[schoolsup],	
    'romantic':[romantic],	
    'Fjob':[fjob],	
    'Dalc':[dalc],	
    'paid':[paid],	
    'Mjob':[mjob],	
    'Medu':[medu],	
    'reason':[reason],	
    'goout':[goout],	
    'Fedu':[fedu], 
  }

  input_features = pd.DataFrame.from_dict(input_features)
  
  if st.button('Predict Marks'):
    pred_marks = predict_g3(input_features)
    st.success(f'The predicted mark is: {round(pred_marks, 3)}')


def predict_g3(input_features) :
  with st.spinner('Loading Model...'):
    regressor_model = tf.keras.models.load_model(r'model.h5', compile = False)
    g3_marks = regressor_model.predict(input_features)
    return g3_marks[0][0]

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
          
             background-image: url("https://images.unsplash.com/photo-1497864149936-d3163f0c0f4b?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1169&q=80");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )


if __name__ == '__main__' : main()
