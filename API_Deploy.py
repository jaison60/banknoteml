#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import streamlit as st 
import numpy as np

pickle_in = open("BankNote.pickle","rb")
classifier=pickle.load(pickle_in)


def predict_note_authentication(variance,skewness,curtosis,entropy):
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
    print(prediction)
    return prediction

def main():
    st.title("Currency Detection")
    Glucose = st.text_input("variance")
    Bp = st.text_input("skewness")
    Insulin = st.text_input("curtosis")
    BMI = st.text_input("entropy")
    result=""
    if st.button("Predict"):
        result=predict_note_authentication(variance,skewness,curtosis,entropy)
    st.success('The output is {}'.format(result))
    

if __name__=='__main__':
    main()
    


# In[ ]:




