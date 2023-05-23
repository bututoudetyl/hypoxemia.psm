import streamlit as st

import numpy as np
from numpy import array
from numpy import argmax
from numpy import genfromtxt
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import pickle
import shap
import matplotlib.pyplot as plt
from xgboost.sklearn import XGBClassifier

st.set_page_config(page_title="Probability Prediction of Hypoxemia in Gastrointestinal Endoscopy under Sedation", layout="wide")

plt.style.use('default')

df=pd.read_csv('traindata1.csv',encoding='utf8')

trainy=df.Hypoxemia
trainx=df.drop('Hypoxemia',axis=1)

xgb = XGBClassifier(colsample_bytree=1,gamma=0.2,learning_rate=0.3,max_depth=3,
                    n_estimators =25,min_child_weight=1,objective= 'binary:logistic',
                    random_state = 10)

xgb.fit(trainx,trainy)

def user_input_features():
    st.title("Probability Prediction of Hypoxemia")
    st.sidebar.header('User input parameters below')
    a1=st.sidebar.number_input("Baseline SpO2(%)",min_value=75,max_value=100)
    a2=st.sidebar.number_input("BMI(kg/m2)",min_value=14.3,max_value=35.7,step=0.1)
    a3=st.sidebar.number_input("Neck circumference(cm)",min_value=27,max_value=46)
    a4=st.sidebar.number_input("Hemoglobin(g/L)",min_value=74,max_value=180)
    a5=st.sidebar.number_input("ASA",min_value=1,max_value=4)
    a6=st.sidebar.number_input("Induction propofol dose(mg)",min_value=20,max_value=180)
    a7=st.sidebar.number_input("Total propofol dose(mg)",min_value=30,max_value=500)
    a8=st.sidebar.selectbox('Diabetes',('No','Yes'))
    
    result=""
    if a8=="Yes":
        a8=1
    else: 
        a8=0 
    #if a11=="Yes":
        #a11=1
    output=[a1,a2,a3,a4,a5,a6,a7,a8]
    return output

outputdf = user_input_features()
outputdf = pd.DataFrame([outputdf], columns= trainx.columns)

   
p1 = xgb.predict(outputdf)[0]
p2 = xgb.predict_proba(outputdf)

p3 = p2[:,1]
result=""
if st.button("Predict"):
  #st.write(p2)  
  #st.write(f'The probability of hypoxemia during endscopies: {p3*100}')
  st.success('The probability of hypoxemia during endscopies: {:.2f}%'.format(p3[0]*100))
  #if p3 > 0.174:
      #b="High risk"
  #else:
      #b="Low risk"
  #st.success('The risk group:'+ b)
  
  explainer = shap.KernelExplainer(xgb.predict,trainx)
  shap_values = explainer.shap_values(outputdf)


  from shap.plots import _waterfall
#st_shap(shap.plots.waterfall(shap_values[0]),  height=500, width=1700)
  st.set_option('deprecation.showPyplotGlobalUse', False)
  _waterfall.waterfall_legacy(explainer.expected_value,shap_values[0,:],feature_names=trainx.columns)
#shap.summary_plot(shap_values,outputdf,feature_names=X.columns)
  st.pyplot(bbox_inches='tight')

#p3 = p2[:,1]
#result=""
#if st.button("Predict"):
  #st.write(p2)  
  #st.write(f'The probability of hypoxemia during endscopies: {p3*100}')
  #st.success('The probability of hypoxemia during endscopies: {:.2f}%'.format(p3[0]*100))
#p3 = p2[:,1]*100
    #st.success(p2)
    #st.success('The probability of hypoxemia during endscopies: {:.1f}%'.format(p2*100))
    #st.write('The Gastric Volume',round(p2,2))
    #st.success(p2.reshape(1))