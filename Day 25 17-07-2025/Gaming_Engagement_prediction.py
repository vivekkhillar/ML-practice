# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder,LabelEncoder, label_binarize
from sklearn.naive_bayes  import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
#from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, classification_report, confusion_matrix,ConfusionMatrixDisplay
from itertools import cycle
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os

def load_data():
    df = pd.read_csv('online_gaming_behavior_dataset.csv')
    df = df.drop(columns=['PlayerID'])
    return df

def train_model(df):
    X = df.iloc[:,:-1] # Apart from Engagement Level
    y = df.iloc[:,-1] # only Engagement Level
    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size =0.3, random_state=42)

    # Find the Categorical Column
    cat_col = X_train.select_dtypes(include ='object').columns
    num_col = X_train.select_dtypes(exclude ='object').columns

    # Do the imputaion for the numberical and categorical columns
    num_pipeline = Pipeline([
    ('impute_num', SimpleImputer(strategy = 'median')),
    ('std_scaler', StandardScaler())
    ])
    cat_pipeline = Pipeline([
    ('impute_cat', SimpleImputer(strategy = 'most_frequent')),
    ('ohe', OneHotEncoder())
    ])

    preprocessing = ColumnTransformer([
    ('num', num_pipeline, num_col), 
    ('cat', cat_pipeline, cat_col)
    ])

    model = LogisticRegression()
    final_pipeline =  Pipeline([
    ('pre', preprocessing),
    ('model_Logistic', model)
    ])

    final_pipeline.fit(X_train,y_train)
    y_pred_train = final_pipeline.predict(X_train)
    y_pred_test = final_pipeline.predict(X_test)
    
    train_accuracy =  accuracy_score(y_train, y_pred_train)
    test_accuracy =  accuracy_score(y_test, y_pred_test)

    print('_'*50)
    print('Logistic Regression')
    print(f'Training Accuracy : {train_accuracy * 100:.2f}%')
    print(f'Testing Accuracy : {test_accuracy * 100:.2f}%')
    
    ## Predict probabilities on test set
    return final_pipeline, num_col, cat_col

df =  load_data()
final_pipeline, num_col, cat_col = train_model(df)

st.title('Game Engagement Predictor')
st.write('Input the features below to predict Game Engagement value')
with st.form('Prediction Form'):
    input_data = {}
    for col in num_col:
        value = st.number_input(f'{col}', value = float(df[col].median()))
        input_data[col]= value
    for col in cat_col:
        options = df[col].dropna().unique().tolist()
        value = st.selectbox(f'{col}', options)
        input_data[col]= value
    submitted =  st.form_submit_button('Predict Game Engagement')

if submitted:
    input_df = pd.DataFrame([input_data])
    prediction = final_pipeline.predict(input_df)[0]
    st.success(f'Estimated Median Prediction : **${prediction}**')

