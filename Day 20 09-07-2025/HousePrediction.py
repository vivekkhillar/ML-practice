import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def load_data():
    df =  pd.read_csv('housing_with_ocean_proximity.csv')
    return df

def train_model(df):
    X =  df.drop(columns= ['median_house_value'])
    y = df['median_house_value']
    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size =0.3, random_state=42)
    cat_col =  X_train.select_dtypes(include = 'object').columns
    num_col =  X_train.select_dtypes(exclude = 'object').columns
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
    model =Ridge(alpha = 0.1)
    final_pipeline =  Pipeline([
    ('pre', preprocessing),
    ('model_ridge', model)
    ])
    final_pipeline.fit(X_train, y_train)
    return final_pipeline, num_col, cat_col


df =  load_data()
final_pipeline, num_col, cat_col = train_model(df)


st.title('California House Value Predictor')
st.write('Input the features below to predict median house value')
with st.form('Prediction Form'):
    input_data = {}
    for col in num_col:
        value = st.number_input(f'{col}', value = float(df[col].median()))
        input_data[col]= value
    for col in cat_col:
        options = df[col].dropna().unique().tolist()
        value = st.selectbox(f'{col}', options)
        input_data[col]= value
    submitted =  st.form_submit_button('Predict House Value')

if submitted:
    input_df = pd.DataFrame([input_data])
    prediction = final_pipeline.predict(input_df)[0]
    st.success(f'Estimated Median House Value : **${prediction:.2f}**')
    