import math
#import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#import sklearn.metrics
import streamlit as st
#from keras.callbacks import EarlyStopping
#from keras.layers import *
#from keras.models import Sequential
#from numpy.random import seed
#from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
from sklearn import metrics
#import io
#from io import BytesIO
#import tensorflow as tf
#from keras import backend as K
from lazypredict import LazyClassifier, LazyRegressor
# from lazypredict.Supervised import LazyRegressor, LazyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
import sweetviz
import codecs
#import streamlit.components.v1 as components
#from sklearn.feature_selection import VarianceThreshold
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from mordred import Calculator, descriptors
#from rdkit import Chem, DataStructs
#from rdkit.Chem import Descriptors, Draw, AllChem, MACCSkeys
#from rdkit.ML.Descriptors import MoleculeDescriptors
#import pubchempy as pcp
#from pubchempy import Compound
#from pubchempy import get_compounds
#import mols2grid
#from PIL import Image
#import urllib

# Page expands to full width
st.set_page_config(page_title='AIDrugApp', page_icon='🌐', layout="wide")
# For hiding streamlit messages
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_option('deprecation.showfileUploaderEncoding', False)

# Create title and subtitle
html_temp = """
        <div style="background-color:teal">
        <h1 style="font-family:arial;color:white;text-align:center;">AIDrugApp</h1>
        <h4 style="font-family:arial;color:white;text-align:center;">Artificial Intelligence Based Virtual Screening Web-App for Drug Discovery</h4>
        </div>
        <br>
        """
st.markdown(html_temp, unsafe_allow_html=True)

st.sidebar.title("AIDrugApp v1.2.5")

#st.sidebar.header("Custom ML Menu")

#add_selectbox = st.sidebar.radio("Select ML tool", ("Auto-Multi-ML", "Auto-DL", "DesCal", "Mol_Identifier"))

CB = st.sidebar.checkbox("Auto-Multi-ML")

if CB == "Auto-Multi-ML":
    st.title('Auto-Multi-ML')
    st.success(
        "This module of [**AIDrugApp v1.2.5**](https://aidrugapp.streamlit.app/) aids in the development and comparison of multiple machine learning models on user data in order to select the best performing machine learning algorithm. "
        " It also helps to predict target data based on user specific machine learning models.")

    expander_bar = st.expander("👉 More information")
    expander_bar.markdown("""
    * **Python libraries:** scikit-learn, streamlit, pandas, numpy, lazypredict, sweetviz
    * **Publication:** Divya Karade. (2021). Custom ML Module of AIDrugApp for Molecular Identification, Descriptor Calculation, and Building ML/DL QSAR Models. [ChemRxiv Preprint](https://doi.org/10.33774/chemrxiv-2021-3f1f9).
    """)

    expander_bar = st.expander("👉 How to use Auto-Multi-ML?")
    expander_bar.markdown("""
                **Step 1:** In the User input-side panel, select the type of algorithm ('Classification' or 'Regression') for building the multiple ML models.
                """)
    expander_bar.markdown("""
                **Step 2:** Upload descriptor data (included with target data) for building multiple ML models. (*Example input file provided*)
                """)
    expander_bar.markdown("""
                **Step 3:** For exploratory data analysis and/or interpreting and comparing multiple ML models, tick the checkboxes for 'EDA of uploaded data', 'EDA of feature engineered data', and 'Interpret multiple ML models'.
                """)
    expander_bar.markdown("""
                **Step 4:** After evaluating and understanding the outcomes of different ML models, select ML algorithm (ideally the top-performing ML model for the best results) for building an ML model on new data and predicting target data.
                """)
    expander_bar.markdown("""
                **Step 5:** Upload descriptor data (excluded with target data) based on feature-engineered model data for making target predictions by applying selected built ML mode. (*Example input file provided*)
                """)
    expander_bar.markdown("""
                **Step 6:** Click the "✨ PREDICT" button and the results will be displayed below to view and download.
                """)

    """---"""

    st.sidebar.header('⚙️ USER INPUT PANEL')
st.sidebar.write(
    '**1. Which Auto-Multi-ML algorithm would you like to select for predicting model performance?**')
add_selectbox = st.sidebar.radio(
    "Select your algorithm",
    ("Regression", "Classification"))

st.sidebar.write('**2. Upload data file for building multiple ML models**')
uploaded_file = st.sidebar.file_uploader("Upload input .csv file", type=["csv"])
st.sidebar.markdown("""[Example .csv input files](https://github.com/DivyaKarade/Example-.csv-input-files--AIDrugApp-v1.2)
                            """)
st.sidebar.write(
    '**3. Select checkbox for exploratory data analysis (EDA) and/or for interpreting and comparing multiple ML models**')
EDA = st.sidebar.checkbox("EDA of uploaded data")
EDA1 = st.sidebar.checkbox("EDA of feature engineered data")
DA = st.sidebar.checkbox("Interpret multiple ML models")

st.sidebar.write('**4. Select your ML model**')
choose_model = st.sidebar.selectbox("Select model",
                                    ("Random Forest Regressor", "Random Forest Classifier",
                                     "Logistic Regression",
                                     "Decision Tree", "KNeighborsClassifier", "SVC",
                                     "LinearDiscriminantAnalysis",
                                     "LinearRegression"))

st.sidebar.write("**5. Upload data file for predictions: **")
file_upload = st.sidebar.file_uploader("Upload .csv file", type=["csv"])
st.sidebar.markdown("""[Example .csv input batch file](https://github.com/DivyaKarade/Example-.csv-input-files--AIDrugApp-v1.2)
                                                                    """)
if file_upload is not None:
    data = pd.read_csv(file_upload)
    # data1 = data.dropna()
    features = data.iloc[:, 0:100]
    X = features

    st.info("**Uploaded data for prediction: **")
    st.write('Data Dimension: ' + str(data.shape[0]) + ' rows and ' + str(data.shape[1]) + ' columns.')
    st.write(data.style.highlight_max(axis=0))

else:
    st.info('Awaiting .csv file to be uploaded for making predictions')


def st_display_sweetviz(report_html, width=1000, height=500):
    report_file = codecs.open(report_html, 'r')
    page = report_file.read()
    components.html(page, width=width, height=height, scrolling=True)


if EDA:
    if uploaded_file is not None:
        @st.cache
        def load_data():
            data = pd.read_csv(uploaded_file)
            return data

        # Load the dataset
        data = load_data()
        st.markdown("**EDA of uploaded data**")
        st.write(data)
        st.write(data.describe())
        st.write('Dimension: ' + str(data.shape[0]) + ' rows and ' + str(data.shape[1]) + ' columns.')
        st.info('Data Shape: ' + str(data.shape))
        report = sweetviz.analyze([data, "Data"])
        report.show_html()
        st_display_sweetviz('SWEETVIZ_REPORT.html')

    else:
        st.info('Awaiting file to be uploaded.')

if EDA1:
    if uploaded_file is not None:
        @st.cache
        def load_data():
            data = pd.read_csv(uploaded_file)
            return data

        # Load the dataset
        data = load_data()
        data = data.dropna()
        st.markdown("**EDA of feature engineered data**")
        st.write(data)
        st.write(data.describe())
        st.write('Dimension: ' + str(data.shape[0]) + ' rows and ' + str(data.shape[1]) + ' columns.')
        st.info('Data Shape: ' + str(data.shape))
        report = sweetviz.analyze([data, "Data"])
        report.show_html()
        st_display_sweetviz('SWEETVIZ_REPORT.html')

    else:
        st.info('Awaiting file to be uploaded.')

if DA:
    if uploaded_file is not None:
        @st.cache
        def load_data():
            data = pd.read_csv(uploaded_file)
            return data

        # Load the dataset
        data = load_data()
        st.markdown("**Interpretation of multiple ML models**")

        if add_selectbox == "Auto-Multi-ML":
            if choose_model == 'Random Forest Regressor':
                st.write('Random Forest Regressor')
            if choose_model == 'Random Forest Classifier':
                st.write('Random Forest Classifier')
            if choose_model == 'Logistic Regression':
                st.write('Logistic Regression')
            if choose_model == 'Decision Tree':
                st.write('Decision Tree')
            if choose_model == 'KNeighborsClassifier':
                st.write('KNeighborsClassifier')
            if choose_model == 'SVC':
                st.write('Support Vector Classifier')
            if choose_model == 'LinearDiscriminantAnalysis':
                st.write('Linear Discriminant Analysis')
            if choose_model == 'LinearRegression':
                st.write('Linear Regression')

    else:
        st.info('Awaiting file to be uploaded.')

if choose_model == 'Logistic Regression':
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    classifier = LogisticRegression(max_iter=1000)  # Ensure scikit-learn is up-to-date
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    if st.sidebar.button("✨ PREDICT"):
        st.info('**Model Evaluation for Logistic Regression**')
        st.write('Model Accuracy Score:', metrics.accuracy_score(y_test, y_pred))
        st.write('Confusion Matrix:', metrics.confusion_matrix(y_test, y_pred))
        st.write('Precision:', metrics.precision_score(y_test, y_pred))
        st.write('ROC AUC:', metrics.roc_auc_score(y_test, y_pred))
        st.write('Recall:', metrics.recall_score(y_test, y_pred))
        st.write('F1 score:', metrics.f1_score(y_test, y_pred))

        st.info("**Find the Predicted Results below: **")

        data_2 = sc.fit_transform(data)
        predictions = classifier.predict(data_2)
        data['Target_value'] = predictions
        st.write(data)
        st.download_button('Download CSV', data.to_csv(), 'data.csv', 'text/csv')
        st.sidebar.warning('Prediction Created Successfully!')

else:
    st.info('Awaiting for uploaded dataset')
