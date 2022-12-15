import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn.metrics
import streamlit as st
from keras.callbacks import EarlyStopping
from keras.layers import *
from keras.models import Sequential
from numpy.random import seed
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#from tensorflow import set_random_seed
from sklearn import metrics
import io
from io import BytesIO
import tensorflow as tf
from keras import backend as K
#from lazypredict.Supervised import LazyRegressor, LazyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
import sweetviz
import codecs
import streamlit.components.v1 as components
from sklearn.feature_selection import VarianceThreshold
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mordred import Calculator, descriptors
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Draw, AllChem, MACCSkeys
from rdkit.ML.Descriptors import MoleculeDescriptors
import pubchempy as pcp
from pubchempy import Compound
from pubchempy import get_compounds
import mols2grid
from PIL import Image
import urllib

# Page expands to full width
st.set_page_config(page_title='AIDrugApp', page_icon='🌐', layout="wide")
# For hiding streamlite messages
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

st.sidebar.title("AIDrugApp v1.2.4")

st.sidebar.header("Custom ML Menu")

add_selectbox = st.sidebar.radio("Select ML tool", ("Auto-Multi-ML", "Auto-DL", "DesCal", "Mol_Identifier"))

if add_selectbox == "Auto-Multi-ML":
    st.title('Auto-Multi-ML')
    st.success(
        "This module of **AIDrugApp v1.2.4** aids in the development and comparison of multiple machine learning models on user data in order to select the best performing machine learning algorithm. "
        " It also helps to predict target data based on user specific machine learning models.")

    expander_bar = st.expander("👉 More information")
    expander_bar.markdown("""
    * **Python libraries:** scikit-learn, streamlit, pandas, numpy, lazypredict, sweetviz
    * **Publication:** Divya Karade. (2021). Custom ML Module of AIDrugApp for Molecular Identification, Descriptor Calculation, and Building ML/DL QSAR Models. [ChemRxiv Preprint] (https://doi.org/10.33774/chemrxiv-2021-3f1f9).
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


    # Load dataset
    if uploaded_file is not None:
        data_1 = pd.read_csv(uploaded_file)
        # data_1 = data_1.loc[:10000]  # FOR TESTING PURPOSE, COMMENT THIS OUT FOR PRODUCTION
        X1 = data_1.iloc[:, :-1]  # Using all column except for the last column as X
        Y1 = data_1.iloc[:, -1]  # Selecting the last column as Y
        # labels = data_1['Activity_value']
        # features = data_1.iloc[:, 0:8]
        # X = features
        # y = np.ravel(labels)
        st.info("**Uploaded data for building multiple ML models: **")
        st.write('Data Dimension: ' + str(data_1.shape[0]) + ' rows and ' + str(data_1.shape[1]) + ' columns.')
        st.write(data_1.style.highlight_max(axis=0))

        if EDA:
            report = sweetviz.analyze([(data_1), "Uploaded data"], "Target")
            report.show_html()
            """---"""
            st.info('**Exploratory Data Analysis report for uploaded data**')
            st_display_sweetviz("SWEETVIZ_REPORT.html")
            components.html("SWEETVIZ_REPORT.html")

        selection = VarianceThreshold(threshold=(.8 * (1 - .8)))
        X = selection.fit_transform(X1)
        st.info("**Feature engineered data (from uploaded data for model building): **")
        df = pd.DataFrame(X)
        df0 = pd.DataFrame(Y1)
        frames1 = [df, df0]
        FE = pd.concat(frames1, axis=1)
        st.write('Data Dimension: ' + str(FE.shape[0]) + ' rows and ' + str(FE.shape[1]) + ' columns.')
        st.write(FE.style.highlight_max(axis=0))
        st.download_button('Download CSV', FE.to_csv(), 'FE.csv', 'text/csv')

        X = FE.iloc[:, :-1]  # Using all column except for the last column as X
        Y = FE.iloc[:, -1]

        # Data split
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, shuffle=True, random_state=42)

        st.write('**Training set**')
        df1 = pd.DataFrame(X_train)
        # data_1 = data_1.loc[:1000]  # FOR TESTING PURPOSE, COMMENT THIS OUT FOR PRODUCTION

        df2 = pd.DataFrame(y_train)
        # data_1 = data_1.loc[:1000]  # FOR TESTING PURPOSE, COMMENT THIS OUT FOR PRODUCTION

        frames1 = [df1, df2]
        Train = pd.concat(frames1, axis=1)
        st.write('Data Dimension: ' + str(Train.shape[0]) + ' rows and ' + str(Train.shape[1]) + ' columns.')
        st.write(Train)
        st.download_button('Download CSV', Train.to_csv(), 'Train.csv', 'text/csv')

        st.write('**Test set**')
        df3 = pd.DataFrame(X_test)
        df4 = pd.DataFrame(y_test)
        frames2 = [df3, df4]
        Test = pd.concat(frames2, axis=1)
        st.write('Data Dimension: ' + str(Test.shape[0]) + ' rows and ' + str(Test.shape[1]) + ' columns.')
        st.write(Test)
        st.download_button('Download CSV', Test.to_csv(), 'Test.csv', 'text/csv')

        if EDA1:
            report = sweetviz.compare([(Train), "Training Set"], [(Test), "Test Set"], "Target")
            report.show_html()
            """---"""
            st.info('**Exploratory Data Analysis report for feature engineered data**')
            st_display_sweetviz("SWEETVIZ_REPORT.html")
            components.html("SWEETVIZ_REPORT.html")

        if add_selectbox == 'Regression':
            if DA:
                # Defines and builds the LazyRegressor
                reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None, predictions=True)
                models_train, predictions_train = reg.fit(X_train, X_train, y_train, y_train)
                models_test, predictions_test = reg.fit(X_train, X_test, y_train, y_test)

                """---"""
                # Prints the model performance
                st.info("**Model performance: Training set**")
                st.write('**Model evaluation**')
                st.write('Data Dimension: ' + str(models_train.shape[0]) + ' rows and ' + str(
                    models_train.shape[1]) + ' columns.')
                st.write(models_train)
                st.download_button('Download CSV', models_train.to_csv(), 'models_train.csv', 'text/csv')

                st.write('**Predictions for training set**')
                st.write('Data Dimension: ' + str(predictions_train.shape[0]) + ' rows and ' + str(
                    predictions_train.shape[1]) + ' columns.')
                st.write(predictions_train)
                st.download_button('Download CSV', predictions_train.to_csv(), 'predictions_train.csv', 'text/csv')

                # Performance table of the test set (30% subset)
                st.info("**Model performance: Test set**")
                st.write('**Model evaluation**')
                st.write('Data Dimension: ' + str(models_test.shape[0]) + ' rows and ' + str(
                    models_test.shape[1]) + ' columns.')
                st.write(models_test)
                st.download_button('Download CSV', models_test.to_csv(), 'models_test.csv', 'text/csv')

                st.write('**Predictions for test set**')
                st.write('Data Dimension: ' + str(predictions_test.shape[0]) + ' rows and ' + str(
                    predictions_test.shape[1]) + ' columns.')
                st.write(predictions_test)
                st.download_button('Download CSV', predictions_test.to_csv(), 'predictions_test.csv', 'text/csv')

        if add_selectbox == 'Classification':
            if DA:
                # Defines and builds the lazyclassifier
                clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None, predictions=True)
                models_train, predictions_train = clf.fit(X_train, X_train, y_train, y_train)
                models_test, predictions_test = clf.fit(X_train, X_test, y_train, y_test)

                """---"""
                # Prints the model performance
                st.info("**Model performance: Training set**")
                st.write('**Model evaluation**')
                st.write('Data Dimension: ' + str(models_train.shape[0]) + ' rows and ' + str(
                    models_train.shape[1]) + ' columns.')
                st.write(models_train)
                st.download_button('Download CSV', models_train.to_csv(), 'models_train.csv', 'text/csv')

                st.write('**Predictions for training set**')
                st.write('Data Dimension: ' + str(predictions_train.shape[0]) + ' rows and ' + str(
                    predictions_train.shape[1]) + ' columns.')
                st.write(predictions_train)

                # Performance table of the test set (30% subset)
                st.info("**Model performance: Test set**")
                st.write('**Model evaluation**')
                st.write('Data Dimension: ' + str(models_test.shape[0]) + ' rows and ' + str(
                    models_test.shape[1]) + ' columns.')
                st.write(models_test)
                st.download_button('Download CSV', models_test.to_csv(), 'models_test.csv', 'text/csv')

                st.write('**Predictions for test set**')
                st.write('Data Dimension: ' + str(predictions_test.shape[0]) + ' rows and ' + str(
                    predictions_test.shape[1]) + ' columns.')
                st.write(predictions_test)
                st.download_button('Download CSV', predictions_test.to_csv(), 'predictions_test.csv', 'text/csv')

        if choose_model == 'Random Forest Regressor':
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

            regressor = RandomForestRegressor(n_estimators=20, random_state=0)
            regressor.fit(X_train, y_train)
            y_pred = regressor.predict(X_test)

            if st.sidebar.button("✨ PREDICT"):
                st.info('**Model Evaluation for Random Forest Regressor**')
                st.write('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
                st.write('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
                st.write('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
                st.write('Coefficient of determination ($R^2$):', metrics.r2_score(y_test, y_pred))
                st.info("**Find the Predicted Results below: **")

                data_2 = sc.fit_transform(data)
                predictions = regressor.predict(data_2)
                data['Target_value'] = predictions
                st.write(data)
                st.download_button('Download CSV', data.to_csv(), 'data.csv', 'text/csv')
                st.sidebar.warning('Prediction Created Sucessfully!')

        if choose_model == 'Random Forest Classifier':

            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

            classifier = RandomForestClassifier(n_estimators=20, random_state=0)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)

            if st.sidebar.button("✨ PREDICT"):
                st.info('**Model Evaluation for Random Forest Classifier**')
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
                st.sidebar.warning('Prediction Created Sucessfully!')

        if choose_model == 'Logistic Regression':
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

            classifier = LogisticRegression()
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
                st.sidebar.warning('Prediction Created Sucessfully!')

        if choose_model == 'Decision Tree':
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

            classifier = DecisionTreeClassifier()
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)

            if st.sidebar.button("✨ PREDICT"):
                st.info('**Model Evaluation for Decision Tree**')
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
                st.sidebar.warning('Prediction Created Sucessfully!')

        if choose_model == 'KNeighborsClassifier':
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

            KNN_model = KNeighborsClassifier()
            classifier = KNN_model
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)

            if st.sidebar.button("✨ PREDICT"):
                st.info('**Model Evaluation for KNeighborsClassifier**')
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

                st.sidebar.warning('Prediction Created Sucessfully!')

        if choose_model == 'LinearRegression':
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

            classifier = linear_model.LinearRegression()
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)

            if st.sidebar.button("✨ PREDICT"):
                st.info('**Model Evaluation for LinearRegression**')
                st.write('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
                st.write('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
                st.write('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
                st.write('Coefficient of determination ($R^2$):', metrics.r2_score(y_test, y_pred))

                st.info("**Find the Predicted Results below: **")

                data_2 = sc.fit_transform(data)
                predictions = classifier.predict(data_2)
                data['Target_value'] = predictions
                st.write(data)

                st.download_button('Download CSV', data.to_csv(), 'data.csv', 'text/csv')

                st.sidebar.warning('Prediction Created Sucessfully!')

        if choose_model == 'SVC':
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

            svc = svm.SVC()
            classifier = svc
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)

            if st.sidebar.button("✨ PREDICT"):
                st.info('**Model Evaluation for SVC**')
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

                st.sidebar.warning('Prediction Created Sucessfully!')

        if choose_model == 'LinearDiscriminantAnalysis':
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

            classifier = LinearDiscriminantAnalysis()
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)

            if st.sidebar.button("✨ PREDICT"):
                st.info('**Model Evaluation for LinearDiscriminantAnalysis**')
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

                st.sidebar.warning('Prediction Created Sucessfully!')

    else:
        st.info('Awaiting .csv file to be uploaded for building multiple ML models')

if add_selectbox == 'Auto-DL':
    st.title('Auto-DL')
    st.success(
        "This module of **AIDrugApp v1.2.4** helps to build best Deep Learning model on users data."
        "It also helps to predict target data using same deep learning algorithm.")

    st.warning(
        "AutoDL can be launched from here 👉 [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/divyakarade/autodl/main/AutoDL.py)")

    expander_bar = st.expander("👉 More information")
    expander_bar.markdown("""
        * **Python libraries:** Tensorflow, AutoKeras, scikit-learn, streamlit, pandas, numpy, matplotlib
        * **Publication:** 1. Divya Karade. (2021). Custom ML Module of AIDrugApp for Molecular Identification, Descriptor Calculation, and Building ML/DL QSAR Models. [ChemRxiv Preprint] (https://doi.org/10.33774/chemrxiv-2021-3f1f9).
        2. Divya Karade. (2021, March 23). AutoDL: Automated Deep Learning (Machine learning module of AIDrugApp - Artificial Intelligence Based Virtual Screening Web-App for Drug Discovery) (Version 1.0.0). [Zenodo] (http://doi.org/10.5281/zenodo.4630119)
        """)

    expander_bar = st.expander("👉 How to use Auto-DL?")
    expander_bar.markdown("""
                            **Step 1:** In the User input-side panel, select the type of algorithm ('Classification' or 'Regression') for building the DL model.
                            """)
    expander_bar.markdown("""
                            **Step 2:** Upload descriptor data (included with target data) for building DL model (*Example input file given*)
                            """)
    expander_bar.markdown("""
                            **Step 3:** For developing the model, specify parameters such as 'Train-Test split percent', 'random seed number', 'maximum trial number', and 'epochs number'.
                                """)
    expander_bar.markdown("""
                            **Step 4:** Upload descriptor data (excluded with target data) for making target predictions (*Example input file provided*)
                            """)
    expander_bar.markdown("""
                            **Step 5:** Click the "✨ PREDICT" button and the results will be displayed below to view and download
                            """)

    """---"""

if add_selectbox == 'DesCal':
    st.title('DesCal')
    st.success(
        "DesCal is a molecular descriptor calculator module of **AIDrugApp v1.2.4** that helps to generate various molecular 2-D descriptors and fingerprints "
        "on users data. "
        "It also helps to generate customised descriptors as selected by the user on their data.")

    expander_bar = st.expander("👉 More information")
    expander_bar.markdown("""
            * **Python libraries:** Tensorflow, Keras, scikit-learn, Streamlit, Pandas, Numpy, Mordred and RDKit
            * **Data source for calculating logS:** https://pubs.acs.org/doi/10.1021/ci034243x
            * **Data source for calculating mordred 2-D descriptors:** https://jcheminf.biomedcentral.com/articles/10.1186/s13321-018-0258-y
            * Custom descriptor logS calculates by using deep learning model (Training set: $R^2$ - 0.99, RMSE - 0.14 ; Test set: $R^2$ - 0.93, RMSE - 0.54) 
            * **Publications:** Divya Karade. (2021). Custom ML Module of AIDrugApp for Molecular Identification, Descriptor Calculation, and Building ML/DL QSAR Models. [ChemRxiv Preprint] (https://doi.org/10.33774/chemrxiv-2021-3f1f9).
            """)

    expander_bar = st.expander("👉 How to use DesCal?")
    expander_bar.markdown("""
                    **Step 1:** On the "User Input Panel" first select whether you would like to calculate molecular descriptors for a single molecule or upload a batch file for multiple molecules (Example input batch file given)
                    """)
    expander_bar.markdown("""
                    **Step 2:** Select type of molecular descriptor for calculation (Users can select molecular descriptors like logP, logS, Vabc or van der waals volume etc of their choice using 'Custom descriptors')
                    """)
    expander_bar.markdown("""
                    **Step 3:** Input canonical SMILES of your molecule in single string or batch file (Input batch .csv file should contain atleast one column of 'smiles' for calculating molecular descriptors)
                        """)
    expander_bar.markdown("""
                    **Step 4:** Click the "Calculate" button and the results will be displayed below to view and download.
                    """)

    """---"""


    class RDKit_2D:
        def __init__(self, smiles):
            self.mols = [Chem.MolFromSmiles(i) for i in smiles]
            self.smiles = smiles

        def compute_2Drdkit(self, name):
            rdkit_2d_desc = []
            calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
            header = calc.GetDescriptorNames()
            for i in range(len(self.mols)):
                ds = calc.CalcDescriptors(self.mols[i])
                rdkit_2d_desc.append(ds)
            df = pd.DataFrame(rdkit_2d_desc, columns=header)
            df.insert(loc=0, column='smiles', value=self.smiles)
            return df

        def mordred_compute(self):
            calc = Calculator(descriptors, ignore_3D=True)
            df1 = calc.pandas(self.mols)
            df1.insert(loc=0, column='smiles', value=self.smiles)
            return df1

        def compute_MACCS(self, name):
            MACCS_list = []
            header = ['bit' + str(i) for i in range(167)]
            for i in range(len(self.mols)):
                ds = list(MACCSkeys.GenMACCSKeys(self.mols[i]).ToBitString())
                MACCS_list.append(ds)
            df2 = pd.DataFrame(MACCS_list, columns=header)
            df2.insert(loc=0, column='smiles', value=self.smiles)
            return df2

        def mol2fp(self, mol, radius=1):
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius)
            array = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp, array)
            return array

        def compute_ECFP6(self, name):
            bit_headers = ['bit' + str(i) for i in range(2048)]
            arr = np.empty((0, 2048), int).astype(int)
            for i in self.mols:
                fp = self.mol2fp(i)
                arr = np.vstack((arr, fp))
            df_ecfp6 = pd.DataFrame(np.asarray(arr).astype(int), columns=bit_headers)
            df_ecfp6.insert(loc=0, column='smiles', value=self.smiles)
            return df_ecfp6

        def mol2fp2(self, mol, radius=2):
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius)
            array = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp, array)
            return array

        def compute_ECFP62(self, name):
            bit_headers = ['bit' + str(i) for i in range(2048)]
            arr = np.empty((0, 2048), int).astype(int)
            for i in self.mols:
                fp = self.mol2fp2(i)
                arr = np.vstack((arr, fp))
            df_ecfp6 = pd.DataFrame(np.asarray(arr).astype(int), columns=bit_headers)
            df_ecfp6.insert(loc=0, column='smiles', value=self.smiles)
            return df_ecfp6

        def mol2fp3(self, mol, radius=3):
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius)
            array = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp, array)
            return array

        def compute_ECFP63(self, name):
            bit_headers = ['bit' + str(i) for i in range(2048)]
            arr = np.empty((0, 2048), int).astype(int)
            for i in self.mols:
                fp = self.mol2fp3(i)
                arr = np.vstack((arr, fp))
            df_ecfp6 = pd.DataFrame(np.asarray(arr).astype(int), columns=bit_headers)
            df_ecfp6.insert(loc=0, column='smiles', value=self.smiles)
            return df_ecfp6


    # Sidebar
    # Collects user input features into dataframe
    st.sidebar.header('⚙️ USER INPUT PANEL')
    st.sidebar.subheader('1. Type of Input data')
    add_selectbox = st.sidebar.radio(
        "How would you like to generate descriptors?",
        ("Single molecule", "Multiple molecules (Batch)"))

    st.sidebar.subheader('2. Select Molecular Descriptor')
    # Select fingerprint
    choose_desc = st.sidebar.selectbox("Select descriptor type",
                                       ("RDKit_2D", "Mordred_2-D",
                                        "Fingerprints", "Custom descriptors"))
    if choose_desc == 'Fingerprints':
        fp_name = st.sidebar.radio("Which fingerprint would you like to select?",
                                   ("MACCS", "Morgan/Circular (radius 1)", "Morgan/Circular (radius 2)",
                                    "Morgan/Circular (radius 3)"))
    if choose_desc == 'Custom descriptors':
        Des1 = ['MolLogP', 'MolWt', 'NumHAcceptors', 'NumHDonors', 'NumRotatableBonds', 'RingCount', 'TPSA',
                'HeavyAtomCount']
        Des2 = ['nAromAtom', 'WPath', 'Vabc', 'nAcid', 'nBase']
        Des3 = ['logS', 'AromaticProportion', 'Density']
        CD1 = st.sidebar.multiselect('Select 2-D descriptors - List1', Des1, Des1)
        CD2 = st.sidebar.multiselect('List2', Des2, Des2)
        CD3 = st.sidebar.multiselect('List3', Des3, Des3)

        DA = st.sidebar.checkbox(
            "View logS model data and prediction evaluation (click before clicking 'Calculate' button)")

    if add_selectbox == 'Single molecule':

        # Get the feature input from the user
        st.sidebar.subheader('3. Enter Canonical SMILES')
        # def user_input_features():
        name_smiles = st.sidebar.text_input('Enter column name for SMILES', 'canonical_smiles')

        # Store a dictionary into a variable
        user_data = {'smiles': name_smiles}

        # Transform a data into a dataframe
        user_input = pd.DataFrame(user_data, index=[0])
        df3 = pd.concat([user_input['smiles']], axis=1)
        df3.to_csv('molecule.smi', sep='\t', header=False, index=False)
        st.write(df3)
        smiles = df3['smiles'].values

        if st.sidebar.button("✨ CALCULATE"):
            if choose_desc == 'RDKit_2D':
                ## Compute RDKit_2D Fingerprints and export a csv file.
                RDKit_descriptor = RDKit_2D(smiles)
                x1 = RDKit_descriptor.compute_2Drdkit(df3)
                st.subheader('molecular descriptors')
                st.write(x1)
                # st.dataframe(x1)
                st.download_button('Download CSV', x1.to_csv(), 'DesCal_RDKit_2D.csv', 'text/csv')

            if choose_desc == 'Mordred_2-D':
                ## Compute RDKit_2D Fingerprints and export a csv file.
                RDKit_descriptor = RDKit_2D(smiles)
                x2 = RDKit_descriptor.mordred_compute()
                st.subheader('molecular descriptors')
                st.write(x2)
                st.download_button('Download CSV', x2.to_csv(), 'DesCal_Mordred_2-D.csv', 'text/csv')
                # x3 = x2.swapaxes("index", "columns")
                # st.write(x3)

            if choose_desc == 'Fingerprints':
                if fp_name == 'MACCS':
                    ## Compute RDKit_2D Fingerprints and export a csv file.
                    RDKit_descriptor = RDKit_2D(smiles)
                    x3 = RDKit_descriptor.compute_MACCS(df3)
                    st.subheader('molecular fingerprints')
                    st.write(x3)
                    st.download_button('Download CSV', x3.to_csv(), 'DesCal_MACCS-FP.csv', 'text/csv')

                elif fp_name == 'Morgan/Circular (radius 1)':
                    ## Compute RDKit_2D Fingerprints and export a csv file.
                    RDKit_descriptor = RDKit_2D(smiles)
                    x4 = RDKit_descriptor.compute_ECFP6(df3)
                    st.subheader('molecular fingerprints')
                    st.write(x4)
                    st.download_button('Download CSV', x4.to_csv(), 'DesCal_Morgan/Circular1-FP.csv', 'text/csv')

                elif fp_name == 'Morgan/Circular (radius 2)':
                    ## Compute RDKit_2D Fingerprints and export a csv file.
                    RDKit_descriptor = RDKit_2D(smiles)
                    x4 = RDKit_descriptor.compute_ECFP62(df3)
                    st.subheader('molecular fingerprints')
                    st.write(x4)
                    st.download_button('Download CSV', x4.to_csv(), 'DesCal_Morgan/Circular2-FP.csv', 'text/csv')

                elif fp_name == 'Morgan/Circular (radius 3)':
                    ## Compute RDKit_2D Fingerprints and export a csv file.
                    RDKit_descriptor = RDKit_2D(smiles)
                    x4 = RDKit_descriptor.compute_ECFP63(df3)
                    st.subheader('molecular fingerprints')
                    st.write(x4)
                    st.download_button('Download CSV', x4.to_csv(), 'DesCal_Morgan/Circular3-FP.csv', 'text/csv')

            if choose_desc == 'Custom descriptors':
                RDKit_descriptor = RDKit_2D(smiles)
                x5 = RDKit_descriptor.compute_2Drdkit(df3)
                # Filtering data
                selected_des1 = x5[CD1]

                x6 = RDKit_descriptor.mordred_compute()
                # Filtering data
                selected_des2 = x6[CD2]

                # AromaticProportion
                AromaticAtoms = x6.nAromAtom
                HeavyAtomCount = x5.HeavyAtomCount
                AromaticProportion = AromaticAtoms / HeavyAtomCount
                desc_AromaticProportion = pd.DataFrame(AromaticProportion, columns=['AromaticProportion'])
                # st.dataframe(desc_AromaticProportion)

                # Density
                Mass = x5.MolWt
                Volume = x6.Vabc
                Density = Mass / Volume
                desc_Density = pd.DataFrame(Density, columns=['Density'])


                # for LogS

                def rmse(y_true, y_pred):
                    from keras import backend
                    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


                # coefficient of determination (R^2) for regression
                def r_square(y_true, y_pred):
                    from keras import backend as K
                    SS_res = K.sum(K.square(y_true - y_pred))
                    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
                    return 1 - SS_res / (SS_tot + K.epsilon())


                seed(0)
                set_random_seed(3)
                np.random.seed(1)

                session_conf = tf.ConfigProto(
                    intra_op_parallelism_threads=1,
                    inter_op_parallelism_threads=1)

                # Force Tensorflow to use a single thread
                sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)

                K.set_session(sess)

                # for LogS
                sol = pd.read_csv('delaney.csv')
                # print(sol.head)

                RDKit_descriptor = RDKit_2D(sol.SMILES)
                X1 = RDKit_descriptor.compute_2Drdkit(sol)
                X2 = X1.iloc[:, 1:]
                # print(X2)

                X3 = RDKit_descriptor.compute_MACCS(sol)
                X4 = X3.iloc[:, 1:]
                # print(X4)

                X5 = pd.concat([X2, X4], axis=1)
                # print(X5)

                Y = sol.iloc[:, 1]
                # print(Y)

                # Data split
                X_train, X_test, y_train, y_test = train_test_split(X5, Y, test_size=0.25, shuffle=True, random_state=4)

                sc = StandardScaler()
                X_train = sc.fit_transform(X_train)
                X_test = sc.transform(X_test)
                # print(X_train.shape, y_train.shape)
                # print(X_test.shape, y_test.shape)

                # identify outliers in the training dataset
                iso = IsolationForest(contamination=0.1, n_estimators=100, random_state=0, verbose=0)
                yhat = iso.fit_predict(X_train)
                yhat_1 = iso.fit_predict(X_test)

                # select all rows that are not outliers
                mask = yhat != -1
                X_train, y_train = X_train[mask, :], y_train[mask]

                mask = yhat_1 != -1
                X_test, y_test = X_test[mask, :], y_test[mask]

                # print(X_train.shape, y_train.shape)
                # print(X_test.shape, y_test.shape)

                # Define the model
                model = Sequential()
                model.add(Dense(100, input_dim=375, activation='relu'))
                model.add(Dense(100, activation='relu'))
                model.add(Dense(100, activation='relu'))
                model.add(Dense(1, activation='linear'))
                model.compile(loss='mean_squared_error', optimizer='adam',
                              metrics=['mae', 'MAPE', rmse, r_square])

                # enable early stopping based on mean_squared_error
                earlystopping = EarlyStopping(monitor='val_r_square', patience=13, verbose=1, mode='max')

                # Train the model
                result = model.fit(
                    X_train,
                    y_train,
                    epochs=250,
                    batch_size=50,
                    shuffle=True,
                    verbose=2,
                    validation_data=(X_test, y_test))

                # User input
                x9 = x5.iloc[:, 1:]
                # print(x9)

                RDKit_descriptor = RDKit_2D(smiles)
                x7 = RDKit_descriptor.compute_MACCS(df3)
                x8 = x7.iloc[:, 1:]
                # print(x8)

                x10 = pd.concat([x9, x8], axis=1)
                # print(x10)

                x11 = sc.fit_transform(x10)

                logS = model.predict(x11)
                desc_logS = pd.DataFrame(logS, columns=['logS'])
                # st.dataframe(desc_logS)

                selected_des3 = pd.concat([desc_AromaticProportion, desc_Density, desc_logS], axis=1)
                selected_des4 = selected_des3[CD3]

                selected_des = pd.concat([selected_des1, selected_des2, selected_des4], axis=1)
                st.subheader('molecular descriptor/s')
                selected_des.insert(loc=0, column='smiles', value=smiles)
                st.dataframe(selected_des)
                st.download_button('Download CSV', selected_des.to_csv(), 'Custom_des.csv', 'text/csv')

                # Make a prediction with the neural network
                y_pred = model.predict(X_test)
                x_pred = model.predict(X_train)
                # Model Data visualization
                if DA:
                    # Show statistics on data
                    st.subheader('Model data for logS: ')
                    # data1 = pd.read_csv("clinical_trial_model_derived.csv")
                    st.write(
                        'Data Dimension: ' + str(sol.shape[0]) + ' rows and ' + str(sol.shape[1]) + ' columns.')
                    st.dataframe(sol.style.highlight_max(axis=0))

                    # Descriptors
                    st.subheader('RDKit 2-D descriptors and MACCS fingerprints generated for model data: ')
                    Z = pd.concat([X5, Y], axis=1)
                    st.write(Z)

                    st.info("Training set model validated based on test set")
                    st.subheader('Train Test split of model data: ')

                    st.subheader('Updated training & test dataset after removal of outliers')
                    st.write("Training set:{}".format(X_train.shape))
                    st.write("Test set:{}".format(X_test.shape))

                    # Model summary
                    s = io.StringIO()
                    model.summary(print_fn=lambda x: s.write(x + '\n'))
                    model_summary = s.getvalue()
                    s.close()

                    st.subheader('Model summary')
                    st.write("Model: Sequential")
                    plt.text(0.3, 0.2, model_summary)
                    plt.setp(plt.gca(), frame_on=False, xticks=(), yticks=())
                    plt.grid(False)
                    st.pyplot()

                    st.header('logS Model prediction evaluation: ')

                    # -----------------------------------------------------------------------------
                    # print statistical figures of merit for training set
                    # -----------------------------------------------------------------------------
                    st.subheader('Trained_error_rate:')
                    st.write("\n")
                    st.write(
                        "Mean absolute error (MAE):      %f" % sklearn.metrics.mean_absolute_error(y_train,
                                                                                                   x_pred))
                    st.write(
                        "Mean squared error (MSE):       %f" % sklearn.metrics.mean_squared_error(y_train,
                                                                                                  x_pred))
                    st.write("Root mean squared error (RMSE): %f" % math.sqrt(
                        sklearn.metrics.mean_squared_error(y_train, x_pred)))
                    st.write(
                        "Coefficient of determination ($R^2$): %f" % sklearn.metrics.r2_score(y_train, x_pred))

                    # -----------------------------------------------------------------------------
                    # print statistical figures of merit for test set
                    # -----------------------------------------------------------------------------
                    st.subheader('Test_error_rate:')
                    st.write("\n")
                    st.write(
                        "Mean absolute error (MAE):      %f" % sklearn.metrics.mean_absolute_error(y_test,
                                                                                                   y_pred))
                    st.write(
                        "Mean squared error (MSE):       %f" % sklearn.metrics.mean_squared_error(y_test,
                                                                                                  y_pred))
                    st.write("Root mean squared error (RMSE): %f" % math.sqrt(
                        sklearn.metrics.mean_squared_error(y_test, y_pred)))
                    st.write(
                        "Coefficient of determination ($R^2$): %f" % sklearn.metrics.r2_score(y_test, y_pred))

                    # plot training curve for R^2 (beware of scale, starts very low negative)
                    st.subheader('Training curve for R^2')
                    fig, ax = plt.subplots(figsize=(8, 6))
                    plt.plot(result.history['val_r_square'])
                    plt.plot(result.history['r_square'])
                    plt.title('Model R^2')
                    plt.ylabel('R^2')
                    plt.xlabel('epoch')
                    plt.legend(['test', 'train'], loc='upper left')
                    st.pyplot()

                    # plot training curve for rmse
                    st.subheader('Training curve for RMSE')
                    fig, ax = plt.subplots(figsize=(8, 6))
                    plt.plot(result.history['rmse'])
                    plt.plot(result.history['val_rmse'])
                    plt.title('RMSE')
                    plt.ylabel('RMSE')
                    plt.xlabel('epoch')
                    plt.legend(['train', 'test'], loc='upper left')
                    st.pyplot()

                    # print the linear regression and display datapoints
                    regressor = LinearRegression()
                    regressor.fit(y_train.values.reshape(-1, 1), x_pred)
                    x_fit = regressor.predict(x_pred)

                    reg_intercept = round(regressor.intercept_[0], 4)
                    reg_coef = round(regressor.coef_.flatten()[0], 4)
                    reg_label = "y = " + str(reg_intercept) + "*x +" + str(reg_coef)

                    st.subheader(
                        'Linear regression displaying observed and predicted datapoints from training set')
                    plt.scatter(y_train, x_pred, color='blue', label='data')
                    plt.plot(x_pred, x_fit, color='red', linewidth=2, label='Linear regression\n' + reg_label)
                    plt.title('Linear Regression')
                    plt.legend()
                    plt.xlabel('Observed')
                    plt.ylabel('Predicted')
                    st.pyplot()

                    # print the linear regression and display datapoints
                    regressor = LinearRegression()
                    regressor.fit(y_test.values.reshape(-1, 1), y_pred)
                    y_fit = regressor.predict(y_pred)

                    reg_intercept = round(regressor.intercept_[0], 4)
                    reg_coef = round(regressor.coef_.flatten()[0], 4)
                    reg_label = "y = " + str(reg_intercept) + "*x +" + str(reg_coef)

                    st.subheader('Linear regression displaying observed and predicted datapoints from test set')
                    plt.scatter(y_test, y_pred, color='blue', label='data')
                    plt.plot(y_pred, y_fit, color='red', linewidth=2, label='Linear regression\n' + reg_label)
                    plt.title('Linear Regression')
                    plt.legend()
                    plt.xlabel('Observed')
                    plt.ylabel('Predicted')
                    st.pyplot()


    elif add_selectbox == 'Multiple molecules (Batch)':
        # Sidebar
        with st.sidebar.subheader('3. Upload your CSV data'):
            uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
            st.sidebar.markdown("""
        [Example CSV input file](https://github.com/DivyaKarade/Example-.csv-input-files--AIDrugApp-v1.2/blob/main/smiles.csv)
        """)

        if uploaded_file is not None:
            # Read CSV data
            @st.cache
            def load_csv():
                csv = pd.read_csv(uploaded_file)
                return csv


            df4 = load_csv()
            df5 = df4.iloc[:, 0:8]
            X = df5
            # Write CSV data
            df5.to_csv('molecule.smi', sep='\t', header=False, index=False)
            st.subheader('Uploaded data')
            st.write(df5)
            smiles = df5['smiles'].values

            if st.sidebar.button("✨ CALCULATE"):
                if choose_desc == 'RDKit_2D':
                    ## Compute RDKit_2D Fingerprints and export a csv file.
                    RDKit_descriptor = RDKit_2D(smiles)
                    x1 = RDKit_descriptor.compute_2Drdkit(df5)
                    st.subheader('molecular descriptors')
                    st.write(x1)
                    st.download_button('Download CSV', x1.to_csv(), 'DesCal_RDKit_2D.csv', 'text/csv')

                if choose_desc == 'Mordred_2-D':
                    ## Compute RDKit_2D Fingerprints and export a csv file.
                    RDKit_descriptor = RDKit_2D(smiles)
                    x2 = RDKit_descriptor.mordred_compute()
                    st.subheader('molecular descriptors')
                    st.write(x2)
                    st.download_button('Download CSV', x2.to_csv(), 'DesCal_Mordred_2-D.csv', 'text/csv')
                    # x3 = x2.swapaxes("index", "columns")
                    # st.write(x3)

                if choose_desc == 'Fingerprints':
                    if fp_name == 'MACCS':
                        ## Compute RDKit_2D Fingerprints and export a csv file.
                        RDKit_descriptor = RDKit_2D(smiles)
                        x3 = RDKit_descriptor.compute_MACCS(df5)
                        st.subheader('molecular fingerprints')
                        st.write(x3)
                        st.download_button('Download CSV', x3.to_csv(), 'DesCal_MACCS-FP.csv', 'text/csv')

                    elif fp_name == 'Morgan/Circular (radius 1)':
                        ## Compute RDKit_2D Fingerprints and export a csv file.
                        RDKit_descriptor = RDKit_2D(smiles)
                        x4 = RDKit_descriptor.compute_ECFP6(df5)
                        st.subheader('molecular fingerprints')
                        st.write(x4)
                        st.download_button('Download CSV', x4.to_csv(), 'DesCal_Morgan/Circular1-FP.csv', 'text/csv')

                    elif fp_name == 'Morgan/Circular (radius 2)':
                        ## Compute RDKit_2D Fingerprints and export a csv file.
                        RDKit_descriptor = RDKit_2D(smiles)
                        x4 = RDKit_descriptor.compute_ECFP62(df5)
                        st.subheader('molecular fingerprints')
                        st.write(x4)
                        st.download_button('Download CSV', x4.to_csv(), 'DesCal_Morgan/Circular2-FP.csv', 'text/csv')

                    elif fp_name == 'Morgan/Circular (radius 3)':
                        ## Compute RDKit_2D Fingerprints and export a csv file.
                        RDKit_descriptor = RDKit_2D(smiles)
                        x4 = RDKit_descriptor.compute_ECFP63(df5)
                        st.subheader('molecular fingerprints')
                        st.write(x4)
                        st.download_button('Download CSV', x4.to_csv(), 'DesCal_Morgan/Circular3-FP.csv', 'text/csv')

                if choose_desc == 'Custom descriptors':
                    RDKit_descriptor = RDKit_2D(smiles)
                    x5 = RDKit_descriptor.compute_2Drdkit(df5)
                    # Filtering data
                    selected_des1 = x5[CD1]

                    x6 = RDKit_descriptor.mordred_compute()
                    # Filtering data
                    selected_des2 = x6[CD2]

                    # AromaticProportion
                    AromaticAtoms = x6.nAromAtom
                    HeavyAtomCount = x5.HeavyAtomCount
                    AromaticProportion = AromaticAtoms / HeavyAtomCount
                    desc_AromaticProportion = pd.DataFrame(AromaticProportion, columns=['AromaticProportion'])
                    # st.dataframe(desc_AromaticProportion)

                    # Density
                    Mass = x5.MolWt
                    Volume = x6.Vabc
                    Density = Mass / Volume
                    desc_Density = pd.DataFrame(Density, columns=['Density'])


                    # for LogS

                    def rmse(y_true, y_pred):
                        from keras import backend
                        return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


                    # coefficient of determination (R^2) for regression
                    def r_square(y_true, y_pred):
                        from keras import backend as K
                        SS_res = K.sum(K.square(y_true - y_pred))
                        SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
                        return 1 - SS_res / (SS_tot + K.epsilon())


                    seed(0)
                    set_random_seed(3)
                    np.random.seed(1)

                    session_conf = tf.ConfigProto(
                        intra_op_parallelism_threads=1,
                        inter_op_parallelism_threads=1)

                    # Force Tensorflow to use a single thread
                    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)

                    K.set_session(sess)

                    # for LogS
                    sol = pd.read_csv('delaney.csv')
                    # print(sol.head)

                    RDKit_descriptor = RDKit_2D(sol.SMILES)
                    X1 = RDKit_descriptor.compute_2Drdkit(sol)
                    X2 = X1.iloc[:, 1:]
                    # print(X2)

                    X3 = RDKit_descriptor.compute_MACCS(sol)
                    X4 = X3.iloc[:, 1:]
                    # print(X4)

                    X5 = pd.concat([X2, X4], axis=1)
                    # print(X5)

                    Y = sol.iloc[:, 1]
                    # print(Y)

                    # Data split
                    X_train, X_test, y_train, y_test = train_test_split(X5, Y, test_size=0.25, shuffle=True,
                                                                        random_state=4)

                    sc = StandardScaler()
                    X_train = sc.fit_transform(X_train)
                    X_test = sc.transform(X_test)
                    # print(X_train.shape, y_train.shape)
                    # print(X_test.shape, y_test.shape)

                    # identify outliers in the training dataset
                    iso = IsolationForest(contamination=0.1, n_estimators=100, random_state=0, verbose=0)
                    yhat = iso.fit_predict(X_train)
                    yhat_1 = iso.fit_predict(X_test)

                    # select all rows that are not outliers
                    mask = yhat != -1
                    X_train, y_train = X_train[mask, :], y_train[mask]

                    mask = yhat_1 != -1
                    X_test, y_test = X_test[mask, :], y_test[mask]

                    # print(X_train.shape, y_train.shape)
                    # print(X_test.shape, y_test.shape)

                    # Define the model
                    model = Sequential()
                    model.add(Dense(100, input_dim=375, activation='relu'))
                    model.add(Dense(100, activation='relu'))
                    model.add(Dense(100, activation='relu'))
                    model.add(Dense(1, activation='linear'))
                    model.compile(loss='mean_squared_error', optimizer='adam',
                                  metrics=['mae', 'MAPE', rmse, r_square])

                    # enable early stopping based on mean_squared_error
                    earlystopping = EarlyStopping(monitor='val_r_square', patience=13, verbose=1, mode='max')

                    # Train the model
                    result = model.fit(
                        X_train,
                        y_train,
                        epochs=250,
                        batch_size=50,
                        shuffle=True,
                        verbose=2,
                        validation_data=(X_test, y_test))

                    # Make a prediction with the neural network
                    y_pred = model.predict(X_test)
                    x_pred = model.predict(X_train)

                    # User input
                    x9 = x5.iloc[:, 1:]
                    # print(x9)

                    RDKit_descriptor = RDKit_2D(smiles)
                    x7 = RDKit_descriptor.compute_MACCS(df5)
                    x8 = x7.iloc[:, 1:]
                    # print(x8)

                    x10 = pd.concat([x9, x8], axis=1)
                    # print(x10)

                    x11 = sc.fit_transform(x10)

                    logS = model.predict(x11)
                    desc_logS = pd.DataFrame(logS, columns=['logS'])
                    # st.dataframe(desc_logS)

                    selected_des3 = pd.concat([desc_AromaticProportion, desc_Density, desc_logS], axis=1)
                    selected_des4 = selected_des3[CD3]

                    selected_des = pd.concat([selected_des1, selected_des2, selected_des4], axis=1)
                    st.subheader('molecular descriptor/s')
                    selected_des.insert(loc=0, column='smiles', value=smiles)
                    st.dataframe(selected_des)
                    st.download_button('Download CSV', selected_des.to_csv(), 'Custom_des.csv', 'text/csv')

                    # Model Data visualization
                    if DA:
                        # Show statistics on data
                        st.subheader('Model data: ')
                        # data1 = pd.read_csv("clinical_trial_model_derived.csv")
                        st.write(
                            'Data Dimension: ' + str(sol.shape[0]) + ' rows and ' + str(sol.shape[1]) + ' columns.')
                        st.dataframe(sol.style.highlight_max(axis=0))

                        # Descriptors
                        st.subheader('RDKit 2-D descriptors and MACCS fingerprints generated for model data: ')
                        Z = pd.concat([X5, Y], axis=1)
                        st.write(Z)

                        st.info("Training set model validated based on test set")
                        st.subheader('Train Test split of model data: ')

                        st.subheader('Updated training & test dataset after removal of outliers')
                        st.write("Training set:{}".format(X_train.shape))
                        st.write("Test set:{}".format(X_test.shape))

                        # Model summary
                        s = io.StringIO()
                        model.summary(print_fn=lambda x: s.write(x + '\n'))
                        model_summary = s.getvalue()
                        s.close()

                        st.subheader('Model summary')
                        st.write("Model: Sequential")
                        plt.text(0.3, 0.2, model_summary)
                        plt.setp(plt.gca(), frame_on=False, xticks=(), yticks=())
                        plt.grid(False)
                        st.pyplot()

                        st.header('logS Model prediction evaluation: ')

                        # -----------------------------------------------------------------------------
                        # print statistical figures of merit for training set
                        # -----------------------------------------------------------------------------
                        st.subheader('Trained_error_rate:')
                        st.write("\n")
                        st.write(
                            "Mean absolute error (MAE):      %f" % sklearn.metrics.mean_absolute_error(y_train,
                                                                                                       x_pred))
                        st.write(
                            "Mean squared error (MSE):       %f" % sklearn.metrics.mean_squared_error(y_train,
                                                                                                      x_pred))
                        st.write("Root mean squared error (RMSE): %f" % math.sqrt(
                            sklearn.metrics.mean_squared_error(y_train, x_pred)))
                        st.write(
                            "Coefficient of determination ($R^2$): %f" % sklearn.metrics.r2_score(y_train, x_pred))

                        # -----------------------------------------------------------------------------
                        # print statistical figures of merit for test set
                        # -----------------------------------------------------------------------------
                        st.subheader('Test_error_rate:')
                        st.write("\n")
                        st.write(
                            "Mean absolute error (MAE):      %f" % sklearn.metrics.mean_absolute_error(y_test,
                                                                                                       y_pred))
                        st.write(
                            "Mean squared error (MSE):       %f" % sklearn.metrics.mean_squared_error(y_test,
                                                                                                      y_pred))
                        st.write("Root mean squared error (RMSE): %f" % math.sqrt(
                            sklearn.metrics.mean_squared_error(y_test, y_pred)))
                        st.write(
                            "Coefficient of determination ($R^2$): %f" % sklearn.metrics.r2_score(y_test, y_pred))

                        # plot training curve for R^2 (beware of scale, starts very low negative)
                        st.subheader('Training curve for R^2')
                        fig, ax = plt.subplots(figsize=(8, 6))
                        plt.plot(result.history['val_r_square'])
                        plt.plot(result.history['r_square'])
                        plt.title('Model R^2')
                        plt.ylabel('R^2')
                        plt.xlabel('epoch')
                        plt.legend(['test', 'train'], loc='upper left')
                        st.pyplot()

                        # plot training curve for rmse
                        st.subheader('Training curve for RMSE')
                        fig, ax = plt.subplots(figsize=(8, 6))
                        plt.plot(result.history['rmse'])
                        plt.plot(result.history['val_rmse'])
                        plt.title('RMSE')
                        plt.ylabel('RMSE')
                        plt.xlabel('epoch')
                        plt.legend(['train', 'test'], loc='upper left')
                        st.pyplot()

                        # print the linear regression and display datapoints
                        regressor = LinearRegression()
                        regressor.fit(y_train.values.reshape(-1, 1), x_pred)
                        x_fit = regressor.predict(x_pred)

                        reg_intercept = round(regressor.intercept_[0], 4)
                        reg_coef = round(regressor.coef_.flatten()[0], 4)
                        reg_label = "y = " + str(reg_intercept) + "*x +" + str(reg_coef)

                        st.subheader(
                            'Linear regression displaying observed and predicted datapoints from training set')
                        plt.scatter(y_train, x_pred, color='blue', label='data')
                        plt.plot(x_pred, x_fit, color='red', linewidth=2, label='Linear regression\n' + reg_label)
                        plt.title('Linear Regression')
                        plt.legend()
                        plt.xlabel('Observed')
                        plt.ylabel('Predicted')
                        st.pyplot()

                        # print the linear regression and display datapoints
                        regressor = LinearRegression()
                        regressor.fit(y_test.values.reshape(-1, 1), y_pred)
                        y_fit = regressor.predict(y_pred)

                        reg_intercept = round(regressor.intercept_[0], 4)
                        reg_coef = round(regressor.coef_.flatten()[0], 4)
                        reg_label = "y = " + str(reg_intercept) + "*x +" + str(reg_coef)

                        st.subheader('Linear regression displaying observed and predicted datapoints from test set')
                        plt.scatter(y_test, y_pred, color='blue', label='data')
                        plt.plot(y_pred, y_fit, color='red', linewidth=2, label='Linear regression\n' + reg_label)
                        plt.title('Linear Regression')
                        plt.legend()
                        plt.xlabel('Observed')
                        plt.ylabel('Predicted')
                        st.pyplot()

        else:
            st.info('Awaiting for CSV file to be uploaded.')

if add_selectbox == 'Mol_Identifier':
    st.title('Mol_Identifier')
    st.success(
        "Mol_Identifier is a module of **AIDrugApp v1.2.4** that aids in molecule identification by converting chemical names to SMILES, "
        "identifying compound names and their 2-D structures from SMILE strings and detecting molecular similarities from users data.")

    expander_bar = st.expander("👉 More information")
    expander_bar.markdown("""
                * **Python libraries:** pubchempy, pandas, rdkit, mols2grid, matplotlib
                * **Publications:** Divya Karade. (2021). Custom ML Module of AIDrugApp for Molecular Identification, Descriptor Calculation, and Building ML/DL QSAR Models. [ChemRxiv Preprint] (https://doi.org/10.33774/chemrxiv-2021-3f1f9).
                """)

    expander_bar = st.expander("👉 How to use Mol_Identifier?")
    expander_bar.markdown("""
                    **Step 1:** On the "User Input Panel" first select whether you would like to search for a single molecule or upload a batch file for multiple molecules (Example input batch file given)
                    """)
    expander_bar.markdown("""
                    **Step 2:** Select type of data to identify
                    """)
    expander_bar.markdown("""
                    **Step 3:** Input molecular names or SMILES as per requirement in single string or batch file
                        """)
    expander_bar.markdown("""
                    **Step 4:** Click on the "button" and the results will be displayed below to copy or download
                    """)

    # Sidebar
    # Collects user input features into dataframe
    st.sidebar.header('⚙️ USER INPUT PANEL')
    st.sidebar.subheader('1. Type of Input data')
    add_selectbox1 = st.sidebar.radio(
        "How would you like to search?",
        ("Single molecule", "Multiple molecules (Batch)"))

    st.sidebar.subheader('2. Type of retrieving data')
    add_selectbox2 = st.sidebar.radio(
        "What would you like to identify?",
        ("Name to SMILE", "SMILE to Compound", "Molecular Similarity", "2-D Structure"))

    if add_selectbox2 == "Name to SMILE":
        add_selectbox3 = st.sidebar.selectbox("Select type of SMILES to retrieve",
                                              ("Canonical SMILES", "Isomeric SMILES"))

    if add_selectbox1 == 'Single molecule':
        if add_selectbox2 == "Name to SMILE":
            # Get the feature input from the user
            st.sidebar.subheader('3. Enter compound name')
            # def user_input_features():
            mol_name = st.sidebar.text_input('Enter compound name to retrieve SMILES', 'compound name')

            # Store a dictionary into a variable
            user_data = {'Name': mol_name}

            # Transform a data into a dataframe
            user_input = pd.DataFrame(user_data, index=[0])
            df = pd.concat([user_input['Name']], axis=1)
            df.to_csv('molecule.txt', sep='\t', header=False, index=False)
            st.write(df)
            List_of_Chemicals = df['Name'].values

            if st.sidebar.button("😊 GET SMILES"):
                if add_selectbox3 == "Canonical SMILES":
                    # list of chemical names
                    for chemical_name in List_of_Chemicals:
                        cid = pcp.get_cids(chemical_name)
                        prop2 = pcp.get_compounds(chemical_name, 'name')
                        for compound in prop2:
                            x = compound.canonical_smiles
                            x1 = (chemical_name + ' ' + str(x))
                            st.subheader('Canonical SMILES')
                            st.write(x1)

                if add_selectbox3 == "Isomeric SMILES":
                    # list of chemical names
                    for chemical_name in List_of_Chemicals:
                        cid = pcp.get_cids(chemical_name)
                        prop2 = pcp.get_compounds(chemical_name, 'name')
                        for compound in prop2:
                            x = compound.isomeric_smiles
                            x1 = (chemical_name + ' ' + str(x))
                            st.subheader('Isomeric SMILES')
                            st.write(x1)

        if add_selectbox2 == "SMILE to Compound":
            # Get the feature input from the user
            st.sidebar.subheader('3. Enter Canonical SMILES')
            # def user_input_features():
            smiles = st.sidebar.text_input('Enter SMILES to retrieve compound info', 'Canonical SMILES')

            # Store a dictionary into a variable
            user_data = {'smiles': smiles}

            # Transform a data into a dataframe
            user_input = pd.DataFrame(user_data, index=[0])
            df = pd.concat([user_input['smiles']], axis=1)
            df.to_csv('molecule.smi', sep='\t', header=False, index=False)
            st.write(df)
            smile = df['smiles'].values

            if st.sidebar.button("😊 GET COMPOUND"):
                # list of chemical names
                for smiles in smile:
                    prop2 = pcp.get_compounds(smiles, 'smiles')
                    comp = (smiles + ' ' + str(prop2))
                    st.subheader('PubChem Compound ID')
                    st.write(comp)
                    for compound in prop2:
                        df4 = pcp.compounds_to_frame(prop2,
                                                     properties=['synonyms', 'canonical_smiles', 'molecular_formula',
                                                                 'iupac_name', 'inchi', 'inchikey'])
                        st.subheader('Compound Info')
                        st.write(df4)
                        st.download_button('Download CSV', df4.to_csv(), 'Cmpd_info.csv', 'text/csv')

        if add_selectbox2 == "Molecular Similarity":
            # Get the feature input from the user
            st.sidebar.subheader('3. Enter SMILES to get molecular similarity')
            # def user_input_features():
            smiles1 = st.sidebar.text_input('Enter SMILES1', 'Canonical SMILES1')
            smiles2 = st.sidebar.text_input('Enter SMILES2', 'Canonical SMILES2')

            # Store a dictionary into a variable
            user_data1 = {'smiles1': smiles1}
            user_data2 = {'smiles2': smiles2}

            # Transform a data into a dataframe
            user_input1 = pd.DataFrame(user_data1, index=[0])
            user_input2 = pd.DataFrame(user_data2, index=[0])
            df = pd.concat([user_input1['smiles1'], user_input2['smiles2']], axis=1)
            # df2 = pd.concat([df['smiles']], axis=1)
            df.to_csv('molecule.smi', sep='\t', header=False, index=False)
            st.write(df)
            # s1 = df['smiles1'].values
            # s2 = df['smiles2'].values

            if st.sidebar.button("😊 GET SIMILARITY"):
                # list
                st.subheader('Similarity Scores')

                mol1 = Chem.MolFromSmiles(smiles1)
                mol2 = Chem.MolFromSmiles(smiles2)
                fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 3, nBits=2048)
                fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 3, nBits=2048)

                st.write("Tanimoto    :", round(DataStructs.TanimotoSimilarity(fp1, fp2), 4))
                st.write("Dice        :", round(DataStructs.DiceSimilarity(fp1, fp2), 4))
                st.write("Cosine      :", round(DataStructs.CosineSimilarity(fp1, fp2), 4))
                st.write("Sokal       :", round(DataStructs.SokalSimilarity(fp1, fp2), 4))
                st.write("McConnaughey:", round(DataStructs.McConnaugheySimilarity(fp1, fp2), 4))

        if add_selectbox2 == "2-D Structure":
            # Get the feature input from the user
            st.sidebar.subheader('3. Enter Canonical SMILES')
            # def user_input_features():
            smiles = st.sidebar.text_input('Enter SMILES to get 2-D structure', 'Canonical SMILES')

            # Store a dictionary into a variable
            user_data = {'smiles': smiles}

            # Transform a data into a dataframe
            user_input = pd.DataFrame(user_data, index=[0])
            df = pd.concat([user_input['smiles']], axis=1)
            df.to_csv('molecule.smi', sep='\t', header=False, index=False)
            st.write(df)
            smile = df['smiles'].values

            if st.sidebar.button("😊 GET STRUCTURE"):
                st.subheader('2-D structure')
                mol = Chem.MolFromSmiles(smiles)
                # x = st.image(Draw.MolToImage(mol))
                img = (Draw.MolToImage(mol))
                bio = BytesIO()
                img.save(bio, format='png')
                st.image(img)

    if add_selectbox1 == 'Multiple molecules (Batch)':
        if add_selectbox2 == "Name to SMILE":
            # Sidebar
            with st.sidebar.subheader('3. Upload your CSV data'):
                uploaded_file = st.sidebar.file_uploader(
                    "Upload your input CSV file containing 'Name' as one column of compound names", type=["csv"])
                st.sidebar.markdown("""
            [Example CSV input file](https://github.com/DivyaKarade/Example-.csv-input-files--AIDrugApp-v1.2/blob/main/Mol_identifier.csv)
            """)

            if uploaded_file is not None:
                # Read CSV data
                @st.cache
                def load_csv():
                    csv = pd.read_csv(uploaded_file)
                    return csv


                df1 = load_csv()
                df2 = df1.iloc[:, 0:8]
                X = df2
                # Write CSV data
                df2.to_csv('molecule.txt', sep='\t', header=False, index=False)
                st.subheader('Uploaded data')
                st.write(df2)
                List_of_Chemicals = df2['Name'].values

                if st.sidebar.button("😊 GET SMILES"):
                    if add_selectbox3 == "Canonical SMILES":
                        st.subheader('Canonical SMILES')
                        # list of chemical names
                        data = []
                        for Name in List_of_Chemicals:
                            try:
                                df4 = pcp.get_properties(
                                    ['canonical_smiles'], Name, 'name')

                                # st.write(df4)
                                data.append(df4)
                            except (pcp.BadRequestError, TimeoutError, urllib.error.URLError, ValueError):
                                pass
                        # st.write(data)

                        rows = []
                        columns = data[0][0].keys()
                        for i in range(len(data)):
                            rows.append(data[i][0].values())
                        props_df = pd.DataFrame(data=rows, columns=columns)
                        # st.write(props_df)
                        props_df.insert(0, 'Name', df2['Name'], True)
                        st.write(props_df)
                        st.download_button('Download CSV', props_df.to_csv(), 'Can_smile.csv', 'text/csv')

                    if add_selectbox3 == "Isomeric SMILES":
                        # list of chemical names
                        st.subheader('Isomeric SMILES')
                        # list of chemical names
                        data = []
                        for Name in List_of_Chemicals:
                            try:
                                df4 = pcp.get_properties(
                                    ['isomeric_smiles'], Name, 'name')

                                # st.write(df4)
                                data.append(df4)
                            except (pcp.BadRequestError, TimeoutError, urllib.error.URLError, ValueError):
                                pass
                        # st.write(data)

                        rows = []
                        columns = data[0][0].keys()
                        for i in range(len(data)):
                            rows.append(data[i][0].values())
                        props_df = pd.DataFrame(data=rows, columns=columns)
                        # st.write(props_df)
                        props_df.insert(0, 'Name', df2['Name'], True)
                        st.write(props_df)
                        st.download_button('Download CSV', props_df.to_csv(), 'isomeric_smile.csv', 'text/csv')

        if add_selectbox2 == "SMILE to Compound":
            # Sidebar
            with st.sidebar.subheader('3. Upload your CSV data'):
                uploaded_file = st.sidebar.file_uploader(
                    "Upload your input CSV file containing 'smile' as one column of molecular SMILES", type=["csv"])
                st.sidebar.markdown("""
                [Example CSV input file](https://github.com/DivyaKarade/Example-.csv-input-files--AIDrugApp-v1.2/blob/main/Mol_identifier.csv)
                """)

            if uploaded_file is not None:
                # Read CSV data
                @st.cache
                def load_csv():
                    csv = pd.read_csv(uploaded_file)
                    return csv


                df1 = load_csv()
                df2 = df1.iloc[:, 0:]
                X = df2
                # Write CSV data
                df2.to_csv('molecule.smi', sep='\t', header=False, index=False)
                st.subheader('Uploaded data')
                st.write(df2)
                smile = df2['smiles'].values

                if st.sidebar.button("😊 GET COMPOUND"):

                    # list of chemical names
                    st.subheader('PubChem Compound Information')

                    data = []
                    for smiles in smile:
                        try:
                            df4 = pcp.get_properties(
                                ['canonical_smiles', 'molecular_formula', 'iupac_name', 'inchi', 'inchikey'], smiles,
                                'smiles')
                            # st.write(df4)
                            data.append(df4)
                        except (pcp.BadRequestError, TimeoutError, urllib.error.URLError, ValueError):
                            pass
                    # st.write(data)

                    rows = []
                    columns = data[0][0].keys()
                    for i in range(len(data)):
                        rows.append(data[i][0].values())
                    props_df = pd.DataFrame(data=rows, columns=columns)
                    # st.write(props_df)
                    props_df.insert(0, 'smiles', df2['smiles'], True)
                    st.write(props_df)
                    st.download_button('Download CSV', props_df.to_csv(), 'Cmpd_info.csv', 'text/csv')

        if add_selectbox2 == "Molecular Similarity":
            # Sidebar
            with st.sidebar.subheader('3. Upload your CSV data'):
                uploaded_file = st.sidebar.file_uploader(
                    "Upload input file containing 'Name' & 'smile' as two columns of compound names and SMILES",
                    type=["csv"])
                st.sidebar.markdown("""
                                [Example CSV input file](https://github.com/DivyaKarade/Example-.csv-input-files--AIDrugApp-v1.2/blob/main/Mol_identifier.csv)
                                """)

            if uploaded_file is not None:
                # Read CSV data
                @st.cache
                def load_csv():
                    csv = pd.read_csv(uploaded_file)
                    return csv


                df1 = load_csv()
                df2 = df1.iloc[:, 0:]
                X = df2
                # Write CSV data
                df2.to_csv('molecule.smi', sep='\t', header=False, index=False)
                st.subheader('Uploaded data')
                st.write(df2)
                smile = df2['smiles'].values
                Name = df2['Name'].values

                if st.sidebar.button("😊 GET SIMILARITY"):
                    st.write("****")
                    st.subheader('Computation of similarity scores')
                    st.info('**Similarity scores between compounds**')

                    mols = [Chem.MolFromSmiles(smiles) for smiles in smile]
                    fps = [MACCSkeys.GenMACCSKeys(x) for x in mols]
                    st.write("Number of compounds:", len(mols))
                    st.write("Number of fingerprints:", len(fps))
                    st.write("The number of compound pairs:", (len(fps) * (len(fps) - 1)) / 2)

                    scores = []

                    for i in range(0, len(fps)):

                        if i == 0:
                            print("Processing compound ", end='')

                        if i % 100 == 0:
                            print(i, end=' ')

                        for j in range(i + 1, len(fps)):
                            scores.append(DataStructs.FingerprintSimilarity(fps[i], fps[j]))

                    st.write("Number of scores : ", len(scores))
                    st.write("****")
                    st.info('**Similarity Scores**')
                    st.write("Tanimoto    :", round(DataStructs.TanimotoSimilarity(fps[0], fps[1]), 4))
                    st.write("Dice        :", round(DataStructs.DiceSimilarity(fps[0], fps[1]), 4))
                    st.write("Cosine      :", round(DataStructs.CosineSimilarity(fps[0], fps[1]), 4))
                    st.write("Sokal       :", round(DataStructs.SokalSimilarity(fps[0], fps[1]), 4))
                    st.write("McConnaughey:", round(DataStructs.McConnaugheySimilarity(fps[0], fps[1]), 4))

                    # Generate a histogram that shows the distribution of the pair-wise scores
                    st.write("****")
                    st.subheader('Distribution of similarity scores')
                    st.info('**Histograms showing the distribution of the pair-wise scores**')
                    mybins = [x * 0.01 for x in range(101)]

                    fig = plt.figure(figsize=(8, 4), dpi=300)

                    plt.subplot(1, 2, 1)
                    plt.title("Distribution")
                    plt.hist(scores, bins=mybins)

                    plt.subplot(1, 2, 2)
                    plt.title("Cumulative Distribution")
                    plt.hist(scores, bins=mybins, density=True, cumulative=1)
                    plt.plot([0, 1], [0.95, 0.95])
                    st.pyplot()

                    st.info("**Interpretation of similarity scores**")
                    st.write(
                        "Average similarity score between two compounds (computed using the Tanimoto equation and MACCS keys) :",
                        sum(scores) / len(scores))
                    # to find a threshold for top 3% compound pairs (i.e., 97% percentile)
                    st.write("Total compound pairs:   ", len(scores))
                    st.write("95% of compound pairs:  ", len(scores) * 0.97)
                    st.write("Score at 95% percentile:", scores[round(len(scores) * 0.97)])

                    st.info(
                        '**Table showing the % of compound pairs and their similarity scores.** (Table Columns: Similarity score, Number of compound pairs, % of compound pairs)')
                    for i in range(21):
                        thresh = i / 20
                        num_similar_pairs = len([x for x in scores if x >= thresh])
                        prob = num_similar_pairs / len(scores) * 100
                        st.write("%.3f  %8d  (%8.4f %%)" % (thresh, num_similar_pairs, round(prob, 4)))

                    st.write("****")
                    st.info('**Pair-wise similarity scores among molecules**'
                            ' (To make higher scores easier to find, they are indicated with the "+" character(s).)')
                    for i in range(0, len(fps)):
                        for j in range(i + 1, len(fps)):

                            score = DataStructs.FingerprintSimilarity(fps[i], fps[j])
                            st.write(Name[i], "vs.", Name[j], ":", round(score, 3), end='')

                            if score >= 0.85:
                                st.write(" ++++ ")
                            elif score >= 0.75:
                                st.write(" +++ ")
                            elif score >= 0.65:
                                st.write(" ++ ")
                            elif score >= 0.55:
                                st.write(" + ")
                            else:
                                st.write(" ")

        if add_selectbox2 == "2-D Structure":
            # Sidebar
            with st.sidebar.subheader('3. Upload your CSV data'):
                uploaded_file = st.sidebar.file_uploader(
                    "Upload your input CSV file containing 'smile' as one column of molecular SMILES", type=["csv"])
                st.sidebar.markdown("""
                        [Example CSV input file](https://github.com/DivyaKarade/Example-.csv-input-files--AIDrugApp-v1.2/blob/main/Mol_identifier.csv)
                        """)

            if uploaded_file is not None:
                # Read CSV data
                @st.cache
                def load_csv():
                    csv = pd.read_csv(uploaded_file)
                    return csv


                df1 = load_csv()
                df2 = df1.iloc[:, 0:]
                X = df2
                # Write CSV data
                df2.to_csv('molecule.smi', sep='\t', header=False, index=False)
                st.subheader('Uploaded data')
                st.write(df2)
                smile = df2['smiles'].values

                if st.sidebar.button("😊 GET STRUCTURE"):
                    st.subheader('2-D structure')
                    # raw_html = mols2grid.display(df2, mapping={"smiles": "SMILES"}, subset=["img", "iupac_name"], tooltip=["molecular_formula", "inchikey"])._repr_html_()
                    raw_html = mols2grid.display(df2, mapping={"smiles": "SMILES"}, tooltip=["SMILES"])._repr_html_()
                    components.html(raw_html, width=900, height=900, scrolling=True)
