import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from lazypredict import LazyClassifier, LazyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
import sweetviz
import codecs
from sklearn.feature_selection import VarianceThreshold
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import streamlit.components.v1 as components

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

st.sidebar.title("AIDrugApp v1.2.6")
st.sidebar.header("Menu")

#st.sidebar.header("Custom ML Menu")

#add_selectbox = st.sidebar.radio("Select ML tool", ("Auto-Multi-ML", "Auto-DL", "DesCal", "Mol_Identifier"))

CB = st.sidebar.checkbox("Auto-Multi-ML")

#if CB == "Auto-Multi-ML":
if CB:
    st.title('Auto-Multi-ML')
    st.success(
        "This module of [**AIDrugApp v1.2.6**](https://aidrugapp.streamlit.app/) aids in the development and comparison of multiple machine learning models on user data in order to select the best performing machine learning algorithm. "
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
