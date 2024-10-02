# AIDrugApp: Auto-Multi-ML Module

AIDrugApp is an AI-powered virtual screening web-app for drug discovery. This repository contains the code for the Auto-Multi-ML module, which allows users to develop and compare multiple machine learning models to select the best-performing algorithm for molecular data.

## Features

- Multiple ML models for classification and regression tasks.
- Exploratory data analysis (EDA) for uploaded and feature-engineered data.
- Performance comparison of ML models.
- Model prediction on user-supplied data.
- Downloadable data for further analysis.

## How to Use

1. Select the type of algorithm ('Classification' or 'Regression').
2. Upload your descriptor data file (included with target data).
3. Select options for EDA and interpreting ML models.
4. Evaluate the results of different ML models, and select one for further predictions.
5. Upload descriptor data for predictions, and use the selected ML model to predict target data.

## Installation

1. Install Python and necessary libraries:
    ```bash
    pip install streamlit pandas scikit-learn sweetviz lazypredict numpy
    ```

2. Clone the repository:
    ```bash
    git clone https://github.com/DivyaKarade/AIDrugApp.git
    cd AIDrugApp
    ```

3. Run the application:
    ```bash
    streamlit run app.py
    ```

## Python Libraries Used

- scikit-learn
- Streamlit
- pandas
- numpy
- lazypredict
- sweetviz

## Instructions for the Auto-Multi-ML Tool

### User Inputs

- **Algorithm Selection**: Choose between regression and classification tasks.
- **File Upload**: Upload a .csv file for training the model or for making predictions.
- **Exploratory Data Analysis**: Optional EDA for uploaded and feature-engineered data.
- **Model Selection**: Choose your machine learning model from options such as Random Forest, Logistic Regression, Decision Tree, etc.

### Output

- Performance metrics of different ML models.
- Predictions based on the selected model.
- Downloadable CSV files of the results and predictions.

## Example Input Files

- [Example .csv input file for ML model building](https://github.com/DivyaKarade/Example-.csv-input-files--AIDrugApp-v1.2)
- [Example .csv input file for predictions](https://github.com/DivyaKarade/Example-.csv-input-files--AIDrugApp-v1.2)

## Publication

Divya Karade. (2021). Custom ML Module of AIDrugApp for Molecular Identification, Descriptor Calculation, and Building ML/DL QSAR Models. [ChemRxiv Preprint](https://doi.org/10.33774/chemrxiv-2021-3f1f9).

## How to Run the App

1. Clone this repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the app using the command: `streamlit run app.py`.
4. The app will open in your default browser.

For more details on how to use the Auto-Multi-ML module, refer to the [AIDrugApp](https://aidrugapp.streamlit.app/).
