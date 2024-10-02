# AIDrugApp: Auto-Multi-ML Module

AIDrugApp is an AI-powered virtual screening web-app for drug discovery. This repository contains the code for the Auto-Multi-ML module, which allows users to develop and compare multiple machine learning models to select the best-performing algorithm for molecular data. Additionally, the app helps predict target data based on user-specific machine learning models.

## Features

- **Auto-Multi-ML Module**: Aids in developing and comparing various machine learning models for classification and regression tasks, helping users select the best-performing model for their data.
- **Exploratory Data Analysis (EDA)**: Provides detailed insights into the uploaded and feature-engineered data.
- **Multiple Machine Learning Models**: Performance comparison of variety of ML algorithms (e.g., Random Forest, Logistic Regression, Decision Tree).
- **Data Prediction**: Enables users to predict target data using their trained models.
- **Downloadable Reports and Data**: Model performance and predictions can be downloaded as CSV files.

## Installation

1. Install Python and necessary libraries:
    ```bash
    pip install streamlit pandas scikit-learn sweetviz lazypredict numpy
    ```

2. Clone the repository:
    ```bash
    git clone https://github.com/DivyaKarade/CustomML.git
    cd AIDrugApp
    ```

3. Run the application:
    ```bash
    streamlit run app.py
    ```

## Instructions for the Auto-Multi-ML Tool

### Sidebar Menu
- **Auto-Multi-ML Checkbox**: Enables the Auto-Multi-ML module for model comparison and selection.
- **Upload .csv Files**: Upload data files for building models and making predictions.
  - Example input files are provided in the repository:
    [Example .csv input file for ML model building](https://github.com/DivyaKarade/Example-.csv-input-files--AIDrugApp-v1.2)
    [Example .csv input file for predictions](https://github.com/DivyaKarade/Example-.csv-input-files--AIDrugApp-v1.2)

### Step-by-Step Instructions

1. **Select Algorithm**: Choose between 'Classification' or 'Regression'.
2. **Upload Descriptor Data**: Upload a `.csv` file containing descriptor data (with target data for model building).
3. **Select Exploratory Data Analysis (EDA) Options**: Check the boxes for EDA on uploaded or feature-engineered data.
4. **Build ML Model**: Evaluate multiple models and select the top-performing one for predictions from options such as Random Forest, Logistic Regression, Decision Tree, etc.
5. **Upload New Data for Prediction**: Provide data excluding target values to predict using the selected model.
6. **View and Download Results**: Results are displayed, and users can download predictions.

### Output

- Performance metrics of different ML models.
- Predictions based on the selected model.
- Downloadable CSV files of the results and predictions.

### Exploratory Data Analysis (EDA)
Enable detailed analysis of the uploaded dataset using Sweetviz. Users can visualize patterns, check correlations, and understand their data before building models.

### Upload CSV Files
The user can upload two types of CSV files:
- **Training Data**: For building multiple machine learning models.
- **Prediction Data**: For applying the best-performing model to new datasets.

## Publication

Divya Karade. (2021). Custom ML Module of AIDrugApp for Molecular Identification, Descriptor Calculation, and Building ML/DL QSAR Models. [ChemRxiv Preprint](https://doi.org/10.33774/chemrxiv-2021-3f1f9).

## How to Run the App

1. Clone this repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the app using the command: `streamlit run app.py`.
4. The app will open in your default browser.

## License
AIDrugApp is licensed under the MIT License. See the `LICENSE` file for more details.

## Contributions
Contributions are welcome. Feel free to submit a pull request or open an issue for discussion.


For more details on how to use the Auto-Multi-ML module, refer to the [AIDrugApp](https://aidrugapp.streamlit.app/).
