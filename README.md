# Osteoporosis Prediction in ICU Patients using Machine Learning

This project aims to develop a supervised machine learning model to predict osteoporosis risk in ICU patients, leveraging patient data related to Parathyroid Hormone (PTH) levels and urinary calcium-to-creatinine ratio (CCR). By identifying high-risk individuals early, this model can assist medical professionals in implementing timely interventions and personalized care.

## Project Structure

The project folder is organized as follows:

```plaintext
├── notebook
│   ├── data_collection.ipynb       # Data collection process using MIMIC-IV dataset
│   ├── data_processing.ipynb       # Data cleaning and preprocessing
│   ├── data_modeling.ipynb         # Model training, evaluation, and feature analysis
│
├── plot
│   ├── Hyperparathyroidism_feature_importance.svg  # Feature importance plot for Hyperparathyroidism
│   ├── Osteoporosis_feature_importance.svg         # Feature importance plot for Osteoporosis
│
├── data
│   ├── processed_Op_Hp_LabEvents.csv  # Cleaned dataset ready for modeling
│   ├── raw_Op_Hp_LabEvents.csv        # Original raw dataset collected from MIMIC-IV
```
## Project Overview

### Research Objective
The goal of this project is to predict osteoporosis risk in ICU patients based on PTH levels and CCR values. The model is designed to help healthcare professionals identify high-risk individuals, ultimately supporting preventive measures and improved patient outcomes.

### Dependencies
All required Python libraries are listed in `requirements.txt`:

```plaintext
imblearn==0.0
matplotlib==3.7.2
numpy==1.24.3
pandas==2.0.3
protobuf==3.20.3
scikit_learn==1.3.2
seaborn==0.13.2
```
To install these dependencies, run:

```bash
pip install -r requirements.txt
```

### Methodology

The project follows a structured pipeline comprising data collection, preprocessing, modeling, and evaluation steps:

1. Data Collection (`data_collection.ipynb`):

    - Sources ICU patient data from the MIMIC-IV dataset.

    - Filters relevant items (e.g., PTH and calcium data) for osteoporosis risk prediction.

2. Data Processing (`data_processing.ipynb`):

    - Cleans the collected data, handling missing values, normalizing features, and encoding categorical variables.

    - Prepares a processed dataset (processed_Op_Hp_LabEvents.csv) for modeling.

3. Modeling and Evaluation (`data_modeling.ipynb`):

    - Trains and evaluates machine learning models (e.g., Logistic Regression, Random Forest, Support Vector Machine).

    - Uses techniques like SMOTE for handling class imbalance.

    - Evaluates model performance using precision, recall, F1-score, and AUC-ROC.
    - Conducts feature importance analysis and visualizes significant predictors.

### Data Files

- `raw_Op_Hp_LabEvents.csv`: The original raw dataset extracted from MIMIC-IV for ICU patients.
- `processed_Op_Hp_LabEvents.csv`: The processed version of the raw dataset, cleaned and prepped for model training.

### Plots

- `Hyperparathyroidism_feature_importance.svg`: A plot illustrating the most important features for predicting hyperparathyroidism in the ICU patient dataset.

- `Osteoporosis_feature_importance.svg`: A feature importance plot for predicting osteoporosis, showing key risk indicators.

## Results
The modeling process produced a machine learning model capable of predicting osteoporosis with notable accuracy, identifying key predictors in ICU patients. By interpreting feature importance through plots, we gain insights into which variables, such as PTH and CCR, most significantly influence the model's predictions.

### Model Results

| Hyperparathyroidism    | precision | recall | f1-score | accuracy | ROC-AUC |
| ---------------------- | --------- | ------ | -------- | -------- | ------- |
| Logistic Regression    | 0.60      | 0.60   | 0.60     | 0.60     | 0.66    |
| Random Forest          | 0.69      | 0.66   | 0.64     | 0.66     | 0.71    |
| Support Vector Machine | 0.62      | 0.62   | 0.61     | 0.62     | 0.70    |


| Osteoporosis           | precision | recall | f1-score | accuracy | ROC-AUC |
| ---------------------- | --------- | ------ | -------- | -------- | ------- |
| Logistic Regression    | 0.74      | 0.74   | 0.74     | 0.74     | 0.79    |
| Random Forest          | 0.74      | 0.74   | 0.74     | 0.74     | 0.79    |
| Support Vector Machine | 0.75      | 0.74   | 0.73     | 0.74     | 0.79    |


### Feature Importance Accross different ML Models 

![Hyperparathyroidism_feature_importance](./plot/Hyperparathyroidism_feature_importance.svg)

![Osteoporosis_feature_importance](./plot/Osteoporosis_feature_importance.svg)

## How to Run

- Data Collection: Open `data_collection.ipynb` and run each cell to collect data from MIMIC-IV.

- Data Processing: Run `data_processing.ipynb` to clean and preprocess the collected data.

- Model Training and Evaluation: Open `data_modeling.ipynb` to train the model, optimize hyperparameters, and evaluate performance. Use the plot folder for visualization of feature importance.

## Reference

1. Johnson, A., Bulgarelli, L., Pollard, T., Gow, B., Moody, B., Horng, S., Celi, L. A., & Mark, R. (2024). MIMIC-IV (version 3.1). PhysioNet. https://doi.org/10.13026/kpb9-mt58.

2. Johnson, A.E.W., Bulgarelli, L., Shen, L. et al. MIMIC-IV, a freely accessible electronic health record dataset. Sci Data 10, 1 (2023). https://doi.org/10.1038/s41597-022-01899-x

3. Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220.

## Contact
For further information, please contact project team members:

- Chunxi Wang (ID: 1118838)
- Ze Pang (ID: 955698)
- Yuqi Wang (ID: 1445371)
- Pengyuan Yu (ID: 1433539)
