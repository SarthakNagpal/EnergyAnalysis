# Energy Consumption Prediction with LSTM

This project focuses on predicting energy consumption using Long Short-Term Memory (LSTM) neural networks and time series machine learning techniques. The dataset utilized is the PJM Hourly Energy Consumption dataset, which contains hourly power consumption data from PJM Interconnection LLC.

## Project Overview

In this project, we aim to:
- Utilize LSTM neural networks to predict energy consumption based on historical data.
- Perform data preprocessing and feature engineering to prepare the dataset for model training.
- Split the dataset into training and testing sets for model evaluation.
- Train multiple LSTM models with varying architectures and hyperparameters.
- Evaluate the performance of each model using metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R2) score.
- Compare the performance of different models and identify the most effective one for energy consumption prediction.

## Steps Taken

1. **Data Collection:** Acquired the PJM Hourly Energy Consumption dataset from PJM Interconnection LLC.
2. **Data Preprocessing:** Processed the dataset to handle missing values, convert data types, and remove unnecessary columns. Additionally, extracted features from the 'Datetime' column.
3. **Feature Engineering:** Created additional features such as hour of the day, day of the week, month, year, etc., to improve model performance.
4. **Train-Test Split:** Split the dataset into training and testing sets, with data before 2015 used for training and data from 2015 onwards used for testing.
5. **Model Training:** Built and trained LSTM models using TensorFlow/Keras, with varying architectures and hyperparameters.
6. **Model Evaluation:** Evaluated the performance of each model using metrics such as MSE, MAE, and R2 score on the testing set.
7. **Comparison:** Compared the performance of different models to identify the most effective one for energy consumption prediction.

## Results

The project resulted in the development of multiple LSTM models for energy consumption prediction. The performance of each model was evaluated using various metrics, with the best-performing model achieving the lowest MSE, MAE, and highest R2 score.


## Dependencies

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- TensorFlow/Keras
- Scikit-learn

