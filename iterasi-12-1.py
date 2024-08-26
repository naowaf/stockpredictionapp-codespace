import pandas as pd
import numpy as np
from prophet import Prophet
from lightgbm import LGBMRegressor
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st
import yfinance as yf
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.transformations.series.detrend import Detrender
import plotly.graph_objs as go
import joblib
import os


# Function to load stock data
def load_stock_data(symbol, start_date, end_date):
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        if data.empty:
            st.error(f"No data found for {symbol} in the specified date range.")
            return None
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


# Function to preprocess data
def preprocess_data(df):
    df = df[['Close']].reset_index()
    df.columns = ['ds', 'y']
    df['ds'] = pd.to_datetime(df['ds'])

    # Remove rows with NaN values
    df = df.dropna()

    # If there are still NaN values after dropping, interpolate
    if df['y'].isnull().any():
        df['y'] = df['y'].interpolate()

    return df


# Function to train Prophet model
def train_prophet(df):
    m = Prophet()
    m.fit(df)
    return m


# Function to predict with Prophet
def predict_prophet(model, future_dates):
    return model.predict(future_dates)


# Function to optimize LightGBM with Optuna
def optimize_lgbm(df, n_trials=100):
    def objective(trial):
        forecaster = PolynomialTrendForecaster(degree=1)
        detrender = Detrender(forecaster=forecaster)
        y_detrended = detrender.fit_transform(df['y'])

        X = df['ds'].astype(np.int64) // 10 ** 9  # Convert to Unix timestamp

        param = {
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'learning_rate': trial.suggest_float('learning_rate', 1e-8, 1.0, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 10, 1000),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
        }

        model = LGBMRegressor(**param)
        model.fit(X.values.reshape(-1, 1), y_detrended)

        future_dates = pd.date_range(start=df['ds'].max(), periods=365, freq='D')
        future_df = pd.DataFrame({'ds': future_dates})
        X_future = future_df['ds'].astype(np.int64) // 10 ** 9
        y_detrended_pred = model.predict(X_future.values.reshape(-1, 1))
        y_pred = detrender.inverse_transform(pd.Series(y_detrended_pred))

        y_true = df['y'][-len(y_pred):]
        y_pred = y_pred[:len(y_true)]

        # Remove any NaN values before calculating MSE
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        return mean_squared_error(y_true, y_pred)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


# Function to train LightGBM model
def train_lgbm(df, params):
    forecaster = PolynomialTrendForecaster(degree=1)
    detrender = Detrender(forecaster=forecaster)
    y_detrended = detrender.fit_transform(df['y'])

    X = df['ds'].astype(np.int64) // 10 ** 9  # Convert to Unix timestamp
    model = LGBMRegressor(**params)
    model.fit(X.values.reshape(-1, 1), y_detrended)
    return model, detrender


# Function to predict with LightGBM
def predict_lgbm(model, detrender, future_dates):
    X_future = future_dates['ds'].astype(np.int64) // 10 ** 9
    y_detrended_pred = model.predict(X_future.values.reshape(-1, 1))
    y_pred = detrender.inverse_transform(pd.Series(y_detrended_pred))
    y_pred = y_pred.fillna(method='ffill').fillna(method='bfill')  # Handle NaN values
    return y_pred


# Function to calculate evaluation metrics
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return mae, mse, rmse, mape, r2


# Streamlit app
def main():
    st.title('Stock Price Prediction: Prophet with LightGBM and Optuna')

    # User inputs
    symbol = st.text_input('Enter stock symbol (e.g., AAPL for Apple):')
    start_date = st.date_input('Start date')
    end_date = st.date_input('End date')
    forecast_period = st.selectbox('Select forecast period (years):', [1, 2, 3, 4, 5])
    n_trials = st.slider('Number of optimization trials', 10, 500, 100)

    if symbol and start_date and end_date:
        if start_date >= end_date:
            st.error('End date must be after start date.')
            return

        # Load and preprocess data
        data = load_stock_data(symbol, start_date, end_date)
        if data is None:
            return
        df = preprocess_data(data)

        # Check if dataframe is empty after preprocessing
        if df.empty:
            st.error("No valid data available after preprocessing. Please try a different date range or stock symbol.")
            return

        # Train models
        prophet_model = train_prophet(df)

        # Check if optimized parameters exist
        params_file = f'{symbol}_lgbm_params.joblib'
        if os.path.exists(params_file):
            best_params = joblib.load(params_file)
            st.info('Loaded optimized parameters from file.')
        else:
            with st.spinner('Optimizing LightGBM parameters...'):
                best_params = optimize_lgbm(df, n_trials)
            joblib.dump(best_params, params_file)
            st.success('Optimization complete. Parameters saved.')

        lgbm_model, detrender = train_lgbm(df, best_params)

        # Make predictions
        future = prophet_model.make_future_dataframe(periods=forecast_period * 365)
        prophet_pred = predict_prophet(prophet_model, future)
        lgbm_pred = predict_lgbm(lgbm_model, detrender, future)

        # Combine predictions
        combined_pred = (prophet_pred['yhat'] + lgbm_pred) / 2

        # Display processed data
        st.subheader('Processed Stock Data')
        st.write(df)

        # Plot predictions
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Actual'))
        fig.add_trace(go.Scatter(x=prophet_pred['ds'], y=prophet_pred['yhat'], mode='lines', name='Prophet'))
        fig.add_trace(go.Scatter(x=future['ds'], y=lgbm_pred, mode='lines', name='LGBM'))
        fig.add_trace(go.Scatter(x=future['ds'], y=combined_pred, mode='lines', name='Combined'))
        fig.update_layout(title='Stock Price Predictions', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig)

        # Calculate and display evaluation metrics
        y_true = df['y'][-len(lgbm_pred):]
        y_pred = combined_pred[:len(y_true)]
        mae, mse, rmse, mape, r2 = calculate_metrics(y_true, y_pred)

        st.subheader('Evaluation Metrics')
        st.write(f'MAE: {mae:.2f}')
        st.write(f'MSE: {mse:.2f}')
        st.write(f'RMSE: {rmse:.2f}')
        st.write(f'MAPE: {mape:.2f}%')
        st.write(f'R2 Score: {r2:.4f}')

    else:
        st.write('Please enter a stock symbol and select date range to begin.')


if __name__ == '__main__':
    main()