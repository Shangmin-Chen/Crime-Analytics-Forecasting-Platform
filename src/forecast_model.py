# forecast_model.py
import pandas as pd
from data_prep import load_and_preprocess_data, validate_data
from prophet import Prophet
import joblib
import os
import yaml
from datetime import timedelta

# Check if cmdstanpy is installed and cmdstan is available
try:
    import cmdstanpy
    try:
        cmdstanpy.utils.cmdstan_path()
    except Exception:
        print("ERROR: CmdStan is not installed.")
        print("Please run: python -c 'import cmdstanpy; cmdstanpy.install_cmdstan()'")
        print("Or run: make install (which will install it automatically)")
        exit(1)
except ImportError:
    print("ERROR: cmdstanpy is not installed.")
    print("Please run: pip install cmdstanpy")
    print("Then run: python -c 'import cmdstanpy; cmdstanpy.install_cmdstan()'")
    print("Or run: make install (which will install it automatically)")
    exit(1)

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        print(f"WARNING: Config file {config_path} not found. Using defaults.")
        return {
            'training': {
                'use_fixed_dates': True,
                'train_start': '2023-01-01',
                'train_end': '2024-05-31',
                'test_end': '2024-10-31',
                'train_ratio': 0.8,
                'test_ratio': 0.15
            },
            'forecasting': {
                'future_months': 2,
                'forecast_start_offset_days': 1
            },
            'model': {
                'min_training_records': 10
            }
        }
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_data_date_range(daily_counts, config):
    """
    Calculate date ranges based on available data.
    
    Args:
        daily_counts: DataFrame with daily crime counts
        config: Configuration dictionary
    
    Returns:
        dict: Date ranges for training, testing, and forecasting
    """
    dates = pd.to_datetime(daily_counts['DATE'])
    data_start = dates.min()
    data_end = dates.max()
    
    train_ratio = config['training'].get('train_ratio', 0.8)
    test_ratio = config['training'].get('test_ratio', 0.15)
    
    # Calculate split points
    total_days = (data_end - data_start).days
    train_days = int(total_days * train_ratio)
    test_days = int(total_days * test_ratio)
    
    train_start = data_start
    train_end = data_start + pd.Timedelta(days=train_days)
    test_end = train_end + pd.Timedelta(days=test_days)
    
    # Future forecast: start after data_end
    forecast_start = data_end + pd.Timedelta(days=config['forecasting'].get('forecast_start_offset_days', 1))
    future_months = config['forecasting'].get('future_months', 2)
    forecast_end = forecast_start + pd.Timedelta(days=future_months * 30)
    
    return {
        'train_start': train_start,
        'train_end': train_end,
        'test_end': test_end,
        'future_start': forecast_start,
        'future_end': forecast_end
    }

def validate_training_data(daily_counts, train_start, train_end, min_records=10):
    """
    Validate we have sufficient data for training.
    
    Args:
        daily_counts: DataFrame with daily crime counts
        train_start: Start of training period
        train_end: End of training period
        min_records: Minimum number of records required
    
    Returns:
        bool: True if validation passes
    """
    available_data = daily_counts[
        (pd.to_datetime(daily_counts['DATE']) >= train_start) &
        (pd.to_datetime(daily_counts['DATE']) <= train_end)
    ]
    
    if len(available_data) < min_records:
        raise ValueError(f"Insufficient training data: only {len(available_data)} records (minimum: {min_records})")
    
    # Check for large gaps
    dates = pd.to_datetime(available_data['DATE']).sort_values()
    if len(dates) > 1:
        gaps = dates.diff().dt.days
        max_gap = gaps.max()
        
        if max_gap > 30:
            print(f"WARNING: Large gap of {max_gap} days in training data")
    
    return True

if __name__ == "__main__":
    # Load configuration
    config = load_config()
    
    # Load and preprocess daily counts by district
    validate_on_load = config.get('data', {}).get('validate_on_load', True)
    daily_counts = load_and_preprocess_data(validate=validate_on_load)

    # Create the forecasts directory if it doesn't exist
    os.makedirs("output/forecasts", exist_ok=True)

    # Determine date ranges based on configuration
    if config['training'].get('use_fixed_dates', True):
        # Use fixed dates from config
        train_start = pd.Timestamp(config['training']['train_start'])
        train_end = pd.Timestamp(config['training']['train_end'])
        test_end = pd.Timestamp(config['training']['test_end'])
        
        # Future forecasting period from config
        future_months = config['forecasting'].get('future_months', 2)
        future_start = pd.Timestamp(config['training']['test_end']) + pd.Timedelta(days=1)
        future_end = future_start + pd.Timedelta(days=future_months * 30)
        
        print(f"Using fixed dates from config:")
        print(f"  Training: {train_start.date()} to {train_end.date()}")
        print(f"  Testing: {train_end.date()} to {test_end.date()}")
        print(f"  Future: {future_start.date()} to {future_end.date()}")
    else:
        # Calculate dates dynamically from available data
        date_ranges = get_data_date_range(daily_counts, config)
        train_start = date_ranges['train_start']
        train_end = date_ranges['train_end']
        test_end = date_ranges['test_end']
        future_start = date_ranges['future_start']
        future_end = date_ranges['future_end']
        
        print(f"Using dynamic dates based on available data:")
        print(f"  Data range: {pd.to_datetime(daily_counts['DATE']).min().date()} to {pd.to_datetime(daily_counts['DATE']).max().date()}")
        print(f"  Training: {train_start.date()} to {train_end.date()}")
        print(f"  Testing: {train_end.date()} to {test_end.date()}")
        print(f"  Future: {future_start.date()} to {future_end.date()}")

    # Get unique districts
    districts = daily_counts['DISTRICT'].unique()

    for district in districts:
        print(f"Processing district: {district}")
        df = daily_counts[daily_counts['DISTRICT'] == district].copy()

        # Prophet requires 'ds' (date) and 'y' (value) columns
        df.rename(columns={'DATE': 'ds', 'COUNT': 'y'}, inplace=True)
        df['ds'] = pd.to_datetime(df['ds'])

        # Filter data from train_start
        df = df[df['ds'] >= train_start]

        # Split data into training and testing based on the specified cutoff
        train_df = df[(df['ds'] >= train_start) & (df['ds'] <= train_end)]
        test_df = df[(df['ds'] > train_end) & (df['ds'] <= test_end)]

        # Validate training data
        min_records = config['model'].get('min_training_records', 10)
        try:
            validate_training_data(daily_counts[daily_counts['DISTRICT'] == district], 
                                 train_start, train_end, min_records)
        except ValueError as e:
            print(f"Skipping district {district}: {e}")
            continue

        # Train the Prophet model with config parameters
        model = Prophet(
            yearly_seasonality=config['model'].get('yearly_seasonality', True),
            weekly_seasonality=config['model'].get('weekly_seasonality', True),
            daily_seasonality=config['model'].get('daily_seasonality', False)
        )
        model.fit(train_df)

        # Forecast for the test period
        if not test_df.empty:
            # Calculate how many days to forecast for the test set
            forecast_days_test = (test_df['ds'].max() - train_end).days
            if forecast_days_test < 1:
                forecast_days_test = 30
        else:
            # No test data, just forecast 30 days beyond train_end
            forecast_days_test = 30

        future_test = model.make_future_dataframe(periods=forecast_days_test)
        forecast_test = model.predict(future_test)

        # Evaluate on test data if available
        if not test_df.empty:
            merged = pd.merge(test_df, forecast_test[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds', how='left')
            mae = (merged['y'] - merged['yhat']).abs().mean()
            test_period_str = f"{train_end.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}"
            print(f"District {district} - Mean Absolute Error (Test: {test_period_str}): {mae:.2f}")
            merged.to_csv(f"output/forecasts/{district}_test_results.csv", index=False)
        else:
            print(f"No test data available after {train_end.strftime('%Y-%m-%d')} for district {district}.")
            merged = None

        # Save model and forecast for the test period
        forecast_test.to_csv(f"output/forecasts/{district}_forecast.csv", index=False)
        joblib.dump(model, f"output/forecasts/{district}_prophet_model.joblib")
        print(f"Test period forecast saved for district {district}.")

        # Now forecast the next 2 months (Dec 2024 & Jan 2025)
        # Calculate total days from train_end to future_end
        total_days_future = (future_end - train_end).days
        if total_days_future < 1:
            print("Future end date is before training end date, adjust your dates.")
            continue

        future_all = model.make_future_dataframe(periods=total_days_future)
        forecast_future = model.predict(future_all)

        # Filter forecast to future period
        forecast_2months = forecast_future[(forecast_future['ds'] >= future_start) & (forecast_future['ds'] <= future_end)]
        future_period_str = f"{future_start.strftime('%Y-%m-%d')} to {future_end.strftime('%Y-%m-%d')}"
        print(f"Predictions for {district} ({future_period_str}):")
        print(forecast_2months[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

        # Optionally, save the long-term forecast
        forecast_2months.to_csv(f"output/forecasts/{district}_2months_future_forecast.csv", index=False)
        print(f"2-month future forecast saved for district {district}.")
