# forecast_model.py
"""
Crime Forecasting Model using Facebook Prophet

This module trains time series forecasting models for crime prediction by district.
It implements industry-standard ML practices including:
- Comprehensive evaluation metrics (MAE, RMSE, MAPE, coverage)
- Train/test/future data splitting
- Baseline model comparisons
- Holiday effects
- Retraining on full dataset before production forecasts
- Outlier detection and handling

Author: Simon Chen
Last Updated: December 2025
"""

import pandas as pd
import numpy as np
from data_prep import load_and_preprocess_data, validate_data
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import joblib
import os
import yaml
from datetime import timedelta

# ============================================================================
# DEPENDENCY CHECKS
# ============================================================================

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

# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================

def load_config(config_path="config.yaml"):
    """
    Load configuration from YAML file with sensible defaults.
    
    Args:
        config_path (str): Path to configuration YAML file
        
    Returns:
        dict: Configuration dictionary with all required keys
    """
    if not os.path.exists(config_path):
        print(f"WARNING: Config file {config_path} not found. Using defaults.")
        return {
            'training': {
                'use_fixed_dates': True,
                'train_start': '2023-01-01',
                'train_end': '2024-12-31',
                'test_end': '2025-04-04',
                'train_ratio': 0.8,
                'test_ratio': 0.15
            },
            'forecasting': {
                'future_months': 2,
                'forecast_start_offset_days': 1
            },
            'model': {
                'min_training_records': 10,
                'yearly_seasonality': True,
                'weekly_seasonality': True,
                'daily_seasonality': False,
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 10.0,
                'include_holidays': True
            }
        }
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# ============================================================================
# DATE RANGE CALCULATIONS
# ============================================================================

def get_data_date_range(daily_counts, config):
    """
    Calculate date ranges based on available data.
    
    Uses ratio-based splitting to determine train/test/validation periods
    when fixed dates are not specified in config.
    
    Args:
        daily_counts (pd.DataFrame): DataFrame with daily crime counts
        config (dict): Configuration dictionary
    
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

# ============================================================================
# DATA VALIDATION
# ============================================================================

def validate_training_data(daily_counts, train_start, train_end, min_records=10):
    """
    Validate we have sufficient data for training.
    
    Checks for:
    - Minimum number of records
    - Large gaps in data that could affect model quality
    
    Args:
        daily_counts (pd.DataFrame): DataFrame with daily crime counts
        train_start (pd.Timestamp): Start of training period
        train_end (pd.Timestamp): End of training period
        min_records (int): Minimum number of records required
    
    Returns:
        bool: True if validation passes
        
    Raises:
        ValueError: If insufficient training data is available
    """
    available_data = daily_counts[
        (pd.to_datetime(daily_counts['DATE']) >= train_start) &
        (pd.to_datetime(daily_counts['DATE']) <= train_end)
    ]
    
    if len(available_data) < min_records:
        raise ValueError(f"Insufficient training data: only {len(available_data)} records (minimum: {min_records})")
    
    # Check for large gaps that could indicate data quality issues
    dates = pd.to_datetime(available_data['DATE']).sort_values()
    if len(dates) > 1:
        gaps = dates.diff().dt.days
        max_gap = gaps.max()
        
        if max_gap > 30:
            print(f"WARNING: Large gap of {max_gap} days in training data")
    
    return True

# ============================================================================
# OUTLIER DETECTION
# ============================================================================

def detect_and_handle_outliers(df, threshold=3):
    """
    Detect and cap outliers using z-score method.
    
    Outliers can skew model training. This function identifies values that are
    more than `threshold` standard deviations from the mean and caps them.
    
    Args:
        df (pd.DataFrame): DataFrame with 'y' column containing values
        threshold (float): Number of standard deviations for outlier detection
    
    Returns:
        tuple: (modified_df, outlier_count)
    """
    df = df.copy()
    mean = df['y'].mean()
    std = df['y'].std()
    
    # Identify outliers
    outliers = np.abs(df['y'] - mean) > threshold * std
    outlier_count = outliers.sum()
    
    if outlier_count > 0:
        print(f"  Found {outlier_count} outliers ({outlier_count/len(df)*100:.1f}%) - capping to {threshold} std devs")
        # Cap outliers at threshold
        df.loc[outliers, 'y'] = np.where(
            df.loc[outliers, 'y'] > mean,
            mean + threshold * std,
            mean - threshold * std
        )
    
    return df, outlier_count

# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_model_comprehensive(test_df, forecast_df):
    """
    Comprehensive model evaluation with multiple metrics.
    
    Metrics:
    - MAE: Mean Absolute Error (average magnitude of errors)
    - RMSE: Root Mean Square Error (penalizes large errors more)
    - MAPE: Mean Absolute Percentage Error (relative error as percentage)
    - Coverage: Percentage of actual values within prediction intervals
    
    Args:
        test_df (pd.DataFrame): Test data with 'ds' (date) and 'y' (actual) columns
        forecast_df (pd.DataFrame): Forecast with 'ds', 'yhat', 'yhat_lower', 'yhat_upper'
    
    Returns:
        dict: Dictionary containing all metrics, or None if merge fails
    """
    merged = pd.merge(
        test_df, 
        forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], 
        on='ds', 
        how='inner'
    )
    
    if len(merged) == 0:
        print("  WARNING: No matching dates between test and forecast data")
        return None
    
    y_true = merged['y'].values
    y_pred = merged['yhat'].values
    
    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    
    # Coverage: percentage of actuals within prediction interval
    coverage = np.mean(
        (y_true >= merged['yhat_lower']) & 
        (y_true <= merged['yhat_upper'])
    )
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'coverage': coverage,
        'test_samples': len(merged)
    }

def baseline_forecast(train_df, test_df):
    """
    Create simple baseline forecasts for comparison.
    
    Baselines help establish whether Prophet is actually adding value.
    Common baselines:
    - Naive: Use the last observed value
    - Moving Average: Use average of last N days
    - Seasonal Naive: Use value from same day last week
    
    Args:
        train_df (pd.DataFrame): Training data
        test_df (pd.DataFrame): Test data
    
    Returns:
        dict: MAE scores for each baseline method
    """
    baselines = {}
    
    # Naive forecast: last value persists
    last_value = train_df['y'].iloc[-1]
    baselines['naive'] = mean_absolute_error(
        test_df['y'], 
        [last_value] * len(test_df)
    )
    
    # 7-day moving average
    ma_value = train_df['y'].tail(7).mean()
    baselines['ma_7day'] = mean_absolute_error(
        test_df['y'], 
        [ma_value] * len(test_df)
    )
    
    # 30-day moving average
    ma_value_30 = train_df['y'].tail(30).mean()
    baselines['ma_30day'] = mean_absolute_error(
        test_df['y'], 
        [ma_value_30] * len(test_df)
    )
    
    return baselines

# ============================================================================
# MAIN FORECASTING PIPELINE
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("BOSTON CRIME FORECASTING MODEL")
    print("="*70)
    
    # Load configuration
    config = load_config()
    
    # Load and preprocess daily counts by district
    print("\nLoading data...")
    validate_on_load = config.get('data', {}).get('validate_on_load', True)
    daily_counts = load_and_preprocess_data(validate=validate_on_load)
    print(f"Loaded {len(daily_counts)} records across {daily_counts['DISTRICT'].nunique()} districts")

    # Create output directory
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
        
        print(f"\nUsing fixed dates from config:")
        print(f"  Training:  {train_start.date()} to {train_end.date()}")
        print(f"  Testing:   {train_end.date()} to {test_end.date()}")
        print(f"  Future:    {future_start.date()} to {future_end.date()}")
    else:
        # Calculate dates dynamically from available data
        date_ranges = get_data_date_range(daily_counts, config)
        train_start = date_ranges['train_start']
        train_end = date_ranges['train_end']
        test_end = date_ranges['test_end']
        future_start = date_ranges['future_start']
        future_end = date_ranges['future_end']
        
        print(f"\nUsing dynamic dates based on available data:")
        print(f"  Data range: {pd.to_datetime(daily_counts['DATE']).min().date()} to {pd.to_datetime(daily_counts['DATE']).max().date()}")
        print(f"  Training:   {train_start.date()} to {train_end.date()}")
        print(f"  Testing:    {train_end.date()} to {test_end.date()}")
        print(f"  Future:     {future_start.date()} to {future_end.date()}")

    # Get unique districts
    districts = daily_counts['DISTRICT'].unique()
    print(f"\nProcessing {len(districts)} districts: {', '.join(districts)}")
    
    # Store all metrics for summary report
    all_metrics = []

    # ========================================================================
    # PROCESS EACH DISTRICT
    # ========================================================================
    for district in districts:
        print(f"\n{'='*70}")
        print(f"DISTRICT: {district}")
        print(f"{'='*70}")
        
        # Filter data for this district
        df = daily_counts[daily_counts['DISTRICT'] == district].copy()

        # Prophet requires 'ds' (date) and 'y' (value) columns
        df.rename(columns={'DATE': 'ds', 'COUNT': 'y'}, inplace=True)
        df['ds'] = pd.to_datetime(df['ds'])

        # Filter data from train_start
        df = df[df['ds'] >= train_start]

        # Split data into training and testing
        train_df = df[(df['ds'] >= train_start) & (df['ds'] <= train_end)]
        test_df = df[(df['ds'] > train_end) & (df['ds'] <= test_end)]

        print(f"Data: {len(train_df)} training samples, {len(test_df)} test samples")

        # Validate training data
        min_records = config['model'].get('min_training_records', 10)
        try:
            validate_training_data(
                daily_counts[daily_counts['DISTRICT'] == district], 
                train_start, 
                train_end, 
                min_records
            )
        except ValueError as e:
            print(f"❌ Skipping district {district}: {e}")
            continue

        # Handle outliers in training data
        print("\nOutlier detection:")
        train_df, outlier_count = detect_and_handle_outliers(train_df, threshold=3)

        # ====================================================================
        # STEP 1: TRAIN INITIAL MODEL ON TRAINING DATA
        # ====================================================================
        print(f"\nStep 1: Training initial model on training data...")
        
        model = Prophet(
            yearly_seasonality=config['model'].get('yearly_seasonality', True),
            weekly_seasonality=config['model'].get('weekly_seasonality', True),
            daily_seasonality=config['model'].get('daily_seasonality', False),
            changepoint_prior_scale=config['model'].get('changepoint_prior_scale', 0.05),
            seasonality_prior_scale=config['model'].get('seasonality_prior_scale', 10.0)
        )
        
        # Add holiday effects if configured
        if config['model'].get('include_holidays', True):
            model.add_country_holidays(country_name='US')
            print("  Added US holiday effects")
        
        model.fit(train_df)
        print("  ✓ Model trained")

        # ====================================================================
        # STEP 2: EVALUATE ON TEST SET
        # ====================================================================
        if not test_df.empty:
            print(f"\nStep 2: Evaluating on test set...")
            
            # Calculate baseline performance
            baselines = baseline_forecast(train_df, test_df)
            print(f"\n  Baseline MAE scores:")
            for name, mae in baselines.items():
                print(f"    {name:12s}: {mae:.2f}")
            
            # Generate forecast for test period
            forecast_days_test = (test_df['ds'].max() - train_end).days
            if forecast_days_test < 1:
                forecast_days_test = 30
            
            future_test = model.make_future_dataframe(periods=forecast_days_test)
            forecast_test = model.predict(future_test)
            
            # Comprehensive evaluation
            metrics = evaluate_model_comprehensive(test_df, forecast_test)
            
            if metrics:
                test_period_str = f"{train_end.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}"
                print(f"\n  Prophet Model Performance ({test_period_str}):")
                print(f"    MAE:       {metrics['mae']:.2f} crimes/day")
                print(f"    RMSE:      {metrics['rmse']:.2f} crimes/day")
                print(f"    MAPE:      {metrics['mape']:.2f}%")
                print(f"    Coverage:  {metrics['coverage']*100:.1f}% (95% prediction interval)")
                print(f"    Samples:   {metrics['test_samples']}")
                
                # Compare to baseline
                improvement = (baselines['naive'] - metrics['mae']) / baselines['naive'] * 100
                print(f"\n  Improvement over naive baseline: {improvement:.1f}%")
                
                if improvement < 0:
                    print(f"  ⚠️  WARNING: Model performs worse than naive baseline!")
                
                # Store metrics for summary
                metrics['district'] = district
                metrics['baseline_naive_mae'] = baselines['naive']
                metrics['improvement_pct'] = improvement
                all_metrics.append(metrics)
                
                # Save test results
                merged = pd.merge(
                    test_df, 
                    forecast_test[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], 
                    on='ds', 
                    how='left'
                )
                merged.to_csv(f"output/forecasts/{district}_test_results.csv", index=False)
                print(f"  ✓ Test results saved")
        else:
            print(f"\n  No test data available after {train_end.strftime('%Y-%m-%d')}")
            metrics = None

        # Save initial model and forecast
        forecast_test.to_csv(f"output/forecasts/{district}_forecast.csv", index=False)
        joblib.dump(model, f"output/forecasts/{district}_initial_model.joblib")

        # ====================================================================
        # STEP 3: RETRAIN ON FULL DATASET FOR PRODUCTION FORECAST
        # ====================================================================
        print(f"\nStep 3: Retraining on FULL dataset (train + test) for production forecast...")
        
        # Combine all data up to test_end
        full_train_df = df[df['ds'] <= test_end]
        full_train_df, _ = detect_and_handle_outliers(full_train_df, threshold=3)
        
        print(f"  Training final model on {len(full_train_df)} samples")
        
        # Train final production model
        final_model = Prophet(
            yearly_seasonality=config['model'].get('yearly_seasonality', True),
            weekly_seasonality=config['model'].get('weekly_seasonality', True),
            daily_seasonality=config['model'].get('daily_seasonality', False),
            changepoint_prior_scale=config['model'].get('changepoint_prior_scale', 0.05),
            seasonality_prior_scale=config['model'].get('seasonality_prior_scale', 10.0)
        )
        
        if config['model'].get('include_holidays', True):
            final_model.add_country_holidays(country_name='US')
        
        final_model.fit(full_train_df)
        print("  ✓ Final model trained")

        # ====================================================================
        # STEP 4: GENERATE FUTURE FORECAST
        # ====================================================================
        print(f"\nStep 4: Generating future forecast...")
        
        total_days_future = (future_end - test_end).days
        if total_days_future < 1:
            print("  ❌ ERROR: Future end date is before test end date")
            continue

        future_all = final_model.make_future_dataframe(periods=total_days_future)
        forecast_future = final_model.predict(future_all)

        # Filter forecast to future period only
        forecast_2months = forecast_future[
            (forecast_future['ds'] >= future_start) & 
            (forecast_future['ds'] <= future_end)
        ]
        
        future_period_str = f"{future_start.strftime('%Y-%m-%d')} to {future_end.strftime('%Y-%m-%d')}"
        print(f"\n  Future Forecast Summary ({future_period_str}):")
        print(f"    Mean:   {forecast_2months['yhat'].mean():.1f} crimes/day")
        print(f"    Median: {forecast_2months['yhat'].median():.1f} crimes/day")
        print(f"    Range:  {forecast_2months['yhat'].min():.1f} - {forecast_2months['yhat'].max():.1f}")
        print(f"    Total:  {forecast_2months['yhat'].sum():.0f} crimes (predicted)")

        # Save production forecast and model
        forecast_2months.to_csv(f"output/forecasts/{district}_2months_future_forecast.csv", index=False)
        joblib.dump(final_model, f"output/forecasts/{district}_final_model.joblib")
        
        # Save metrics
        if metrics:
            metrics_df = pd.DataFrame([{
                'district': district,
                'mae': metrics['mae'],
                'rmse': metrics['rmse'],
                'mape': metrics['mape'],
                'coverage': metrics['coverage'],
                'test_samples': metrics['test_samples'],
                'baseline_naive_mae': metrics['baseline_naive_mae'],
                'improvement_pct': metrics['improvement_pct']
            }])
            metrics_df.to_csv(f"output/forecasts/{district}_metrics.csv", index=False)
        
        print(f"  ✓ Future forecast saved")
        print(f"  ✓ Final model saved")

    # ========================================================================
    # SUMMARY REPORT
    # ========================================================================
    print(f"\n{'='*70}")
    print("SUMMARY REPORT")
    print(f"{'='*70}")
    
    if all_metrics:
        summary_df = pd.DataFrame(all_metrics)
        
        print(f"\nAverage Performance Across All Districts:")
        print(f"  MAE:        {summary_df['mae'].mean():.2f} crimes/day")
        print(f"  RMSE:       {summary_df['rmse'].mean():.2f} crimes/day")
        print(f"  MAPE:       {summary_df['mape'].mean():.2f}%")
        print(f"  Coverage:   {summary_df['coverage'].mean()*100:.1f}%")
        print(f"  Improvement over baseline: {summary_df['improvement_pct'].mean():.1f}%")
        
        # Save summary
        summary_df.to_csv("output/forecasts/summary_metrics.csv", index=False)
        print(f"\n✓ Summary metrics saved to output/forecasts/summary_metrics.csv")
        
        # Identify best and worst performing districts
        best_district = summary_df.loc[summary_df['mae'].idxmin(), 'district']
        worst_district = summary_df.loc[summary_df['mae'].idxmax(), 'district']
        
        print(f"\nBest performing district:  {best_district} (MAE: {summary_df['mae'].min():.2f})")
        print(f"Worst performing district: {worst_district} (MAE: {summary_df['mae'].max():.2f})")
    else:
        print("\nNo metrics available - all districts may have been skipped")
    
    print(f"\n{'='*70}")
    print("FORECASTING COMPLETE")
    print(f"{'='*70}")