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
from sklearn.cluster import DBSCAN
import joblib
import os
import yaml
from datetime import timedelta
import logging
import gc
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

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
# CRIME-TYPE ANALYSIS GENERATION
# ============================================================================

def normalize_crime_type_name(crime_type):
    """Normalize crime type name for file naming."""
    # Replace problematic characters
    safe_name = str(crime_type).replace('/', '_').replace('\\', '_').replace(':', '_')
    safe_name = safe_name.replace('?', '').replace('*', '').replace('"', '').replace('<', '').replace('>', '')
    safe_name = safe_name.replace('|', '_')
    return safe_name

def validate_coordinates(df):
    """Validate and filter out invalid coordinates."""
    if df.empty:
        return df
    valid = df[['Lat', 'Long']].notna().all(axis=1)
    valid = valid & np.isfinite(df['Lat']) & np.isfinite(df['Long'])
    valid = valid & (df['Lat'].between(-90, 90)) & (df['Long'].between(-180, 180))
    return df[valid]

def load_and_sample_crime_data(df, crime_type, sample_size=10000, random_state=42):
    """Load and sample data for a specific crime type."""
    df_crime = df[df['OFFENSE_TYPE'] == crime_type].copy()
    if df_crime.empty:
        return pd.DataFrame(), np.array([])
    
    if len(df_crime) < sample_size:
        df_sample = df_crime.copy()
    else:
        df_sample = df_crime.sample(n=sample_size, random_state=random_state)
    
    df_sample = validate_coordinates(df_sample)
    if df_sample.empty:
        return pd.DataFrame(), np.array([])
    
    coords = df_sample[['Lat', 'Long']].values
    return df_sample, coords

def perform_dbscan_clustering(coords, eps=0.0001, min_samples=20):
    """Perform DBSCAN clustering on coordinates."""
    if len(coords) == 0:
        return np.array([])
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='haversine')
    coords_rad = np.radians(coords)
    cluster_labels = dbscan.fit_predict(coords_rad)
    return cluster_labels

def forecast_crime_cluster(df_clustered, cluster_id, periods=30):
    """Forecast crime counts for a specific cluster."""
    cluster_data = df_clustered[df_clustered['cluster'] == cluster_id].copy()
    if cluster_data.empty:
        return None, pd.DataFrame()
    
    cluster_data = cluster_data.rename(columns={'OCCURRED_ON_DATE': 'ds'})
    if cluster_data['ds'].dt.tz is not None:
        cluster_data['ds'] = cluster_data['ds'].dt.tz_convert(None)
    
    daily_counts = cluster_data.set_index('ds').resample('D').size().reset_index(name='y')
    if daily_counts['ds'].dt.tz is not None:
        daily_counts['ds'] = daily_counts['ds'].dt.tz_convert(None)
    
    min_required_points = 730
    if len(daily_counts) < min_required_points:
        if len(daily_counts) < 14:
            return None, pd.DataFrame()
        yearly_seasonality = False
    else:
        yearly_seasonality = True
    
    model = None
    try:
        model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=True,
            daily_seasonality=False
        )
        model.fit(daily_counts)
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        return model, forecast
    except Exception as e:
        logging.warning(f"Error forecasting cluster {cluster_id}: {e}")
        if model is not None:
            try:
                if hasattr(model, 'stan_backend') and model.stan_backend is not None:
                    del model.stan_backend
            except:
                pass
            del model
        gc.collect()
        return None, pd.DataFrame()
    finally:
        # Clean up model after use
        if model is not None:
            try:
                if hasattr(model, 'stan_backend') and model.stan_backend is not None:
                    try:
                        if hasattr(model.stan_backend, 'close'):
                            model.stan_backend.close()
                    except:
                        pass
                    del model.stan_backend
            except:
                pass
            del model
        gc.collect()
        try:
            import cmdstanpy
            if hasattr(cmdstanpy, 'clear_cache'):
                cmdstanpy.clear_cache()
        except:
            pass

def generate_crime_type_analyses():
    """Generate all crime-type analyses and save to files."""
    from preprocess_for_crime import preprocess_data
    
    print("\nLoading preprocessed data...")
    df = preprocess_data()
    
    if df.empty:
        print("  ❌ No data available for crime-type analysis")
        return
    
    # Get all unique crime types
    crime_types = df['OFFENSE_TYPE'].unique()
    print(f"  Found {len(crime_types)} unique crime types")
    
    # Create output directories
    os.makedirs('output/crime_maps', exist_ok=True)
    os.makedirs('output/crime_forecasts', exist_ok=True)
    
    # Prepare arguments for worker functions
    crime_args = [(crime_type, df) for crime_type in crime_types]
    
    # Limit workers to prevent memory exhaustion
    max_workers = min(8, multiprocessing.cpu_count())
    print(f"  Using {max_workers} parallel workers for crime-type processing")
    
    successful = 0
    failed = 0
    skipped = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_crime = {
            executor.submit(process_crime_type_analysis, args): args[0] 
            for args in crime_args
        }
        
        # Process results as they complete
        completed = 0
        total = len(crime_types)
        for future in as_completed(future_to_crime):
            crime_type = future_to_crime[future]
            completed += 1
            try:
                result = future.result()
                if result['status'] == 'success':
                    successful += 1
                    print(f"  [{completed}/{total}] ✓ {crime_type}")
                elif result['status'] == 'skipped':
                    skipped += 1
                    reason = result.get('reason', 'Unknown')
                    print(f"  [{completed}/{total}] ⚠️  {crime_type} - {reason}")
                else:
                    failed += 1
                    error = result.get('error', 'Unknown error')
                    print(f"  [{completed}/{total}] ❌ {crime_type} - {error}")
            except Exception as e:
                failed += 1
                logging.exception(f"Exception processing {crime_type}: {e}")
                print(f"  [{completed}/{total}] ❌ {crime_type} - Exception: {e}")
    
    print(f"\n{'='*70}")
    print(f"Crime-Type Analysis Summary:")
    print(f"  Successful: {successful}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(crime_types)}")
    print(f"{'='*70}")

# ============================================================================
# WORKER FUNCTIONS FOR CONCURRENT EXECUTION
# ============================================================================

def process_district_forecast(args):
    """Process a single district forecast - worker function for multiprocessing."""
    district, daily_counts, config, train_start, train_end, test_end, future_start, future_end = args
    
    try:
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
            return {'district': district, 'status': 'skipped', 'error': str(e)}
        
        # Handle outliers in training data
        train_df, outlier_count = detect_and_handle_outliers(train_df, threshold=3)
        
        # Train initial model
        model = Prophet(
            yearly_seasonality=config['model'].get('yearly_seasonality', True),
            weekly_seasonality=config['model'].get('weekly_seasonality', True),
            daily_seasonality=config['model'].get('daily_seasonality', False),
            changepoint_prior_scale=config['model'].get('changepoint_prior_scale', 0.05),
            seasonality_prior_scale=config['model'].get('seasonality_prior_scale', 10.0)
        )
        
        if config['model'].get('include_holidays', True):
            model.add_country_holidays(country_name='US')
        
        model.fit(train_df)
        
        # Evaluate on test set
        metrics = None
        forecast_test = None
        if not test_df.empty:
            baselines = baseline_forecast(train_df, test_df)
            forecast_days_test = (test_df['ds'].max() - train_end).days
            if forecast_days_test < 1:
                forecast_days_test = 30
            
            future_test = model.make_future_dataframe(periods=forecast_days_test)
            forecast_test = model.predict(future_test)
            metrics = evaluate_model_comprehensive(test_df, forecast_test)
            
            if metrics:
                improvement = (baselines['naive'] - metrics['mae']) / baselines['naive'] * 100
                metrics['district'] = district
                metrics['baseline_naive_mae'] = baselines['naive']
                metrics['improvement_pct'] = improvement
                
                # Save test results
                merged = pd.merge(
                    test_df, 
                    forecast_test[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], 
                    on='ds', 
                    how='left'
                )
                merged.to_csv(f"output/forecasts/{district}_test_results.csv", index=False)
        
        # Generate forecast for full period if test_df was empty
        if forecast_test is None:
            future_all = model.make_future_dataframe(periods=30)
            forecast_test = model.predict(future_all)
        
        # Save initial model and forecast
        forecast_test.to_csv(f"output/forecasts/{district}_forecast.csv", index=False)
        joblib.dump(model, f"output/forecasts/{district}_initial_model.joblib")
        
        # Retrain on full dataset
        full_train_df = df[df['ds'] <= test_end]
        full_train_df, _ = detect_and_handle_outliers(full_train_df, threshold=3)
        
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
        
        # Generate future forecast
        total_days_future = (future_end - test_end).days
        if total_days_future < 1:
            return {'district': district, 'status': 'error', 'error': 'Future end date is before test end date'}
        
        future_all = final_model.make_future_dataframe(periods=total_days_future)
        forecast_future = final_model.predict(future_all)
        
        forecast_2months = forecast_future[
            (forecast_future['ds'] >= future_start) & 
            (forecast_future['ds'] <= future_end)
        ]
        
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
        
        # Cleanup
        del model, final_model
        gc.collect()
        
        return {'district': district, 'status': 'success', 'metrics': metrics}
        
    except Exception as e:
        logging.exception(f"Error processing district {district}: {e}")
        return {'district': district, 'status': 'error', 'error': str(e)}

def process_crime_type_analysis(args):
    """Process a single crime type analysis - worker function for multiprocessing."""
    crime_type, df = args
    
    try:
        safe_name = normalize_crime_type_name(crime_type)
        
        # Load and sample data
        df_sample, coords = load_and_sample_crime_data(df, crime_type)
        if len(coords) == 0:
            return {'crime_type': crime_type, 'status': 'skipped', 'reason': 'No valid coordinates'}
        
        # Perform clustering
        cluster_labels = perform_dbscan_clustering(coords)
        if len(cluster_labels) == 0:
            return {'crime_type': crime_type, 'status': 'skipped', 'reason': 'No clusters found'}
        
        df_sample['cluster'] = cluster_labels
        df_clustered = df_sample[df_sample['cluster'] != -1].copy()
        
        if df_clustered.empty:
            return {'crime_type': crime_type, 'status': 'skipped', 'reason': 'No meaningful clusters'}
        
        # Get top cluster
        cluster_counts = df_clustered['cluster'].value_counts()
        if cluster_counts.empty:
            return {'crime_type': crime_type, 'status': 'skipped', 'reason': 'No cluster counts'}
        
        top_cluster_id = cluster_counts.index[0]
        
        # Generate forecast
        model, forecast = forecast_crime_cluster(df_clustered, top_cluster_id)
        if model is None or forecast.empty:
            return {'crime_type': crime_type, 'status': 'skipped', 'reason': 'Could not generate forecast'}
        
        # Save forecast CSV
        forecast_path = f"output/crime_forecasts/{safe_name}_forecast.csv"
        forecast.to_csv(forecast_path, index=False)
        
        # Generate and save forecast components plot
        try:
            fig = model.plot_components(forecast)
            components_path = f"output/crime_forecasts/{safe_name}_components.png"
            fig.savefig(components_path, bbox_inches='tight', dpi=150)
            plt.close(fig)
        except Exception as e:
            logging.warning(f"Could not save components plot for {crime_type}: {e}")
        
        # Save cluster information
        cluster_info = pd.DataFrame({
            'cluster_id': cluster_counts.index,
            'count': cluster_counts.values
        })
        cluster_info_path = f"output/crime_forecasts/{safe_name}_clusters.csv"
        cluster_info.to_csv(cluster_info_path, index=False)
        
        # Generate and save map
        try:
            center_lat = df_clustered['Lat'].mean()
            center_long = df_clustered['Long'].mean()
            
            m = folium.Map(location=[center_lat, center_long], zoom_start=12)
            
            top_clusters = cluster_counts.head(5).index.tolist()
            colors = ['red', 'blue', 'green', 'purple', 'orange']
            
            for cluster_id in top_clusters:
                cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
                color = colors[top_clusters.index(cluster_id) % len(colors)]
                
                for _, row in cluster_data.iterrows():
                    folium.CircleMarker(
                        location=[row['Lat'], row['Long']],
                        radius=5,
                        popup=f"Cluster {cluster_id}",
                        color=color,
                        fill=True,
                        fillColor=color,
                        fillOpacity=0.6
                    ).add_to(m)
            
            # Add legend to map
            legend_html = '''
            <div style="position: fixed; 
                 top: 10px; right: 10px; width: 180px; height: auto; 
                 background-color: white; z-index:9999; font-size:14px;
                 border:2px solid grey; border-radius:5px; padding: 10px;
                 box-shadow: 0 0 15px rgba(0,0,0,0.2);">
                 <h4 style="margin-top:0; margin-bottom:10px; font-size:16px; font-weight:bold;">Clusters</h4>
            '''
            for i, cluster_id in enumerate(top_clusters):
                color = colors[i % len(colors)]
                count = int(cluster_counts[cluster_id])
                legend_html += f'''
                <p style="margin:5px 0; display:flex; align-items:center;">
                    <span style="display:inline-block; width:16px; height:16px; background-color:{color}; 
                         border:1px solid #333; border-radius:50%; margin-right:8px;"></span>
                    <span>Cluster {int(cluster_id)} ({count:,})</span>
                </p>
                '''
            legend_html += '</div>'
            
            m.get_root().html.add_child(folium.Element(legend_html))
            
            map_path = f"output/crime_maps/{safe_name}_map.html"
            m.save(map_path)
        except Exception as e:
            logging.warning(f"Could not generate map for {crime_type}: {e}")
        
        # Cleanup
        del model
        gc.collect()
        
        return {'crime_type': crime_type, 'status': 'success'}
        
    except Exception as e:
        logging.exception(f"Error processing {crime_type}: {e}")
        return {'crime_type': crime_type, 'status': 'error', 'error': str(e)}

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
    # PROCESS DISTRICTS CONCURRENTLY
    # ========================================================================
    # Prepare arguments for worker functions
    district_args = [
        (district, daily_counts, config, train_start, train_end, test_end, future_start, future_end)
        for district in districts
    ]
    
    # Limit workers to prevent memory exhaustion
    max_workers = min(8, multiprocessing.cpu_count())
    print(f"\nUsing {max_workers} parallel workers for district processing")
    
    successful_districts = 0
    failed_districts = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_district = {
            executor.submit(process_district_forecast, args): args[0] 
            for args in district_args
        }
        
        # Process results as they complete
        for future in as_completed(future_to_district):
            district = future_to_district[future]
            try:
                result = future.result()
                if result['status'] == 'success':
                    successful_districts += 1
                    print(f"✓ Completed: {district}")
                    if 'metrics' in result and result['metrics']:
                        all_metrics.append(result['metrics'])
                elif result['status'] == 'skipped':
                    print(f"⚠️  Skipped: {district} - {result.get('error', result.get('reason', 'Unknown'))}")
                else:
                    failed_districts += 1
                    print(f"❌ Failed: {district} - {result.get('error', 'Unknown error')}")
            except Exception as e:
                failed_districts += 1
                logging.exception(f"Exception processing district {district}: {e}")
                print(f"❌ Exception: {district} - {e}")
    
    print(f"\nDistrict Processing Summary: {successful_districts} successful, {failed_districts} failed")

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
    
    # ========================================================================
    # CRIME-TYPE ANALYSIS GENERATION
    # ========================================================================
    print(f"\n{'='*70}")
    print("GENERATING CRIME-TYPE ANALYSES")
    print(f"{'='*70}")
    
    try:
        generate_crime_type_analyses()
    except Exception as e:
        print(f"\n⚠️  WARNING: Error generating crime-type analyses: {e}")
        logging.exception("Error in generate_crime_type_analyses")
        print("District forecasts completed successfully, but crime-type analyses failed.")
    
    print(f"\n{'='*70}")
    print("ALL FORECASTING COMPLETE")
    print(f"{'='*70}")