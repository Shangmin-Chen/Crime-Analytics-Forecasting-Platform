# data_prep.py
import pandas as pd
import os
from datetime import datetime

def check_data_freshness(file_path="data/Boston Data.csv"):
    """
    Check how recent the data is and return freshness information.
    
    Returns:
        tuple: (latest_date, days_old) or (None, None) if file doesn't exist
    """
    if not os.path.exists(file_path):
        print("ERROR: Data file not found")
        return None, None
    
    try:
        # Read sample to check freshness (faster for large files)
        data = pd.read_csv(file_path, encoding='latin', low_memory=False, nrows=10000)
        data['OCCURRED_ON_DATE'] = pd.to_datetime(data['OCCURRED_ON_DATE'], errors='coerce')
        latest_date = data['OCCURRED_ON_DATE'].max()
        
        if pd.isna(latest_date):
            # If sample doesn't have dates, read more
            data = pd.read_csv(file_path, encoding='latin', low_memory=False, nrows=50000)
            data['OCCURRED_ON_DATE'] = pd.to_datetime(data['OCCURRED_ON_DATE'], errors='coerce')
            latest_date = data['OCCURRED_ON_DATE'].max()
        
        if pd.isna(latest_date):
            print("ERROR: Could not determine latest date")
            return None, None
        
        days_old = (pd.Timestamp.now() - latest_date).days
        
        print(f"Latest data date: {latest_date.date()}")
        print(f"Data is {days_old} days old")
        
        if days_old > 90:
            print("⚠️  WARNING: Data is more than 3 months old. Run 'make update-data' to refresh.")
        elif days_old > 30:
            print("ℹ️  INFO: Data is more than 1 month old. Consider updating.")
        else:
            print("✓ Data is relatively fresh")
        
        return latest_date, days_old
    except Exception as e:
        print(f"ERROR: Could not check data freshness: {e}")
        return None, None

def validate_data(data):
    """
    Validate data quality and completeness.
    
    Args:
        data: DataFrame with crime data
    
    Returns:
        list: List of validation issues (empty if no issues)
    """
    issues = []
    
    if data.empty:
        issues.append("ERROR: Data is empty")
        return issues
    
    # Check date range
    dates = pd.to_datetime(data['OCCURRED_ON_DATE'], errors='coerce')
    invalid_dates = dates.isna().sum()
    if invalid_dates > len(data) * 0.1:
        issues.append(f"WARNING: {invalid_dates} rows ({invalid_dates/len(data)*100:.1f}%) have invalid dates")
    
    # Check for missing districts
    if 'DISTRICT' in data.columns:
        missing_districts = data['DISTRICT'].isna().sum()
        if missing_districts > 0:
            issues.append(f"WARNING: {missing_districts} rows missing district")
    
    # Check date gaps (if we have enough data)
    valid_dates = dates.dropna()
    if len(valid_dates) > 1:
        date_range = valid_dates.max() - valid_dates.min()
        expected_days = date_range.days
        unique_days = valid_dates.dt.date.nunique()
        
        if expected_days > 0 and unique_days < expected_days * 0.8:
            issues.append(f"WARNING: Only {unique_days} unique days out of {expected_days} expected ({unique_days/expected_days*100:.1f}% coverage)")
    
    # Check for reasonable date range
    if len(valid_dates) > 0:
        oldest = valid_dates.min()
        newest = valid_dates.max()
        if (newest - oldest).days > 365 * 10:  # More than 10 years
            issues.append(f"WARNING: Data spans more than 10 years ({oldest.date()} to {newest.date()})")
    
    return issues

def load_and_preprocess_data(file_path="data/Boston Data.csv", validate=True):
    """
    Load and preprocess crime data.
    
    NOTE: Data should be updated via API using 'make update-data', not manually.
    The CSV file is automatically maintained by the update script.
    
    Args:
        file_path: Path to the CSV file (default: "data/Boston Data.csv")
        validate: If True, perform data validation checks
    
    Returns:
        DataFrame with daily crime counts by district
    """
    # Check if data file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Data file '{file_path}' not found. "
            f"Please run 'make update-data' to fetch data from Boston.gov API."
        )
    
    # Load data
    data = pd.read_csv(file_path, encoding='latin', low_memory=False)
    
    # Warn if data appears stale (suggests manual file or outdated)
    if validate:
        latest_date, days_old = check_data_freshness(file_path)
        if days_old and days_old > 30:
            print(f"\n⚠️  DEPRECATION WARNING: Data file appears outdated ({days_old} days old).")
            print("   Data should be updated via API using 'make update-data', not manually.")
            print("   Manual CSV editing is deprecated. Use the automated update script.\n")
    
    # Validate data if requested
    if validate:
        issues = validate_data(data)
        if issues:
            print("Data validation issues found:")
            for issue in issues:
                print(f"  - {issue}")
    
    # Ensure OCCURRED_ON_DATE is datetime
    data['OCCURRED_ON_DATE'] = pd.to_datetime(data['OCCURRED_ON_DATE'], errors='coerce')
    data.dropna(subset=['OCCURRED_ON_DATE'], inplace=True)
    
    # If DISTRICT is missing, fill with 'UNKNOWN'
    if 'DISTRICT' not in data.columns:
        data['DISTRICT'] = 'UNKNOWN'
    data['DISTRICT'] = data['DISTRICT'].fillna('UNKNOWN')
    
    # Aggregate daily crime counts by district
    daily_counts = data.groupby(['DISTRICT', data['OCCURRED_ON_DATE'].dt.date]).size().reset_index()
    daily_counts.columns = ['DISTRICT', 'DATE', 'COUNT']
    
    return daily_counts
