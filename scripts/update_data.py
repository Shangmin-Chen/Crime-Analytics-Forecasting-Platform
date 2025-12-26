# scripts/update_data.py
"""
Script to update crime data from Boston.gov CKAN API
Resource ID: b973d8cb-eeb2-4e7e-99da-c92938efc9c0
API Docs: https://data.boston.gov/api/3/action/datastore_search
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import json
import logging
import sys

# Add parent directory to path to import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# API Configuration
API_BASE = "https://data.boston.gov/api/3/action/datastore_search"
API_SQL_BASE = "https://data.boston.gov/api/3/action/datastore_search_sql"
RESOURCE_ID = "b973d8cb-eeb2-4e7e-99da-c92938efc9c0"
DATA_FILE = "data/Boston Data.csv"
METADATA_FILE = "data/metadata.json"

def get_latest_date_from_file():
    """Get the latest date from existing data file"""
    if not os.path.exists(DATA_FILE):
        return None
    
    try:
        # Read in chunks to handle large files
        chunk_size = 10000
        latest_date = None
        
        for chunk in pd.read_csv(DATA_FILE, encoding='latin', low_memory=False, chunksize=chunk_size):
            chunk['OCCURRED_ON_DATE'] = pd.to_datetime(chunk['OCCURRED_ON_DATE'], errors='coerce')
            chunk_latest = chunk['OCCURRED_ON_DATE'].max()
            if latest_date is None or (pd.notna(chunk_latest) and chunk_latest > latest_date):
                latest_date = chunk_latest
        
        return latest_date
    except Exception as e:
        logging.warning(f"Could not read existing file: {e}")
        return None

def fetch_crime_data(start_date=None, limit=50000):
    """
    Fetch crime data from Boston.gov API
    
    Args:
        start_date: Only fetch records after this date (datetime or None)
        limit: Maximum number of records to fetch
    
    Returns:
        DataFrame with crime data
    """
    all_records = []
    offset = 0
    batch_size = 5000  # API supports up to 5000 per request
    
    logging.info(f"Fetching crime data from Boston.gov API...")
    if start_date:
        logging.info(f"Fetching records after {start_date.date()}")
        # Use SQL query endpoint for date filtering (more reliable than filters parameter)
        date_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
        # Escape single quotes in resource_id for SQL
        resource_id_escaped = RESOURCE_ID.replace("'", "''")
        
        # Build SQL query with date filter
        sql_query = f'SELECT * FROM "{resource_id_escaped}" WHERE "OCCURRED_ON_DATE" > \'{date_str}\' ORDER BY "OCCURRED_ON_DATE" LIMIT {limit} OFFSET {offset}'
        
        try:
            params = {'sql': sql_query}
            response = requests.get(API_SQL_BASE, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if not data.get('success'):
                error_msg = data.get('error', {}).get('message', 'Unknown error') if isinstance(data.get('error'), dict) else data.get('error', 'Unknown error')
                logging.error(f"API returned success=False: {error_msg}")
                return pd.DataFrame()
            
            records = data.get('result', {}).get('records', [])
            if records:
                all_records.extend(records)
                logging.info(f"Fetched {len(records)} records via SQL query")
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching data via SQL: {e}")
            return pd.DataFrame()
    else:
        # Use regular endpoint for full data fetch (no date filter)
        while len(all_records) < limit:
            params = {
                'resource_id': RESOURCE_ID,
                'limit': min(batch_size, limit - len(all_records)),
                'offset': offset
            }
            
            try:
                response = requests.get(API_BASE, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if not data.get('success'):
                    logging.error(f"API returned success=False: {data.get('error', 'Unknown error')}")
                    break
                
                records = data.get('result', {}).get('records', [])
                
                if not records:
                    logging.info("No more records available")
                    break
                
                all_records.extend(records)
                logging.info(f"Fetched {len(records)} records (total: {len(all_records)})")
                
                # Check if we got fewer records than requested (last page)
                if len(records) < batch_size:
                    break
                
                offset += len(records)
                
                # Small delay to be respectful to API
                import time
                time.sleep(0.5)
                
            except requests.exceptions.RequestException as e:
                logging.error(f"Error fetching data: {e}")
                break
    
    if not all_records:
        logging.warning("No records fetched")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(all_records)
    
    # Standardize column names (API might return different casing)
    column_mapping = {
        'INCIDENT_NUMBER': 'INCIDENT_NUMBER',
        'OFFENSE_CODE': 'OFFENSE_CODE',
        'OFFENSE_DESCRIPTION': 'OFFENSE_DESCRIPTION',
        'DISTRICT': 'DISTRICT',
        'OCCURRED_ON_DATE': 'OCCURRED_ON_DATE',
        'YEAR': 'YEAR',
        'MONTH': 'MONTH',
        'DAY_OF_WEEK': 'DAY_OF_WEEK',
        'HOUR': 'HOUR',
        'STREET': 'STREET',
        'Lat': 'Lat',
        'Long': 'Long',
        'Location': 'Location',
        'REPORTING_AREA': 'REPORTING_AREA',
        'SHOOTING': 'SHOOTING',
        'UCR_PART': 'UCR_PART',
        'OFFENSE_CODE_GROUP': 'OFFENSE_CODE_GROUP'
    }
    
    # Rename columns to match existing format
    for api_col, std_col in column_mapping.items():
        if api_col in df.columns:
            df = df.rename(columns={api_col: std_col})
    
    # Ensure OCCURRED_ON_DATE is datetime
    df['OCCURRED_ON_DATE'] = pd.to_datetime(df['OCCURRED_ON_DATE'], errors='coerce')
    
    logging.info(f"Successfully fetched {len(df)} records")
    return df

def merge_with_existing(new_df, existing_file=DATA_FILE):
    """Merge new data with existing data, removing duplicates"""
    if not os.path.exists(existing_file):
        logging.info("No existing file found, using new data only")
        return new_df
    
    try:
        existing_df = pd.read_csv(existing_file, encoding='latin', low_memory=False)
        logging.info(f"Existing file has {len(existing_df)} records")
        
        # Ensure both have OCCURRED_ON_DATE as datetime
        existing_df['OCCURRED_ON_DATE'] = pd.to_datetime(existing_df['OCCURRED_ON_DATE'], errors='coerce')
        new_df['OCCURRED_ON_DATE'] = pd.to_datetime(new_df['OCCURRED_ON_DATE'], errors='coerce')
        
        # Remove duplicates based on INCIDENT_NUMBER (if available) or combination of date, district, offense
        if 'INCIDENT_NUMBER' in existing_df.columns and 'INCIDENT_NUMBER' in new_df.columns:
            # Remove duplicates from new data that already exist
            existing_incidents = set(existing_df['INCIDENT_NUMBER'].dropna().astype(str))
            new_df = new_df[~new_df['INCIDENT_NUMBER'].astype(str).isin(existing_incidents)]
            logging.info(f"After removing duplicates: {len(new_df)} new records")
        
        # Combine
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        
        # Remove any duplicate rows based on key fields
        key_cols = ['OCCURRED_ON_DATE', 'DISTRICT', 'OFFENSE_CODE']
        if all(col in combined.columns for col in key_cols):
            combined = combined.drop_duplicates(subset=key_cols, keep='last', ignore_index=True)
        
        # Sort by date
        combined = combined.sort_values('OCCURRED_ON_DATE').reset_index(drop=True)
        
        logging.info(f"Combined dataset has {len(combined)} records")
        return combined
        
    except Exception as e:
        logging.error(f"Error merging with existing data: {e}")
        return new_df

def save_metadata(df, metadata_file=METADATA_FILE):
    """Save metadata about the dataset"""
    if df.empty:
        return
    
    dates = pd.to_datetime(df['OCCURRED_ON_DATE'], errors='coerce').dropna()
    
    metadata = {
        'last_updated': datetime.now().isoformat(),
        'date_range': {
            'start': str(dates.min().date()) if not dates.empty else None,
            'end': str(dates.max().date()) if not dates.empty else None,
            'total_days': (dates.max() - dates.min()).days if len(dates) > 1 else 0
        },
        'total_records': len(df),
        'districts': sorted(df['DISTRICT'].dropna().unique().tolist()) if 'DISTRICT' in df.columns else [],
        'data_source': 'Boston.gov CKAN API',
        'resource_id': RESOURCE_ID
    }
    
    os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logging.info(f"Metadata saved to {metadata_file}")

def update_data(incremental=True, force_full=False):
    """
    Update crime data from API
    
    Args:
        incremental: If True, only fetch new records since last update
        force_full: If True, fetch all records (ignores incremental)
    
    Returns:
        bool: True if update was successful
    """
    start_date = None
    
    if incremental and not force_full:
        start_date = get_latest_date_from_file()
        if start_date:
            # Fetch from day after latest date
            start_date = start_date + timedelta(days=1)
            logging.info(f"Incremental update: fetching records after {start_date.date()}")
        else:
            logging.info("No existing data found, fetching all available data")
    
    # Fetch new data
    new_df = fetch_crime_data(start_date=start_date, limit=100000)
    
    if new_df.empty:
        logging.warning("No new data fetched")
        return False
    
    # Merge with existing
    combined_df = merge_with_existing(new_df)
    
    # Save updated data
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    combined_df.to_csv(DATA_FILE, index=False, encoding='latin')
    logging.info(f"Data saved to {DATA_FILE}")
    
    # Save metadata
    save_metadata(combined_df)
    
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Update crime data from Boston.gov API')
    parser.add_argument('--full', action='store_true', 
                       help='Force full data refresh (ignore incremental update)')
    parser.add_argument('--incremental', action='store_true', default=True,
                       help='Only fetch new records (default)')
    
    args = parser.parse_args()
    
    success = update_data(incremental=args.incremental and not args.full, 
                         force_full=args.full)
    
    if success:
        print("✓ Data update completed successfully")
    else:
        print("✗ Data update failed or no new data available")
        sys.exit(1)

