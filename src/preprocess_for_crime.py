# preprocess.py
import pandas as pd
import logging

# Configure logging for preprocess.py
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def preprocess_data(crime_data_file='data/Boston Data.csv', offense_codes_file='data/RMS Offense Codes.xlsx'):
    """
    Loads and preprocesses the crime data by merging with offense types.
    
    NOTE: The crime data CSV file should be updated via API using 'make update-data'.
    Manual editing of the CSV file is deprecated.
    
    Parameters:
        crime_data_file (str): Path to the main crime data CSV file.
        offense_codes_file (str): Path to the offense codes mapping CSV file.
    
    Returns:
        pd.DataFrame: Preprocessed DataFrame with offense types.
    """
    import os
    from datetime import datetime
    
    # Check if file exists
    if not os.path.exists(crime_data_file):
        error_msg = (
            f"Crime data file '{crime_data_file}' not found. "
            f"Please run 'make update-data' to fetch data from Boston.gov API."
        )
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    logging.info(f"Loading main crime data from '{crime_data_file}'...")
    logging.info("NOTE: Data should be updated via 'make update-data', not manually edited.")
    
    try:
        # Load main crime data with parsing 'OCCURRED_ON_DATE' as datetime
        df = pd.read_csv(crime_data_file, low_memory=False, parse_dates=["OCCURRED_ON_DATE"])
    except FileNotFoundError:
        logging.error(f"Crime data file '{crime_data_file}' not found.")
        raise
    except Exception as e:
        logging.error(f"Error reading crime data file: {e}")
        raise
    
    logging.info(f"Loading offense codes mapping from '{offense_codes_file}'...")
    try:
        # Load offense codes mapping
        offense_df = pd.read_excel(offense_codes_file)
    except FileNotFoundError:
        logging.error(f"Offense codes file '{offense_codes_file}' not found.")
        raise
    except Exception as e:
        logging.error(f"Error reading offense codes file: {e}")
        raise
    
    # Ensure 'OFFENSE_CODE' and 'code' are numeric for accurate merging
    logging.info("Converting 'OFFENSE_CODE' and 'code' to numeric types...")
    df['OFFENSE_CODE'] = pd.to_numeric(df['OFFENSE_CODE'], errors='coerce')
    offense_df['CODE'] = pd.to_numeric(offense_df['CODE'], errors='coerce')
    
    # Merge main data with offense codes mapping on 'OFFENSE_CODE' and 'code'
    logging.info("Merging main data with offense codes mapping...")
    df = df.merge(offense_df, left_on='OFFENSE_CODE', right_on='CODE', how='left')
    
    # Drop unnecessary columns post-merge
    columns_to_drop = ["_id", "Location", "code"]
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    logging.info(f"Dropping columns: {existing_columns_to_drop}")
    df = df.drop(columns=existing_columns_to_drop, errors='ignore')
    
    # Drop additional unnecessary columns except 'name' (now renamed to 'OFFENSE_TYPE')
    additional_columns_to_drop = ['REPORTING_AREA', 'SHOOTING', 'UCR_PART']
    existing_additional_columns_to_drop = [col for col in additional_columns_to_drop if col in df.columns]
    logging.info(f"Dropping additional columns: {existing_additional_columns_to_drop}")
    df = df.drop(columns=existing_additional_columns_to_drop, errors='ignore')
    
    # Ensure YEAR, MONTH, HOUR are numeric
    logging.info("Ensuring 'YEAR', 'MONTH', and 'HOUR' are numeric...")
    df['YEAR'] = pd.to_numeric(df['YEAR'], errors='coerce')
    df['MONTH'] = pd.to_numeric(df['MONTH'], errors='coerce')
    df['HOUR'] = pd.to_numeric(df['HOUR'], errors='coerce')
    
    # Convert categorical columns to string types for consistency
    categorical_columns = ['DISTRICT', 'DAY_OF_WEEK', 'STREET']
    logging.info(f"Converting categorical columns to string types: {categorical_columns}")
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    # Create 'DAY_OF_YEAR' from 'OCCURRED_ON_DATE'
    logging.info("Creating 'DAY_OF_YEAR' column...")
    df['DAY_OF_YEAR'] = df['OCCURRED_ON_DATE'].dt.dayofyear
    
    # Convert 'Lat' and 'Long' to numeric, coercing errors to NaN
    logging.info("Converting 'Lat' and 'Long' to numeric types...")
    df['Lat'] = pd.to_numeric(df['Lat'], errors='coerce')
    df['Long'] = pd.to_numeric(df['Long'], errors='coerce')
    
    # Drop rows with missing coordinates
    logging.info("Dropping rows with missing 'Lat' or 'Long' values...")
    df = df.dropna(subset=['Lat', 'Long'])
    
    # Handle timezone information in 'OCCURRED_ON_DATE'
    logging.info("Handling timezone information for 'OCCURRED_ON_DATE'...")
    if df['OCCURRED_ON_DATE'].dt.tz is not None:
        logging.info("'OCCURRED_ON_DATE' has timezone information. Converting to timezone-naive...")
        df['OCCURRED_ON_DATE'] = df['OCCURRED_ON_DATE'].dt.tz_convert(None)
    
    # Alternatively, if 'OCCURRED_ON_DATE' is already timezone-naive, no action is needed
    else:
        logging.info("'OCCURRED_ON_DATE' is already timezone-naive.")
    
    # **Handle Missing Offense Types**
    missing_offense_types = df['NAME'].isnull().sum()
    if missing_offense_types > 0:
        logging.warning(f"{missing_offense_types} records have missing offense types after merging.")
        # Optionally, drop these records or handle them accordingly
        logging.info("Dropping records with missing 'OFFENSE_TYPE'...")
        df = df.dropna(subset=['NAME'])
        logging.info(f"Dropped {missing_offense_types} records with missing offense types.")
    
    # Rename 'name' to 'OFFENSE_TYPE' for consistency in main.py
    logging.info("Renaming 'name' column to 'OFFENSE_TYPE'...")
    df = df.rename(columns={'NAME': 'OFFENSE_TYPE'})
    
    logging.info("Preprocessing completed successfully.")
    return df
