# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np
import logging
import matplotlib.pyplot as plt
from shapely.geometry import Point
import geopandas as gpd
import contextily as ctx
from sklearn.cluster import DBSCAN
from prophet import Prophet
import json
from datetime import datetime

from preprocess_for_crime import preprocess_data

# Configure logging for the app
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants
CRS_WGS84 = "EPSG:4326"
CRS_WEB_MERCATOR = "EPSG:3857"

# Parameters for Crime-Type Analysis
SAMPLE_SIZE = 10000
RANDOM_STATE = 42
DBSCAN_EPS = 0.0001
DBSCAN_MIN_SAMPLES = 20
TOP_N_CLUSTERS = 5
FORECAST_PERIODS = 30

# Page configuration
st.set_page_config(
    page_title="Boston Crime Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Helper Functions
# ============================================================================

@st.cache_data
def load_preprocessed_data():
    try:
        return preprocess_data()
    except FileNotFoundError as e:
        st.error(f"**Data file not found:** {e}")
        st.info("**How to fix this:** Please ensure the data files are in the 'data' directory. You can update the data by running `make update-data` in your terminal.")
        st.stop()
    except Exception as e:
        st.error(f"**Error loading data:** {e}")
        st.info("**How to fix this:** Please check that your data files are properly formatted and try again. If the problem persists, try running `make update-data` to refresh the data.")
        logging.exception("Error in load_preprocessed_data")
        st.stop()

def load_metadata():
    """Load metadata about the dataset."""
    metadata_file = "data/metadata.json"
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Could not load metadata: {e}")
    return None

def get_data_info():
    """Get information about the dataset for display."""
    metadata = load_metadata()
    if metadata:
        return {
            "last_updated": metadata.get("last_updated", "Unknown"),
            "date_range": metadata.get("date_range", {}),
            "total_records": metadata.get("total_records", "Unknown"),
            "data_source": metadata.get("data_source", "Boston.gov CKAN API"),
            "districts": metadata.get("districts", [])
        }
    
    # Try to get info from data if metadata not available
    try:
        df = load_preprocessed_data()
        if not df.empty:
            date_col = 'OCCURRED_ON_DATE'
            if date_col in df.columns:
                return {
                    "last_updated": "Unknown",
                    "date_range": {
                        "start": str(df[date_col].min().date()) if pd.notna(df[date_col].min()) else "Unknown",
                        "end": str(df[date_col].max().date()) if pd.notna(df[date_col].max()) else "Unknown"
                    },
                    "total_records": len(df),
                    "data_source": "Boston.gov CKAN API",
                    "districts": sorted(df['DISTRICT'].unique().tolist()) if 'DISTRICT' in df.columns else []
                }
    except Exception:
        pass
    
    return None

def get_top_crime_types(df, n=10):
    top_crimes = df['OFFENSE_TYPE'].value_counts().head(n).index.tolist()
    return top_crimes

def load_and_sample_data(df, sample_size=SAMPLE_SIZE, random_state=RANDOM_STATE, crime_type=None):
    if crime_type:
        df = df[df['OFFENSE_TYPE'] == crime_type]
        if df.empty:
            return pd.DataFrame(), np.array([])
    if len(df) < sample_size:
        df_sample = df.copy()
    else:
        df_sample = df.sample(n=sample_size, random_state=random_state)
    coords = df_sample[['Lat', 'Long']].values
    return df_sample, coords

def perform_dbscan(coords, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES):
    if len(coords) == 0:
        return np.array([])
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='haversine')
    coords_rad = np.radians(coords)
    cluster_labels = dbscan.fit_predict(coords_rad)
    return cluster_labels

def get_top_clusters(df, top_n=TOP_N_CLUSTERS):
    cluster_counts = df['cluster'].value_counts().head(top_n)
    top_clusters = cluster_counts.index.tolist()
    return top_clusters, cluster_counts

def forecast_crime_counts(df, cluster_id, periods=FORECAST_PERIODS):
    cluster_data = df[df['cluster'] == cluster_id].copy()
    if cluster_data.empty:
        return None, pd.DataFrame()
    
    cluster_data = cluster_data.rename(columns={'OCCURRED_ON_DATE': 'ds'})
    if cluster_data['ds'].dt.tz is not None:
        cluster_data['ds'] = cluster_data['ds'].dt.tz_convert(None)
    
    daily_counts = cluster_data.set_index('ds').resample('D').size().reset_index(name='y')
    if daily_counts['ds'].dt.tz is not None:
        daily_counts['ds'] = daily_counts['ds'].dt.tz_convert(None)
    
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(daily_counts)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return model, forecast

# ============================================================================
# Visualization Functions
# ============================================================================

def plot_forecast(model, forecast, cluster_id, crime_type=""):
    """Plot forecast with enhanced annotations and explanations."""
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    model.plot(forecast, ax=ax1)
    title = f'Forecast of Daily Crime Counts for Cluster {cluster_id}'
    if crime_type:
        title += f' ({crime_type})'
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Daily Crime Count', fontsize=12)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig1)
    
    st.info("**Understanding this forecast:** The solid line shows predicted crime counts, while the shaded area represents the confidence interval (uncertainty range). The model captures historical patterns to predict future trends.")
    
    # Components plot
    st.subheader("Forecast Components")
    st.markdown("""
    **What are forecast components?** This breakdown shows how the model understands crime patterns:
    - **Trend**: Long-term increases or decreases in crime over time
    - **Weekly**: Patterns that repeat every week (e.g., higher crime on weekends)
    - **Yearly**: Seasonal patterns that repeat annually (e.g., summer vs. winter trends)
    """)
    fig_components = model.plot_components(forecast)
    st.pyplot(fig_components)

def plot_hotspots(gdf, top_clusters):
    """Plot hotspots with enhanced annotations."""
    fig, ax = plt.subplots(figsize=(14, 12))
    minx, miny, maxx, maxy = gdf.total_bounds
    padding = 1000
    ax.set_xlim(minx - padding, maxx + padding)
    ax.set_ylim(miny - padding, maxy + padding)
    try:
        ctx.add_basemap(
            ax,
            crs=gdf.crs.to_string(),
            source=ctx.providers.Stamen.TonerLite,
            zoom=13
        )
    except (Exception, AttributeError, ValueError) as e:
        logging.warning(f"Failed to load Stamen.TonerLite basemap: {e}. Trying OpenStreetMap...")
        try:
            ctx.add_basemap(
                ax,
                crs=gdf.crs.to_string(),
                source=ctx.providers.OpenStreetMap.Mapnik,
                zoom=13
            )
        except (Exception, AttributeError, ValueError) as e2:
            logging.warning(f"Failed to load OpenStreetMap basemap: {e2}. Continuing without basemap.")
    
    gdf_top_clusters = gdf[gdf['cluster'].isin(top_clusters)]
    if gdf_top_clusters.empty:
        st.warning("No cluster data to plot.")
        return
    
    gdf_top_clusters.plot(
        ax=ax,
        column='cluster',
        cmap='tab10',
        markersize=50,
        alpha=0.7,
        legend=True
    )
    ax.set_title("Top Crime Hotspot Clusters", fontsize=14, fontweight='bold')
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)

    # Add explanation
    st.info("""
    **Understanding the hotspot map:** 
    - Each color represents a different cluster (hotspot area)
    - Clusters are ranked by density (number of incidents per area)
    - Higher density areas indicate locations where this crime type occurs more frequently
    - The top 5 clusters are shown, with cluster 0 being the most dense
    """)

# ============================================================================
# Sidebar Content
# ============================================================================

def render_sidebar():
    """Render enhanced sidebar with information panels."""
    with st.sidebar:
        st.title("Navigation")
        mode = st.radio("Select Mode", ("District-Based Forecasts", "Crime-Type Analysis"))
        
        st.divider()
        
        # App Overview
        with st.expander("📖 About This App", expanded=False):
            st.markdown("""
            **Boston Crime Analysis & Forecasting Dashboard**
            
            This application provides:
            - **District-based forecasting** for strategic resource allocation
            - **Crime-type analysis** to identify patterns and hotspots
            
            **Who can benefit:**
            - City planners and policymakers
            - Law enforcement resource managers
            - Researchers studying urban crime patterns
            - Community organizations working on public safety
            
            **Purpose:** Help inform more equitable and data-driven public safety planning.
            """)
        
        # Data Information
        with st.expander("📊 Data Information", expanded=False):
            data_info = get_data_info()
            if data_info:
                st.markdown(f"**Data Source:** {data_info['data_source']}")
                if data_info.get('date_range'):
                    date_range = data_info['date_range']
                    if date_range.get('start') and date_range.get('end'):
                        st.markdown(f"**Coverage Period:** {date_range['start']} to {date_range['end']}")
                if data_info.get('total_records'):
                    st.markdown(f"**Total Records:** {data_info['total_records']:,}")
                if data_info.get('last_updated'):
                    try:
                        last_update = datetime.fromisoformat(data_info['last_updated'].replace('Z', '+00:00'))
                        st.markdown(f"**Last Updated:** {last_update.strftime('%Y-%m-%d %H:%M')}")
                    except:
                        st.markdown(f"**Last Updated:** {data_info['last_updated']}")
                st.markdown("""
                **Data Update:** Run `make update-data` to refresh data from the Boston.gov API.
                
                **Source:** [Boston.gov CKAN Data API](https://data.boston.gov/api/3/action/datastore_search)
                """)
            else:
                st.info("Data information not available. Please ensure data files are present.")
        
        # Methodology
        with st.expander("🔬 Methodology", expanded=False):
            st.markdown("""
            **Forecasting Model: Prophet**
            - Additive time-series model designed for business forecasting
            - Captures yearly and weekly seasonality patterns
            - Provides confidence intervals for uncertainty quantification
            - Well-suited for crime data with periodic patterns
            
            **Clustering Algorithm: DBSCAN**
            - Density-based clustering to identify hotspots
            - Groups nearby crime incidents into clusters
            - Automatically identifies areas with high crime density
            - Parameters: eps=0.0001, min_samples=20
            
            **Data Processing:**
            - Preprocessing merges crime data with offense codes
            - Geographic filtering removes invalid coordinates
            - Temporal aggregation to daily counts for forecasting
            - Dynamic date splitting (80% train, 15% test, 5% buffer)
            """)
        
        # Glossary
        with st.expander("📚 Glossary", expanded=False):
            st.markdown("""
            - **Confidence Intervals**: Range of likely values around the prediction (e.g., 80% confidence that actual value falls within the range)
            - **Seasonality**: Recurring patterns over time (weekly, yearly cycles)
            - **Hotspots**: Geographic areas with high crime density
            - **Clusters**: Groups of nearby crime incidents identified by DBSCAN
            - **Forecast Period**: Future time range for predictions (2 months)
            - **Test Period**: Historical period used to evaluate model accuracy
            """)
        
        # Help & FAQ
        with st.expander("❓ Help & FAQ", expanded=False):
            st.markdown("""
            **How do I use this app?**
            1. Select an analysis mode from above
            2. Choose a district or crime type
            3. Review the forecasts and visualizations
            4. Use the sidebar for methodology details
            
            **What do the forecasts mean?**
            - Forecasts show expected crime counts based on historical patterns
            - Confidence intervals indicate uncertainty
            - Use forecasts for planning, not exact predictions
            
            **How accurate are the forecasts?**
            - Accuracy varies by district and crime type
            - Check the "Actual vs Predicted" comparison
            - Forecasts are more reliable for districts with stable patterns
            
            **How do I update the data?**
            Run `make update-data` in your terminal to fetch the latest data.
            """)
        
        st.divider()
        st.markdown("**💡 Tip:** Expand sections in the sidebar to learn more about the methodology and data.")
    
    return mode

# ============================================================================
# Main Content Sections
# ============================================================================

def render_introduction():
    """Render app introduction and purpose section."""
    st.title("📊 Boston Crime Analysis & Forecasting Dashboard")
    
    st.markdown("""
    ### Welcome
    
    This dashboard provides interactive tools for analyzing and forecasting crime patterns across Boston's police districts. 
    By combining temporal forecasting with spatial clustering, we aim to support data-driven decision-making for public safety 
    resource allocation and planning.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🎯 What This App Does
        
        - **Forecasts crime counts** for each police district using historical patterns
        - **Identifies crime hotspots** through spatial clustering analysis
        - **Analyzes specific crime types** to understand localized patterns
        - **Visualizes trends** over time with confidence intervals
        """)
    
    with col2:
        st.markdown("""
        #### 💡 Why It Matters
        
        - **Resource Allocation**: Help city planners allocate police resources more effectively
        - **Preventive Planning**: Identify patterns to support proactive public safety measures
        - **Data-Driven Decisions**: Provide evidence-based insights for policy making
        - **Community Safety**: Support efforts to address crime at the district level
        """)
    
    st.info("""
    **Data Source**: This app uses crime incident data from the official [Boston.gov CKAN Data API](https://data.boston.gov/api/3/action/datastore_search). 
    Data is automatically updated and covers all reported crime incidents across Boston's police districts.
    """)
    
    st.markdown("---")

def run_district_analysis():
    """Enhanced district-based forecasting analysis."""
    st.header("District-Based Forecasting")
    
    # Introduction
    st.markdown("""
    **What is District-Based Forecasting?**
    
    This analysis provides crime count forecasts for each of Boston's police districts. Each district has its own forecasting model 
    that learns from historical crime patterns specific to that area. This allows for:
    - Understanding district-specific crime trends
    - Planning resource allocation across districts
    - Identifying which districts may need more attention
    - Comparing forecast patterns between different areas
    
    **How Districts Are Defined**: Boston is divided into multiple police districts (A1, A7, A15, B2, B3, C6, C11, D4, D14, E5, E13, E18, 
    plus External and UNKNOWN). Each district represents a geographic area with its own policing jurisdiction.
    """)
    
    # District map
    st.subheader("Boston Police Districts Map")
    try:
        if os.path.exists("images/map.png"):
            st.image("images/map.png", caption="Boston Police District Codes and Their Geographic Locations")
            st.caption("Use this map to identify district locations when reviewing forecasts.")
        else:
            st.warning("District map image not found. Please ensure 'images/map.png' exists.")
    except Exception as e:
        st.warning(f"Could not load district map: {e}")

    forecast_dir = "output/forecasts"
    if not os.path.exists(forecast_dir):
        st.error("""
        **No forecast data available.**
        
        **To generate forecasts:**
        1. Open your terminal
        2. Navigate to the project directory
        3. Run: `make forecast`
        
        This will train forecasting models for each district and generate predictions.
        """)
    else:
        try:
            available_districts = [f.split("_")[0] for f in os.listdir(forecast_dir) if f.endswith("_forecast.csv")]
            available_districts = list(set(available_districts))
            available_districts.sort()

            if available_districts:
                st.markdown("---")
                st.subheader("Select a District to View Forecasts")
                
                district = st.selectbox(
                    "Choose District:",
                    available_districts,
                    help="Select a police district to view its crime forecasts and model performance"
                )
                
                forecast_file = f"{forecast_dir}/{district}_forecast.csv"
                test_results_file = f"{forecast_dir}/{district}_test_results.csv"

                if os.path.exists(forecast_file) and os.path.exists(test_results_file):
                    try:
                        forecast = pd.read_csv(forecast_file, parse_dates=['ds'])
                        test_results = pd.read_csv(test_results_file, parse_dates=['ds'])
                        
                        if forecast.empty or test_results.empty:
                            st.warning(f"""
                            **Forecast data for district {district} is empty.**
                            
                            **To fix this:** Please regenerate forecasts by running `make forecast` in your terminal.
                            """)
                            return

                        # Forecast plot with enhancements
                        st.markdown("---")
                        st.subheader(f"Forecasted Crime Counts for District {district}")
                        
                        st.markdown("""
                        **Understanding this forecast:**
                        - The **solid line** shows predicted daily crime counts
                        - The **dotted lines** above and below represent confidence intervals (uncertainty bounds)
                        - Historical data shows actual crime counts (black dots)
                        - Future forecasts extend beyond the historical data
                        """)
                        
                        try:
                            fig = px.line(
                                forecast, 
                                x='ds', 
                                y='yhat', 
                                title=f'Forecasted Crime Counts for District {district}',
                                labels={'ds': 'Date', 'yhat': 'Predicted Daily Crime Count'}
                            )
                            fig.update_traces(line=dict(width=2))
                            
                            # Add confidence intervals
                            fig.add_scatter(
                                x=forecast['ds'], 
                                y=forecast['yhat_lower'], 
                                mode='lines', 
                                name='Lower Confidence Bound (80%)', 
                                line=dict(dash='dot', color='lightblue'),
                                fill=None
                            )
                            fig.add_scatter(
                                x=forecast['ds'], 
                                y=forecast['yhat_upper'], 
                                mode='lines', 
                                name='Upper Confidence Bound (80%)', 
                                line=dict(dash='dot', color='lightblue'),
                                fill='tonexty',
                                fillcolor='rgba(173, 216, 230, 0.2)'
                            )
                            
                            fig.update_layout(
                                hovermode='x unified',
                                xaxis_title="Date",
                                yaxis_title="Predicted Daily Crime Count",
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.info("""
                            **How to interpret confidence intervals:** The shaded area shows the range where we expect the actual 
                            crime count to fall 80% of the time. A wider interval indicates more uncertainty, while a narrow 
                            interval suggests higher confidence in the prediction.
                            """)
                        except Exception as e:
                            st.error(f"""
                            **Error creating forecast plot:** {e}
                            
                            **What happened:** There was an issue generating the visualization. The data may be corrupted or in an unexpected format.
                            """)
                            logging.exception("Error plotting forecast")

                        # Actual vs predicted with enhancements
                        st.markdown("---")
                        st.subheader("Model Performance: Actual vs Predicted Counts")
                        
                        st.markdown("""
                        **What this comparison shows:**
                        - **Scatter points**: Actual crime counts during the test period
                        - **Blue line**: Model predictions for the same period
                        - **How to read it**: Points close to the line indicate accurate predictions
                        
                        **Test Period Selection**: The test period uses 15% of available historical data that the model 
                        has not seen during training. This allows us to evaluate how well the model generalizes to new data.
                        """)
                        
                        try:
                            test_fig = px.scatter(
                                test_results, 
                                x='ds', 
                                y='y', 
                                title=f'Actual vs Predicted Crime Counts for District {district} (Test Period)',
                                labels={'ds': 'Date', 'y': 'Actual Daily Crime Count'},
                                color_discrete_sequence=['red']
                            )
                            test_fig.add_scatter(
                                x=test_results['ds'], 
                                y=test_results['yhat'], 
                                mode='lines', 
                                name='Predicted Count',
                                line=dict(width=2, color='blue')
                            )
                            test_fig.update_layout(
                                hovermode='x unified',
                                xaxis_title="Date",
                                yaxis_title="Daily Crime Count",
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                            )
                            st.plotly_chart(test_fig, use_container_width=True)
                            
                            # Calculate and display metrics
                            if 'yhat' in test_results.columns and 'y' in test_results.columns:
                                mae = np.mean(np.abs(test_results['y'] - test_results['yhat']))
                                rmse = np.sqrt(np.mean((test_results['y'] - test_results['yhat'])**2))
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Mean Absolute Error (MAE)", f"{mae:.2f}", 
                                              help="Average difference between predicted and actual values. Lower is better.")
                                with col2:
                                    st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}",
                                              help="Measures the magnitude of prediction errors. Lower is better.")
                        except Exception as e:
                            st.error(f"""
                            **Error creating test results plot:** {e}
                            
                            **What happened:** There was an issue generating the performance visualization. Please check the test results data file.
                            """)
                            logging.exception("Error plotting test results")

                        # Future forecast with enhancements
                        future_file = f"output/forecasts/{district}_2months_future_forecast.csv"
                        if os.path.exists(future_file):
                            try:
                                future_forecast = pd.read_csv(future_file, parse_dates=['ds'])
                                if not future_forecast.empty:
                                    st.markdown("---")
                                    st.subheader("Future Forecast: Next 2 Months")
                                    
                                    st.markdown("""
                                    **What is a 2-Month Future Forecast?**
                                    
                                    This forecast extends beyond the historical data to predict crime counts for the next 2 months. 
                                    This allows for:
                                    - **Short-term planning**: Anticipate resource needs in the coming months
                                    - **Proactive measures**: Identify potential increases in crime activity
                                    - **Budget planning**: Inform resource allocation decisions
                                    
                                    **Important Notes:**
                                    - Forecasts become less certain as they extend further into the future
                                    - Unforeseen events (holidays, special events, policy changes) are not captured
                                    - Use forecasts as planning tools, not exact predictions
                                    """)
                                    
                                    fig_future = px.line(
                                        future_forecast, 
                                        x='ds', 
                                        y='yhat', 
                                        title=f'2-Month Future Forecast for District {district}',
                                        labels={'ds': 'Date', 'yhat': 'Predicted Daily Crime Count'}
                                    )
                                    fig_future.update_traces(line=dict(width=2))
                                    fig_future.add_scatter(
                                        x=future_forecast['ds'], 
                                        y=future_forecast['yhat_lower'], 
                                        mode='lines', 
                                        name='Lower Confidence Bound (80%)', 
                                        line=dict(dash='dot', color='lightblue')
                                    )
                                    fig_future.add_scatter(
                                        x=future_forecast['ds'], 
                                        y=future_forecast['yhat_upper'], 
                                        mode='lines', 
                                        name='Upper Confidence Bound (80%)', 
                                        line=dict(dash='dot', color='lightblue'),
                                        fill='tonexty',
                                        fillcolor='rgba(173, 216, 230, 0.2)'
                                    )
                                    fig_future.update_layout(
                                        hovermode='x unified',
                                        xaxis_title="Date",
                                        yaxis_title="Predicted Daily Crime Count",
                                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                                    )
                                    st.plotly_chart(fig_future, use_container_width=True)
                                    
                                    st.warning("""
                                    **Forecast Limitations**: Future forecasts are based on historical patterns and cannot account for:
                                    - Unexpected events or emergencies
                                    - Policy changes or interventions
                                    - Economic or social disruptions
                                    - Seasonal anomalies beyond historical patterns
                                    
                                    Use these forecasts as guidance, not guarantees.
                                    """)
                                else:
                                    st.info("Future forecast file exists but is empty. Please regenerate forecasts.")
                            except Exception as e:
                                st.error(f"""
                                **Error loading future forecast:** {e}
                                
                                **What happened:** The future forecast file could not be read. Please regenerate forecasts.
                                """)
                                logging.exception("Error loading future forecast")
                        else:
                            st.info("""
                            **No future forecast available** for this district. 
                            
                            **To generate future forecasts:** Run `make forecast` in your terminal to generate 2-month future predictions.
                            """)
                        
                    except pd.errors.EmptyDataError:
                        st.error(f"""
                        **Forecast files for district {district} are empty or corrupted.**
                        
                        **To fix this:** Please regenerate forecasts by running `make forecast` in your terminal.
                        """)
                    except Exception as e:
                        st.error(f"""
                        **Error reading forecast data:** {e}
                        
                        **What happened:** There was an issue reading the forecast files. Please check that the files are properly formatted 
                        and try regenerating forecasts with `make forecast`.
                        """)
                        logging.exception("Error reading forecast data")
                else:
                    st.error(f"""
                    **Forecast data files not found for district {district}.**
                    
                    **To generate forecasts:** Run `make forecast` in your terminal to create forecasts for all districts.
                    """)
            else:
                st.warning("""
                **No forecast files found.**
                
                **To generate forecasts:**
                1. Open your terminal
                2. Navigate to the project directory  
                3. Run: `make forecast`
                
                This will create forecast files for all available districts.
                """)
        except Exception as e:
            st.error(f"""
            **Error accessing forecast directory:** {e}
            
            **What happened:** There was an issue reading the forecast directory. Please ensure the 'output/forecasts' directory exists 
            and contains forecast files. If not, run `make forecast` to generate them.
            """)
            logging.exception("Error in run_district_analysis")

def run_crime_type_analysis():
    """Enhanced crime-type analysis with clustering and forecasting."""
    st.header("Crime-Type Analysis")
    
    # Introduction
    st.markdown("""
    **What is Crime-Type Analysis?**
    
    This analysis examines specific crime types to identify spatial patterns and forecast future occurrences. The process involves:
    
    1. **Clustering (DBSCAN)**: Identifies geographic hotspots where crimes of a specific type cluster together
    2. **Forecasting**: Predicts future crime counts for the most dense hotspot areas
    3. **Visualization**: Maps hotspots and shows temporal patterns
    
    **How Clustering Works**: DBSCAN (Density-Based Spatial Clustering of Applications with Noise) groups nearby crime incidents 
    into clusters. Areas with many incidents close together form "hotspots." This helps identify locations where specific crime 
    types are concentrated.
    
    **What Hotspots Represent**: Hotspots are geographic areas where a particular crime type occurs frequently. Understanding these 
    patterns can help target interventions and resource allocation to areas where they'll have the most impact.
    """)
    
    try:
        df = load_preprocessed_data()
        if df.empty:
            st.error("""
            **No data available.**
            
            **To fix this:** Please ensure your data files are in the 'data' directory. You can update the data by running 
            `make update-data` in your terminal.
            """)
            return
        
        st.markdown("---")
        st.subheader("Select a Crime Type to Analyze")
        
        top_10_crimes = get_top_crime_types(df, n=10)
        if not top_10_crimes:
            st.error("""
            **No crime types found in the data.**
            
            **To fix this:** Please check your data files and ensure they contain crime type information.
            """)
            return
        
        # Show crime counts
        crime_counts = df['OFFENSE_TYPE'].value_counts().head(10)
        st.caption(f"**Top 10 crime types by incident count:** {', '.join(crime_counts.index.tolist()[:5])}...")
        
        selected_crime = st.selectbox(
            "Choose a crime type:",
            top_10_crimes,
            help="Select one of the top 10 most common crime types to analyze. The analysis will identify hotspots and forecast trends for this crime type."
        )
        
        # Show sample size for selected crime
        selected_count = len(df[df['OFFENSE_TYPE'] == selected_crime])
        st.info(f"**{selected_crime}**: {selected_count:,} total incidents in the dataset")

        if st.button("Run Crime-Type Analysis", type="primary"):
            with st.spinner("Loading and sampling data..."):
                df_sample, coords = load_and_sample_data(df, crime_type=selected_crime)
            
            if len(coords) == 0:
                st.error(f"""
                **No data found for crime type '{selected_crime}'.**
                
                **What to do:** Please select a different crime type from the dropdown menu.
                """)
                return
            
            st.success(f"Loaded {len(df_sample):,} incidents for analysis")
            
            with st.spinner("Performing clustering analysis to identify hotspots..."):
                cluster_labels = perform_dbscan(coords)
            
            if len(cluster_labels) == 0:
                st.warning("""
                **No clusters identified.**
                
                **Possible reasons:**
                - The selected crime type has incidents spread too far apart
                - There aren't enough incidents to form meaningful clusters
                
                **Try:** Selecting a different crime type with more incidents.
                """)
                return
            
            df_sample['cluster'] = cluster_labels
            df_clustered = df_sample[df_sample['cluster'] != -1].copy()
            noise_count = len(df_sample[df_sample['cluster'] == -1])
            
            if df_clustered.empty:
                st.warning("""
                **No meaningful clusters found.**
                
                All incidents were classified as "noise" (not part of any cluster), which means the incidents are too spread out 
                to form hotspots. This may indicate the crime type occurs randomly across the city rather than in specific locations.
                
                **Try:** Selecting a different crime type that tends to cluster in specific areas.
                """)
                return
            
            st.info(f"Identified {len(df_clustered['cluster'].unique())} clusters. {noise_count} incidents were classified as noise (not in hotspots).")
            
            top_clusters, cluster_counts = get_top_clusters(df_clustered)
            if not top_clusters:
                st.warning("No top clusters identified. Please try a different crime type.")
                return
            
            top_cluster_id = top_clusters[0]
            
            # Show cluster information
            st.markdown("---")
            st.subheader("Hotspot Clusters Identified")
            st.markdown(f"""
            **Top {len(top_clusters)} clusters by density:**
            """)
            for i, cluster_id in enumerate(top_clusters):
                count = cluster_counts[cluster_id]
                st.write(f"{i+1}. Cluster {cluster_id}: {count:,} incidents")
            
            with st.spinner("Training forecasting model for the top hotspot..."):
                try:
                    model, forecast = forecast_crime_counts(df_clustered, top_cluster_id)
                except Exception as e:
                    st.error(f"""
                    **Error training forecast model:** {e}
                    
                    **What happened:** The cluster may have insufficient data points to train a reliable forecasting model.
                    
                    **Possible solutions:**
                    - Try a different crime type with more incidents
                    - The data may be too sparse for this cluster
                    """)
                    logging.exception("Error in forecast_crime_counts")
                    return
            
            if model is None or forecast.empty:
                st.error("""
                **Unable to forecast due to insufficient data in the top cluster.**
                
                **What to do:** Try selecting a different crime type with more incidents in hotspot areas.
                """)
                return
            
            st.markdown("---")
            st.subheader(f"Forecast for {selected_crime} - Top Hotspot (Cluster {top_cluster_id})")
            st.markdown(f"""
            This forecast shows predicted crime counts for the most dense hotspot area (Cluster {top_cluster_id}) 
            where {cluster_counts[top_cluster_id]:,} incidents occurred.
            """)
            
            try:
                plot_forecast(model, forecast, top_cluster_id, selected_crime)
            except Exception as e:
                st.error(f"""
                **Error plotting forecast:** {e}
                
                **What happened:** There was an issue generating the forecast visualization. Please try again.
                """)
                logging.exception("Error in plot_forecast")
                return
            
            # Plot hotspots
            st.markdown("---")
            st.subheader("Hotspot Clusters Map")
            st.markdown("""
            **Understanding the hotspot map:**
            - Each color represents a different hotspot cluster
            - Clusters are ranked by density (Cluster 0 = most dense)
            - The map shows the geographic locations where this crime type is concentrated
            - Use this to identify areas that may benefit from targeted interventions
            """)
            
            try:
                gdf = gpd.GeoDataFrame(
                    df_clustered,
                    geometry=gpd.points_from_xy(df_clustered.Long, df_clustered.Lat),
                    crs=CRS_WGS84
                ).to_crs(CRS_WEB_MERCATOR)
                plot_hotspots(gdf, top_clusters)
            except Exception as e:
                st.error(f"""
                **Error creating hotspot map:** {e}
                
                **What happened:** There was an issue generating the geographic visualization. Please try again.
                """)
                logging.exception("Error in plot_hotspots")
    except Exception as e:
        st.error(f"""
        **An unexpected error occurred:** {e}
        
        **What happened:** Something went wrong during the analysis. Please try again or check that your data files are correct.
        
        **For help:** See the Help & FAQ section in the sidebar for troubleshooting tips.
        """)
        logging.exception("Error in run_crime_type_analysis")

# ============================================================================
# Main App Logic
# ============================================================================

def main():
    """Main application entry point."""
    # Render sidebar and get mode
    mode = render_sidebar()
    
    # Render introduction
    render_introduction()
    
    # Render appropriate analysis based on mode
    if mode == "District-Based Forecasts":
        run_district_analysis()
    else:
        run_crime_type_analysis()
        
        # Footer
        st.markdown("---")
        st.caption("""
        **Boston Crime Analysis & Forecasting Dashboard** | 
        Data Source: [Boston.gov CKAN Data API](https://data.boston.gov/api/3/action/datastore_search) |
        For questions or issues, refer to the Help & FAQ section in the sidebar.
        """)

if __name__ == "__main__":
    main()
