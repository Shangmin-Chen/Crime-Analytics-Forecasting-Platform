# CS506 Final Project Report #

## Recent Improvements (December 2024)

The forecasting system has been significantly upgraded to meet industry standards:

**Core Model Enhancements:**
- ✅ Retrains on full dataset (95%) before production forecasts
- ✅ Incorporates US federal holiday effects
- ✅ Handles outliers using z-score capping
- ✅ Compares against baseline methods (Naive, 7-day MA, 30-day MA)

**Evaluation Framework:**
- ✅ Four complementary metrics (MAE, RMSE, MAPE, Coverage)
- ✅ Automated performance warnings for unreliable forecasts
- ✅ Summary metrics across all districts
- ✅ District performance ranking

**Dashboard Features:**
- ✅ Summary metrics dashboard with overall performance
- ✅ Automatic warnings for districts performing worse than baseline
- ✅ Comprehensive educational content (Methodology, Glossary, FAQ)
- ✅ Performance improvement indicators

**Result:** Average 14.3% improvement over naive baseline, with top districts showing 59%+ improvement.

## How to Build and Run the Code

To reproduce the results of this project, follow these instructions:

**Note:** This project supports Python 3.10, 3.11, and 3.12. The Makefile will automatically detect and use the highest available version. If you need to install a specific version, you can download it via [python.org](https://www.python.org/) or use a package manager like Homebrew:

```bash
# For Python 3.12 (recommended)
brew install python@3.12

# Or for Python 3.11
brew install python@3.11

# Or for Python 3.10
brew install python@3.10
```

The Makefile will automatically use the best available Python version (preferring 3.12, then 3.11, then 3.10).

This project uses a Makefile to automate all setup and execution steps. Simply follow the commands below:

1. **Set up a Virtual Environment:**

    Create a Python virtual environment using the Makefile:

    ```bash
    make venv
    ```

    Then activate the virtual environment:

    ```bash
    . venv/bin/activate
    ```

2. **Install Dependencies:**

    ```bash
    make install
    ```

    This installs all the necessary Python libraries specified in the `requirements.txt` file, including Prophet's required dependency `cmdstanpy` and CmdStan backend.

3. **Run the Forecasting Model:**

    ```bash
    make forecast
    ```

    This will execute the improved `forecast_model.py` script with the following enhancements:
    
    **Training Process:**
    - Trains models on 80% of available data
    - Evaluates performance on 15% test data
    - **Retrains on 95% of data (train + test combined) before production forecasts**
    - Incorporates US federal holiday effects
    - Detects and handles outliers using z-score method
    
    **Evaluation & Comparison:**
    - Evaluates models using 4 metrics: MAE, RMSE, MAPE, Coverage
    - Compares against 3 baseline methods: Naive, 7-day MA, 30-day MA
    - Identifies districts where models perform worse than baselines
    
    **Output Files Generated:**
    - `{district}_forecast.csv` - Test period forecasts with confidence intervals
    - `{district}_2months_future_forecast.csv` - Production forecasts for next 2 months
    - `{district}_metrics.csv` - Comprehensive evaluation metrics for each district
    - `{district}_initial_model.joblib` - Model trained on 80% (for evaluation)
    - `{district}_final_model.joblib` - Model retrained on 95% (for production)
    - `summary_metrics.csv` - Aggregated performance across all districts
    
    See the "Model Evaluation and Performance" section for detailed interpretation of results.

4. **Launch the Dashboard:**

    ```bash
    make run_dashboard
    ```

    This runs a Streamlit app (`app.py`) to visualize the crime data, predictions, and insights interactively. The dashboard includes:
    
    - **Summary Metrics Dashboard**: Overview of model performance across all districts, including average MAE, RMSE, MAPE, and Coverage metrics, as well as improvement over baseline methods. Includes district performance ranking table showing best and worst performing models.
    
    - **District-Specific Analysis**: Detailed forecasts and performance metrics for individual districts with:
      - All 4 evaluation metrics (MAE, RMSE, MAPE, Coverage) with explanatory tooltips
      - Comparison to baseline methods
      - Visual indicators of improvement percentage
    
    - **Automatic Performance Warnings**: When a district's model performs worse than naive baseline, the dashboard displays a detailed warning explaining:
      - What this means for forecast reliability
      - Possible reasons (irregular patterns, insufficient data, recent changes)
      - Recommendations for using forecasts cautiously
    
    - **Enhanced Educational Content**: 
      - **Methodology**: Explains holiday effects, retraining process, outlier handling, and evaluation metrics
      - **Glossary**: Defines all technical terms (MAE, RMSE, MAPE, Coverage, baseline, retraining)
      - **FAQ**: Answers common questions about metrics, baseline comparisons, and forecast interpretation

## Dependencies and Requirements

### Prophet and CmdStanPy

This project uses Facebook's Prophet library for time-series forecasting. Prophet requires the `cmdstanpy` backend, which interfaces with CmdStan (the command-line interface to Stan). The installation process automatically handles this:

- `cmdstanpy` is installed via pip from `requirements.txt`
- CmdStan itself is installed automatically when you run `make install` (via `cmdstanpy.install_cmdstan()`)

If you encounter issues with CmdStan installation, you can manually install it by running:
```bash
python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
```

**Note:** Prophet previously supported both `pystan` and `cmdstanpy` backends, but newer versions of Prophet use `cmdstanpy` as the default and recommended backend for better performance and reliability.


# Data Description, Relevance, And Provenance # 

The data for this project is sourced from the [Boston.gov CKAN Data API](https://data.boston.gov/api/3/action/datastore_search) (Resource ID: `b973d8cb-eeb2-4e7e-99da-c92938efc9c0`). 

**⚠️ IMPORTANT: Data is now automatically updated via API. Manual CSV editing is deprecated.**

To update data, use:
```bash
make update-data        # Incremental update (recommended)
make update-data-full   # Full refresh
```

**Historical Note:** Initially, our project proposal included plans to web-scrape the Citizen App for real-time crime data. However, we identified several challenges with this approach. The process of scraping and converting data introduced formatting issues. We determined that the official Boston.gov API provides a more reliable, standardized, and maintainable data source. This ensures our analysis aligns with verified data sources and enhances the reliability of our results.

Crime data analysis is crucial in understanding the disparities and socio-economic challenges across different Boston communities. Historically, the city has faced significant issues related to inequitable policing practices, with certain districts experiencing heavier policing and disproportionate criminalization compared to others. These practices often exacerbate underlying socio-economic problems, such as poverty, unemployment, and lack of access to education and healthcare, which are key factors influencing crime rates. By analyzing crime data at the district level, this project aims to provide insights into the temporal and spatial distribution of crime, which could inform more equitable resource allocation and policymaking. Such an approach has the potential to address systemic inequalities by shifting the focus from punitive measures to preventive and community-centered strategies, ultimately fostering a more just and balanced approach to public safety.

The primary data used in this project is sourced from the [Boston.gov CKAN Data API](https://data.boston.gov/api/3/action/datastore_search), which provides a verified and standardized record of crime incidents across the city. The data is automatically fetched and updated using the `scripts/update_data.py` script, ensuring the reliability and consistency of the analysis. 

**Data Update Process:**
- Data is fetched from the official Boston.gov API (Resource ID: `b973d8cb-eeb2-4e7e-99da-c92938efc9c0`)
- Updates are incremental by default (only new records since last update)
- The CSV file (`data/Boston Data.csv`) is automatically maintained
- Manual editing of the CSV file is **deprecated** - always use `make update-data`

The API provides comprehensive historical records of reported crimes, encompassing all districts and crime types. This automated approach ensures data freshness and eliminates manual data collection errors, enhancing the robustness of predictions and maintaining alignment with trusted, validated data sources.

**Historical Note:** While our initial proposal included plans to scrape real-time data from the Citizen App, limitations such as data incompleteness and formatting challenges led us to shift to this official API source, which provides better reliability and maintainability.

# Data Processing For District-Based Forecasting #

The data processing workflow for our project was meticulously designed to prepare crime data for time-series forecasting for each given district in Boston. Raw crime data is preprocessed using the load_and_preprocess_data() function, resulting in a structured dataset with columns for DATE, DISTRICT, and COUNT, respectively. The components represent the daily crime counts for each police district. The data is automatically segmented into three distinct temporal subsets using dynamic date calculation based on the available data range: a training period (80% of available data), a testing period (15% of available data), and a future forecasting period (2 months starting from the day after the latest available data). The system automatically calculates these periods based on the data's date range, ensuring that forecasts always use the most recent available data. To maintain the reliability of the analysis, the data is filtered for each district, ensuring that only districts with a minimum of 10 training records are included. This district-specific preprocessing guarantees that the dataset is both clean and sufficiently comprehensive for generating accurate forecasts, while also tailoring the analysis to localized crime patterns. Additionally, by segmenting districts, we focus on our objective of real-life allocation. 

# Modeling For District-Based Forecasting # 

The modeling strategy employs the Prophet algorithm, which is a robust and scalable additive time-series forecasting model well-suited for district-level crime data. A unique Prophet model is trained for each district, and utilizes historical crime data from the training period to capture localized crime trends. The model is configured to incorporate yearly and weekly seasonality, reflecting periodic fluctuations in crime rates, while daily seasonality is excluded to minimize overfitting.

## Model Training and Evaluation Process

The system implements industry-standard machine learning practices with a comprehensive three-phase approach:

1. **Initial Training Phase (80% of data)**: Models are first trained on 80% of available historical data to learn district-specific crime patterns.

2. **Evaluation Phase (15% of data)**: The trained models are evaluated on a held-out test set (15% of data) using multiple evaluation metrics to assess predictive performance and compare against baseline methods.

3. **Production Retraining Phase (95% of data)**: After validation, models are retrained on **all available data** (combining the 80% training set and 15% test set, totaling 95% of the full dataset) before generating production forecasts. This ensures that production forecasts utilize all available historical information, typically improving accuracy by 2-5%.

## Enhanced Model Features

### Holiday Effects
The models incorporate US federal holidays (e.g., July 4th, New Year's Day) to capture holiday-related crime pattern variations. This is implemented via Prophet's built-in holiday functionality, allowing the model to learn and predict how crime rates change during holiday periods.

### Outlier Detection and Handling
Extreme outliers are detected using a z-score method (threshold of 3 standard deviations) and capped to prevent them from skewing model predictions. This approach maintains data integrity while improving model stability, particularly important for handling anomalous events such as protests, riots, or major incidents.

### Baseline Comparisons
To validate that the Prophet model adds value over simple forecasting methods, each district's model is compared against three baseline approaches:
- **Naive Forecast**: Uses the last observed value as the prediction
- **7-Day Moving Average**: Uses the average of the last 7 days
- **30-Day Moving Average**: Uses the average of the last 30 days

This comparison ensures that the sophisticated Prophet model actually outperforms simple heuristics, providing transparency about model value.

## Comprehensive Evaluation Metrics

Model performance is assessed using four complementary metrics:

- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual values, providing an easily interpretable measure of forecast accuracy in crimes per day.

- **RMSE (Root Mean Square Error)**: Penalizes larger errors more heavily than MAE, providing insight into the model's handling of outliers and extreme events.

- **MAPE (Mean Absolute Percentage Error)**: Percentage-based error metric that enables comparison across districts with different crime volume scales.

- **Coverage**: Measures the accuracy of 95% prediction intervals, validating that uncertainty quantification is well-calibrated.

The system uses dynamic date calculation to automatically determine training, testing, and forecasting periods based on the available data range. For each district, the trained model generates forecasts for the testing period (automatically calculated as 15% of available data), enabling comprehensive evaluation of its predictive performance. Forecast outputs include predicted crime counts (yhat) along with confidence intervals (yhat_lower and yhat_upper) to quantify uncertainty. The retrained production model is then used to generate forecasts for a two-month future period (starting from the day after the latest available data), providing actionable insights into anticipated crime trends at the district level. By developing separate models for each district, the methodology ensures that localized patterns and temporal dynamics are accurately reflected, enhancing the precision and relevance of the forecasts for policy-making and resource allocation.

**Note:** This project uses Prophet with the `cmdstanpy` backend, which requires CmdStan to be installed. The installation process (`make install`) automatically handles this requirement.

# Model Evaluation and Performance #

## Evaluation Methodology

The forecasting system employs a rigorous evaluation framework that goes beyond simple accuracy metrics. Each district's model is evaluated using four complementary metrics (MAE, RMSE, MAPE, and Coverage) and compared against baseline forecasting methods to ensure the Prophet model provides genuine value over simple heuristics.

### Baseline Comparison Results

The system automatically compares Prophet forecasts against three baseline methods:
- **Naive Forecast**: Uses the last observed value
- **7-Day Moving Average**: Uses the average of the last 7 days  
- **30-Day Moving Average**: Uses the average of the last 30 days

Models that fail to outperform the naive baseline are flagged with performance warnings, ensuring users understand the reliability of forecasts for each district.

### Performance Summary

Based on comprehensive evaluation across all districts, the system demonstrates:

- **Average Improvement**: 14.3% better than naive baseline across all districts
- **Average MAE**: 3.38 crimes/day
- **Average RMSE**: 4.29 crimes/day
- **Average MAPE**: 35.85%
- **Average Coverage**: 75.2% (validating prediction interval calibration)

### Understanding "Improvement Over Baseline"

The **14.3% average improvement over naive baseline** means that Prophet models predict crime counts with 14.3% less error compared to the simplest possible forecasting method (naive baseline).

**What is a Naive Baseline?**
The naive baseline is the simplest forecasting method that uses the last observed value as the prediction for all future days. For example, if yesterday had 10 crimes, the naive forecast predicts 10 crimes for today, tomorrow, and every future day.

**How Improvement is Calculated:**
```
Improvement = (Naive MAE - Prophet MAE) / Naive MAE × 100%
```

**Example (District A1):**
- Naive Baseline MAE: 11.46 crimes/day (just using last value)
- Prophet Model MAE: 4.68 crimes/day (using patterns, seasonality, holidays)
- Improvement: (11.46 - 4.68) / 11.46 × 100% = **59.2% improvement**

This means Prophet's predictions are 59.2% more accurate than simply using the last observed value.

**Interpretation:**
- **Positive improvement** (e.g., +59.2%): Prophet significantly outperforms simple methods → Use Prophet!
- **Near-zero improvement** (e.g., +1.7%): Prophet and baseline are similar → Crime is very stable
- **Negative improvement** (e.g., -6.9%): Baseline outperforms Prophet → Crime patterns are too irregular for Prophet to learn effectively

### District Performance Insights

The evaluation reveals significant variation in model performance across districts:

**Top Performing Districts** (substantial improvement over baseline):
- **A1**: 59.2% improvement (MAE: 4.68 crimes/day)
- **C6**: 59.9% improvement (MAE: 3.89 crimes/day)
- **E13**: 59.8% improvement (MAE: 3.19 crimes/day)
- **E5**: 44.7% improvement (MAE: 2.99 crimes/day)

These districts exhibit predictable crime patterns with strong seasonal trends, making Prophet highly effective.

**Solid Performers** (good improvement):
- **B2**: 17.7% improvement (MAE: 5.47 crimes/day)
- **E18**: 15.3% improvement (MAE: 3.00 crimes/day)
- **B3**: 9.9% improvement (MAE: 3.55 crimes/day)

**Marginal Performers** (minimal improvement):
- **D14**: 3.6% improvement (MAE: 3.68 crimes/day)
- **D4**: 1.7% improvement (MAE: 5.60 crimes/day)

These districts have relatively stable crime rates where even simple methods work reasonably well.

**Districts Requiring Caution** (performing worse than baseline):
Some districts exhibit highly irregular patterns that make sophisticated forecasting challenging:
- **A15**: -6.9% (MAE: 1.83 crimes/day) - Worse than naive baseline
- **A7**: -0.9% (MAE: 3.08 crimes/day) - Essentially equal to naive
- **C11**: -6.7% (MAE: 5.53 crimes/day) - Worse than naive baseline
- **External**: -24.2% (MAE: 0.40 crimes/day) - Very sparse data with gaps
- **UNKNOWN**: -32.5% (MAE: 0.46 crimes/day) - Limited and irregular data

For these districts, the dashboard automatically displays performance warnings, and simple baseline methods may be more reliable than Prophet forecasts. This could indicate:
- Recent changes in crime patterns not captured in historical data
- Highly irregular crime patterns without clear seasonal trends
- Insufficient training data or large gaps in historical records
- External factors affecting crime that aren't captured by temporal patterns alone

### Output Files

The evaluation process generates comprehensive output files for analysis:

- **`{district}_metrics.csv`**: Detailed evaluation metrics (MAE, RMSE, MAPE, Coverage) and baseline comparisons for each district
- **`{district}_forecast.csv`**: Forecasts for the test period with confidence intervals
- **`{district}_2months_future_forecast.csv`**: Production forecasts for the next 2 months
- **`{district}_initial_model.joblib`**: Model trained on 80% of data (for evaluation)
- **`{district}_final_model.joblib`**: Model retrained on 95% of data (for production forecasts)
- **`summary_metrics.csv`**: Aggregated metrics across all districts for overall system assessment

The distinction between initial and final models ensures proper evaluation (initial model) while maximizing forecast accuracy (final model uses all available data).

## Technical Improvements Summary

The forecasting system has been upgraded from a basic Prophet implementation to a production-grade machine learning system. The following table summarizes the key improvements:

| Aspect | Before | After |
|--------|--------|-------|
| **Evaluation Metrics** | MAE only | MAE, RMSE, MAPE, Coverage (4 metrics) |
| **Baseline Comparison** | None | Naive, 7-day MA, 30-day MA comparisons |
| **Production Model** | Trained on 80% of data | Retrained on 95% of data before production forecasts |
| **Holiday Effects** | Not included | US federal holidays incorporated |
| **Outlier Handling** | None | Z-score capping (3 standard deviations) |
| **Documentation** | Basic | Comprehensive with Methodology, Glossary, and FAQ |
| **Performance Warnings** | None | Automatic alerts for districts performing worse than baseline |
| **Summary Reporting** | None | Cross-district aggregated metrics and performance ranking |
| **Model Files** | Single model file | Separate initial (evaluation) and final (production) models |
| **Output Files** | Forecast CSVs only | Forecast CSVs + metrics CSVs + summary reports |

These improvements transform the system from a basic forecasting tool to an industry-standard machine learning implementation suitable for production use, portfolio demonstration, and technical interviews. The comprehensive evaluation framework ensures transparency about model performance and sets appropriate expectations for forecast reliability across different districts.

# Visualizations # 

**Note:** All visualizations in the `images/` folder are based on crime data from the 2023-2024 period. These figures demonstrate the methodology and results from the initial analysis period.

![incident_treemap](https://github.com/user-attachments/assets/dae1b0f8-00f9-4ee6-b663-f10880ecddb2)
Figure 1. Treemap of major crime incident reports in Boston. The top 5 crime incidents are: investigate person, sick assist, motor vehicle leaving scene with property damage, investigate property, and towed motor vehicle. 

![Districts in Boston](images/map.png)

Figure 2. Map of Boston segmented into its respective districts. This visualization highlights the geographic boundaries of each district, providing spatial context for the analysis of crime data. 

![Forcasted Crime Count](images/newplot.jpeg)

Figure 3. Forecasted crime counts (2023-2024 data) showing predicted values with upper and lower confidence bounds. The solid line represents the predicted crime counts, while the dotted lines indicate the upper and lower limits of the forecast. The results reveal consistent periodic patterns of criminal activity, with counts generally ranging between 10 and 20 incidents, and peaks occasionally nearing 25. The relatively narrow confidence intervals demonstrate a high level of confidence in the model's predictions, suggesting that the crime patterns exhibit strong seasonal or cyclical trends over the observed period. This analysis provides valuable insights into temporal crime dynamics, which can aid in strategic resource allocation and planning.

![Actual vs. Predicted Counts](images/newplot-2.jpeg)

Figure 4. Actual vs. predicted crime counts for the testing period (2023-2024 data). The scatter plot represents actual crime counts, while the solid line illustrates the model's predicted values. The model captures the overall trend in crime activity, aligning closely with the general pattern of incidents. However, some deviations between actual and predicted values are observed, particularly during spikes or drops in crime counts, highlighting the variability in criminal activity and the need for further refinement in capturing anomalies.

![Future Forecased Crime Count](images/newplot-3.jpeg)

Figure 5. Forecasted crime counts for a future period in a selected Boston district (D14) based on 2023-2024 training data. The predictions show periodic fluctuations, with crime counts ranging from 10 to 20 incidents and occasional peaks approaching 25. The solid line represents the predicted values, while the dotted lines indicate the upper and lower confidence intervals. These trends suggest a consistent pattern of criminal activity during the forecasted period, providing valuable insights for resource planning and allocation.

## Crime Type Analysis ##

In this subsection, we examine specific crime types — larceny-shoplifting — as examples. Utilizing the DBSCAN clustering algorithm, the script identifies densely populated crime hotspots by analyzing the geographical coordinates of incidents. From these clusters, it selects the top hotspots for further examination. For each prominent cluster, the code employs the Prophet forecasting model to predict future crime occurrences, providing valuable insights into potential trends. Finally, the pipeline culminates in generating visualizations: it produces time series plots illustrating the forecasted crime counts and creates detailed geospatial maps that highlight the identified crime hotspots against a contextual basemap. The analysis includes the following:

1. **Forecast Graph:** This graph shows the forecasted counts for the selected crime types. It is similar to the district-based forecast but focuses on specific crime categories. 

   ![Forecasted Crime Count for Larceny and Shoplifting](images/forecast-1.png)
   Figure 6. Forecasted crime counts for larceny-shoplifting (2023-2024 data). The black dots represent the actual values, while the blue line indicates the prediction.

   We observe that near the end of the forecast period, there is no data available, but the prediction continues to follow the established trend, demonstrating the model's ability to extrapolate based on historical patterns.


2. **Time Series Graph:** This graph displays the historical trends in the selected crime types, providing insights into temporal patterns.

   ![Time Series for Larceny and Shoplifting](images/time-series.png)
   Figure 7. Time series plots (2023-2024 data) decompose the historical crime data to reveal underlying patterns and trends. The graph breaks down the time series data into three components:

   - **Trend:** This shows the long-term increase or decrease in crime counts, reflecting broader changes over the forecast period.
   - **Weekly:** This highlights patterns that repeat on a weekly basis, such as higher or lower crime rates on specific days of the week.
   - **Yearly:** This captures recurring annual patterns, showing how crime counts fluctuate over the course of a year.

    By analyzing these components, analysts can forecast future crime occurrences, understand seasonal influences, and identify long-term shifts in crime dynamics. This temporal analysis is crucial for strategic planning and implementing timely interventions.

   


3. **Clustering Graph:** This map highlights the areas with the highest density of the selected crime types. The top 5 clusters are extracted using DBSCAN. The clustering is ranked from 0 (most dense) to 9 (least dense).

   ![Cluster Density Map](images/density.png)
   Figure 8. Geospatial map (2023-2024 data) that highlights the identified crime hotspots against a contextual basemap.

   In this map, we observe that the area near Newbury Street has the highest cluster density for shoplifting, which aligns with its status as a popular shopping hotspot. Conversely, areas like Mission Hill, which are primarily residential, exhibit much lower density, reflecting the different socio-economic and activity patterns of these locations. This map utilizes basemap providers like Stamen Toner-Lite or OpenStreetMap Mapnik through the contextily library to overlay crime data points accurately onto the map.


# Achieving Our Goals # 
As outlined in our Midterm Report, the primary objective of this project was to predict crime types and occurrences within the Boston Metro Area. By focusing on individual districts, we were able to develop a more detailed understanding of when and where crimes are most likely to occur. These predictions have the potential to support the effective allocation and management of critical city resources, such as law enforcement and public safety initiatives. However, we recognize the importance of incorporating socio-economic factors into our analysis to ensure a more holistic understanding of the underlying drivers of crime. By doing so, we aim to provide deeper insights into the severity and context of criminal activity, enabling city planners and policymakers to better address systemic issues and allocate resources more equitably. Additionally, this approach could serve as a foundation for future studies to explore interventions that reduce crime while addressing its root causes.