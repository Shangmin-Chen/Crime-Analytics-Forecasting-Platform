# CS506 Final Project Report #

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

    This will execute the `forecast_model.py` script, which processes the data, trains the forecasting models, and generates predictions.

4. **Launch the Dashboard:**

    ```bash
    make run_dashboard
    ```

    This runs a Streamlit app (`app.py`) to visualize the crime data, predictions, and insights interactively.

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

The modeling strategy employs the Prophet algorithm, which is a robust and scalable additive time-series forecasting model well-suited for district-level crime data. A unique Prophet model is trained for each district, and utilizes historical crime data from the training period to capture localized crime trends. The model is then configured to incorporate yearly and weekly seasonality, reflecting periodic fluctuations in crime rates, while daily seasonality is excluded to minimize overfitting.

The system uses dynamic date calculation to automatically determine training, testing, and forecasting periods based on the available data range. For each district, the trained model generates forecasts for the testing period (automatically calculated as 15% of available data), enabling an evaluation of its predictive performance. Forecast outputs include predicted crime counts (yhat) along with confidence intervals (yhat_lower and yhat_upper) to quantify uncertainty. The model is further extended to produce forecasts for a two-month future period (starting from the day after the latest available data), providing actionable insights into anticipated crime trends at the district level. By developing separate models for each district, the methodology ensures that localized patterns and temporal dynamics are accurately reflected, enhancing the precision and relevance of the forecasts for policy-making and resource allocation.

**Note:** This project uses Prophet with the `cmdstanpy` backend, which requires CmdStan to be installed. The installation process (`make install`) automatically handles this requirement.

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


3. **Time Series Graph:** This graph displays the historical trends in the selected crime types, providing insights into temporal patterns.

   ![Time Series for Larceny and Shoplifting](images/time-series.png)
   Figure 7. Time series plots (2023-2024 data) decompose the historical crime data to reveal underlying patterns and trends. The graph breaks down the time series data into three components:

   - **Trend:** This shows the long-term increase or decrease in crime counts, reflecting broader changes over the forecast period.
   - **Weekly:** This highlights patterns that repeat on a weekly basis, such as higher or lower crime rates on specific days of the week.
   - **Yearly:** This captures recurring annual patterns, showing how crime counts fluctuate over the course of a year.

    By analyzing these components, analysts can forecast future crime occurrences, understand seasonal influences, and identify long-term shifts in crime dynamics. This temporal analysis is crucial for strategic planning and implementing timely interventions.

   


5. **Clustering Graph:** This map highlights the areas with the highest density of the selected crime types. The top 5 clusters are extracted using DBSCAN. The clustering is ranked from 0 (most dense) to 9 (least dense).

   ![Cluster Density Map](images/density.png)
   Figure 8. Geospatial map (2023-2024 data) that highlights the identified crime hotspots against a contextual basemap.

   In this map, we observe that the area near Newbury Street has the highest cluster density for shoplifting, which aligns with its status as a popular shopping hotspot. Conversely, areas like Mission Hill, which are primarily residential, exhibit much lower density, reflecting the different socio-economic and activity patterns of these locations. This map utilizes basemap providers like Stamen Toner-Lite or OpenStreetMap Mapnik through the contextily library to overlay crime data points accurately onto the map.


# Achieving Our Goals # 
As outlined in our Midterm Report, the primary objective of this project was to predict crime types and occurrences within the Boston Metro Area. By focusing on individual districts, we were able to develop a more detailed understanding of when and where crimes are most likely to occur. These predictions have the potential to support the effective allocation and management of critical city resources, such as law enforcement and public safety initiatives. However, we recognize the importance of incorporating socio-economic factors into our analysis to ensure a more holistic understanding of the underlying drivers of crime. By doing so, we aim to provide deeper insights into the severity and context of criminal activity, enabling city planners and policymakers to better address systemic issues and allocate resources more equitably. Additionally, this approach could serve as a foundation for future studies to explore interventions that reduce crime while addressing its root causes.




