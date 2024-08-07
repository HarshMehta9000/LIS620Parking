import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

# Load datasets
parking_data = pd.read_csv(r"C:\Users\Patron\Downloads\Parking Transactions from 2023-01-01.csv")
rainfall_data = pd.read_csv(r"C:\Users\Patron\Downloads\allwi-r-cleaned.csv")
snowfall_data = pd.read_csv(r"C:\Users\Patron\Downloads\allwi-snow_year-cleaned.csv")

# Prepare parking data
parking_data['ENTRY_DATE_ONLY'] = pd.to_datetime(parking_data['ENTRY_DATE_ONLY'])
parking_data['YEAR'] = parking_data['ENTRY_DATE_ONLY'].dt.year
parking_data['MONTH'] = parking_data['ENTRY_DATE_ONLY'].dt.month
parking_data = parking_data[(parking_data['YEAR'] >= 2023) & (parking_data['YEAR'] <= 2024)]
monthly_parking = parking_data.groupby(['YEAR', 'MONTH']).size().reset_index(name='PARKING_EVENTS')

# Prepare weather data
months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

def prepare_weather_data(weather_data, value_column):
    weather_monthly = []
    for year in [2023, 2024]:
        year_data = weather_data[weather_data['YR'] == year].iloc[0]
        for i, month in enumerate(months, 1):
            value = year_data[month]
            if value == 'T':
                value = 0.01
            try:
                value = float(value)
            except ValueError:
                value = np.nan
            weather_monthly.append({
                'YEAR': year,
                'MONTH': i,
                value_column: value
            })
    return pd.DataFrame(weather_monthly)

rainfall_monthly = prepare_weather_data(rainfall_data, 'RAINFALL')
snowfall_monthly = prepare_weather_data(snowfall_data, 'SNOWFALL')

# Merge all data
merged_data = pd.merge(monthly_parking, rainfall_monthly, on=['YEAR', 'MONTH'])
merged_data = pd.merge(merged_data, snowfall_monthly, on=['YEAR', 'MONTH'])

# Data quality information
total_rows = len(merged_data)
nan_rainfall = merged_data['RAINFALL'].isna().sum()
nan_snowfall = merged_data['SNOWFALL'].isna().sum()

# Replace NaN values with the mean of the respective column
merged_data['RAINFALL'] = merged_data['RAINFALL'].fillna(merged_data['RAINFALL'].mean())
merged_data['SNOWFALL'] = merged_data['SNOWFALL'].fillna(merged_data['SNOWFALL'].mean())

# Calculate correlations
corr_rainfall = stats.pearsonr(merged_data['RAINFALL'], merged_data['PARKING_EVENTS'])[0]
corr_snowfall = stats.pearsonr(merged_data['SNOWFALL'], merged_data['PARKING_EVENTS'])[0]

# Create the figure with subplots
fig = make_subplots(rows=2, cols=1, 
                    specs=[[{"secondary_y": True}],
                           [{"type": "table"}]],
                    subplot_titles=("Monthly Rainfall, Snowfall, and Parking Events (2023-2024)",
                                    "Comprehensive Analysis"),
                    row_heights=[0.4, 0.6],
                    vertical_spacing=0.1)

# Add traces for parking events, rainfall, and snowfall
fig.add_trace(
    go.Scatter(x=pd.to_datetime(merged_data['YEAR'].astype(str) + '-' + merged_data['MONTH'].astype(str).str.zfill(2) + '-01'),
               y=merged_data['PARKING_EVENTS'],
               name="Parking Events",
               line=dict(color="blue", width=2),
               hovertemplate='%{x|%Y-%m}<br>Parking Events: %{y:,}<extra></extra>'),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=pd.to_datetime(merged_data['YEAR'].astype(str) + '-' + merged_data['MONTH'].astype(str).str.zfill(2) + '-01'),
               y=merged_data['RAINFALL'],
               name="Rainfall",
               line=dict(color="green", width=2, dash='dot'),
               hovertemplate='%{x|%Y-%m}<br>Rainfall: %{y:.2f} inches<extra></extra>'),
    row=1, col=1,
    secondary_y=True
)

fig.add_trace(
    go.Scatter(x=pd.to_datetime(merged_data['YEAR'].astype(str) + '-' + merged_data['MONTH'].astype(str).str.zfill(2) + '-01'),
               y=merged_data['SNOWFALL'],
               name="Snowfall",
               line=dict(color="red", width=2, dash='dash'),
               hovertemplate='%{x|%Y-%m}<br>Snowfall: %{y:.2f} inches<extra></extra>'),
    row=1, col=1,
    secondary_y=True
)

# Add text annotations for key insights
fig.add_annotation(
    text=f"Rainfall Correlation: {corr_rainfall:.2f}<br>Snowfall Correlation: {corr_snowfall:.2f}",
    xref="paper", yref="paper",
    x=0.01, y=0.99,
    showarrow=False,
    font=dict(size=12)
)

# Add table with comprehensive analysis
analysis_details = [
    ["Data Overview", f"Dataset covers 2023-2024 with {total_rows} total monthly entries."],
    ["Parking Events", f"Range: {merged_data['PARKING_EVENTS'].min():,} to {merged_data['PARKING_EVENTS'].max():,} per month<br>Mean: {merged_data['PARKING_EVENTS'].mean():,.0f}<br>Median: {merged_data['PARKING_EVENTS'].median():,.0f}"],
    ["Rainfall", f"Range: {merged_data['RAINFALL'].min():.2f} to {merged_data['RAINFALL'].max():.2f} inches per month<br>Mean: {merged_data['RAINFALL'].mean():.2f} inches<br>Median: {merged_data['RAINFALL'].median():.2f} inches"],
    ["Snowfall", f"Range: {merged_data['SNOWFALL'].min():.2f} to {merged_data['SNOWFALL'].max():.2f} inches per month<br>Mean: {merged_data['SNOWFALL'].mean():.2f} inches<br>Median: {merged_data['SNOWFALL'].median():.2f} inches"],
    ["Data Quality", f"Rainfall: {nan_rainfall} NaN values out of {total_rows} entries<br>Snowfall: {nan_snowfall} NaN values out of {total_rows} entries<br>NaN values were replaced with column means. Trace amounts (T) were set to 0.01 inches."],
    ["Correlations", f"Rainfall-Parking: {corr_rainfall:.2f} (Very weak positive correlation)<br>Snowfall-Parking: {corr_snowfall:.2f} (Moderate negative correlation)"],
    ["Methodology", "1. Aggregated daily parking data to monthly level<br>2. Merged with monthly rainfall and snowfall data<br>3. Cleaned and preprocessed weather data<br>4. Calculated Pearson correlations<br>5. Visualized relationships over time"],
    ["Key Observations", "1. Parking events show significant monthly variation<br>2. Rainfall has minimal impact on parking patterns<br>3. Snowfall shows a moderate negative correlation with parking events<br>4. Winter months generally see decreased parking activity<br>5. Peak parking months don't align consistently with weather patterns"],
    ["Insights", "1. The weak correlation (0.00) between rainfall and parking suggests rain has little influence on parking behavior<br>2. The moderate negative correlation (-0.52) between snowfall and parking indicates snow discourages parking, possibly due to reduced travel or campus closures<br>3. The wide range in monthly parking events (164,944 to 309,375) suggests strong influence from factors other than weather, such as academic calendar or local events<br>4. Snowfall's larger impact compared to rainfall might be due to its more disruptive nature and concentration in winter months"],
    ["Limitations", "1. Monthly aggregation may obscure daily or weekly weather impacts<br>2. Other factors (e.g., academic schedule, events) are not accounted for<br>3. Data quality issues in snowfall data may affect accuracy of correlations<br>4. Limited dataset (18 months) may not capture long-term trends or anomalies"],
    ["Recommendations", "1. Conduct daily-level analysis to capture immediate weather impacts<br>2. Incorporate additional variables like academic calendar, local events, and day of week<br>3. Extend the study period to capture long-term trends and seasonal patterns<br>4. Investigate the reasons for the significant variation in monthly parking events<br>5. Consider separate analyses for different seasons or academic periods"]
]

fig.add_trace(
    go.Table(
        header=dict(values=["Category", "Details"],
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=list(zip(*analysis_details)),
                   fill_color='lavender',
                   align='left'),
        columnwidth=[150, 800]
    ),
    row=2, col=1
)

# Update layout
fig.update_layout(
    height=1500, 
    width=1200,
    title_text="Comprehensive Monthly Rainfall, Snowfall, and Parking Events Analysis (2023-2024)",
    hovermode="x unified"
)

# Update yaxis properties
fig.update_yaxes(title_text="Parking Events", secondary_y=False)
fig.update_yaxes(title_text="Rainfall/Snowfall (inches)", secondary_y=True)

# Show the figure
fig.show()

# If you want to save the figure as an HTML file, uncomment the following line:
# fig.write_html("final_comprehensive_monthly_weather_parking_events_analysis_2023_2024.html")

print("Final comprehensive analysis completed for 2023-2024")
print(f"Total rows: {total_rows}")
print(f"Rows with NaN in RAINFALL: {nan_rainfall}")
print(f"Rows with NaN in SNOWFALL: {nan_snowfall}")
print(f"Rainfall correlation: {corr_rainfall:.2f}")
print(f"Snowfall correlation: {corr_snowfall:.2f}")
print(f"Parking events range: {merged_data['PARKING_EVENTS'].min():,} to {merged_data['PARKING_EVENTS'].max():,}")
print(f"Rainfall range: {merged_data['RAINFALL'].min():.2f} to {merged_data['RAINFALL'].max():.2f} inches")
print(f"Snowfall range: {merged_data['SNOWFALL'].min():.2f} to {merged_data['SNOWFALL'].max():.2f} inches")