import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load the Parking Transactions data
transactions = pd.read_csv('Parking Transactions from 2023-01-01.csv')

# Convert date columns to datetime
transactions['ENTRY_DATETIME'] = pd.to_datetime(transactions['ENTRY_DATE_ONLY'] + ' ' + transactions['ENTRY_TIME_ONLY'])
transactions['EXIT_DATETIME'] = pd.to_datetime(transactions['EXIT_DATE_ONLY'] + ' ' + transactions['EXIT_TIME_ONLY'])

# Calculate parking duration in hours
transactions['PARKING_DURATION'] = (transactions['EXIT_DATETIME'] - transactions['ENTRY_DATETIME']).dt.total_seconds() / 3600

# Group by facility
facility_stats = transactions.groupby('FACILITY_NAME').agg({
    'PARKING_TRANSACTION_UID': 'count',
    'PARKING_DURATION': 'sum',
    'ENTRY_DATETIME': 'min',
    'EXIT_DATETIME': 'max'
}).reset_index()

# Calculate metrics
facility_stats['TOTAL_DAYS'] = (facility_stats['EXIT_DATETIME'] - facility_stats['ENTRY_DATETIME']).dt.total_seconds() / (24 * 3600)
facility_stats['AVG_DAILY_USAGE'] = facility_stats['PARKING_TRANSACTION_UID'] / facility_stats['TOTAL_DAYS']
facility_stats['AVG_PARKING_DURATION'] = facility_stats['PARKING_DURATION'] / facility_stats['PARKING_TRANSACTION_UID']

# Assuming 24/7 operation and 100 spaces per facility for demonstration
# In a real scenario, you would need actual data for available spaces and operating hours
SPACES_PER_FACILITY = 100
HOURS_PER_DAY = 24

facility_stats['TOTAL_AVAILABLE_TIME'] = facility_stats['TOTAL_DAYS'] * SPACES_PER_FACILITY * HOURS_PER_DAY
facility_stats['OCCUPANCY_RATE'] = facility_stats['PARKING_DURATION'] / facility_stats['TOTAL_AVAILABLE_TIME']
facility_stats['TURNOVER_RATE'] = facility_stats['PARKING_TRANSACTION_UID'] / (SPACES_PER_FACILITY * facility_stats['TOTAL_DAYS'])

# Prepare data for clustering
X = facility_stats[['OCCUPANCY_RATE', 'TURNOVER_RATE', 'AVG_DAILY_USAGE', 'AVG_PARKING_DURATION']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform K-means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
facility_stats['CLUSTER'] = kmeans.fit_predict(X_scaled)

# Create the figure
fig = make_subplots(rows=3, cols=2, 
                    subplot_titles=("Occupancy vs Turnover", "Daily Usage vs Parking Duration", "Cluster Summary", "Detailed Explanation"),
                    specs=[[{"type": "scatter"}, {"type": "scatter"}],
                           [{"type": "table", "colspan": 2}, None],
                           [{"type": "table", "colspan": 2}, None]],
                    vertical_spacing=0.1, horizontal_spacing=0.05,
                    row_heights=[0.3, 0.2, 0.5])

# Add scatter plots
for cluster in range(4):
    cluster_data = facility_stats[facility_stats['CLUSTER'] == cluster]
    
    fig.add_trace(go.Scatter(
        x=cluster_data['OCCUPANCY_RATE'],
        y=cluster_data['TURNOVER_RATE'],
        mode='markers',
        marker=dict(size=10),
        name=f'Cluster {cluster}',
        text=cluster_data['FACILITY_NAME'],
        hovertemplate="<b>%{text}</b><br>Occupancy Rate: %{x:.2f}<br>Turnover Rate: %{y:.2f}",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=cluster_data['AVG_DAILY_USAGE'],
        y=cluster_data['AVG_PARKING_DURATION'],
        mode='markers',
        marker=dict(size=10),
        name=f'Cluster {cluster}',
        text=cluster_data['FACILITY_NAME'],
        hovertemplate="<b>%{text}</b><br>Avg Daily Usage: %{x:.2f}<br>Avg Parking Duration: %{y:.2f} hours",
        showlegend=False
    ), row=1, col=2)

# Add table with cluster characteristics
cluster_summary = facility_stats.groupby('CLUSTER').agg({
    'OCCUPANCY_RATE': 'mean',
    'TURNOVER_RATE': 'mean',
    'AVG_DAILY_USAGE': 'mean',
    'AVG_PARKING_DURATION': 'mean',
    'FACILITY_NAME': lambda x: ', '.join(x)
}).reset_index()

fig.add_trace(go.Table(
    header=dict(values=["Cluster", "Avg Occupancy", "Avg Turnover", "Avg Daily Usage", "Avg Duration (hours)", "Facilities"],
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[cluster_summary['CLUSTER'],
                       cluster_summary['OCCUPANCY_RATE'].round(2),
                       cluster_summary['TURNOVER_RATE'].round(2),
                       cluster_summary['AVG_DAILY_USAGE'].round(2),
                       cluster_summary['AVG_PARKING_DURATION'].round(2),
                       cluster_summary['FACILITY_NAME']],
               align='left')
), row=2, col=1)

# Add detailed explanation as a table
detailed_explanation = [
    ["Purpose of Clustering", "Group similar facilities, identify patterns, develop targeted strategies, benchmark performance, inform decision-making."],
    ["Clustering Method", "K-means clustering with 4 clusters based on Occupancy Rate, Turnover Rate, Average Daily Usage, and Average Parking Duration."],
    ["Occupancy Rate (0-1)", "Measures facility utilization. High (>0.7): approaching capacity. Low (<0.3): underutilized. Ideal: 0.7-0.85."],
    ["Turnover Rate", "Indicates parking duration. High values suggest short-term parking, low values indicate longer-term parking."],
    ["Average Daily Usage", "Represents activity level. Higher values indicate busier facilities."],
    ["Average Parking Duration", "Shows typical stay length. Short durations suggest high-turnover, longer durations indicate all-day parking."],
    ["Potential Improvements", "Low occupancy: marketing, pricing adjustments. High occupancy, low turnover: review pricing, implement time limits."],
    ["Limitations", "Assumed 100 spaces per facility and 24/7 operation. Real capacity and hours data would improve accuracy."],
    ["Conclusion", "Clustering provides data-driven insights for optimizing parking operations. Regular analysis can guide improvement efforts."]
]

fig.add_trace(go.Table(
    header=dict(values=["Aspect", "Explanation"],
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=list(zip(*detailed_explanation)),
               align='left',
               height=30)
), row=3, col=1)

# Update layout
fig.update_layout(height=1800, width=1200, title_text="Facility Clustering Analysis")
fig.update_xaxes(title_text="Occupancy Rate", row=1, col=1)
fig.update_yaxes(title_text="Turnover Rate", row=1, col=1)
fig.update_xaxes(title_text="Average Daily Usage", row=1, col=2)
fig.update_yaxes(title_text="Average Parking Duration (hours)", row=1, col=2)

# Add text annotation with calculation methods and observations
calculation_methods = """
Calculation Methods:
1. Occupancy Rate = (Total parked time) / (Total available time)
2. Turnover Rate = (Number of transactions) / (Number of spaces * Number of days)
3. Average Daily Usage = (Total transactions) / (Number of days)
4. Average Parking Duration = (Total parked time) / (Number of transactions)

Note: For this analysis, we assumed 100 spaces per facility and 24/7 operation.
Actual facility capacities and operating hours should be used for more accurate results.
"""

observations = f"""
Observations:
1. Facilities are clustered into 4 groups based on their operational characteristics.
2. Cluster {cluster_summary['OCCUPANCY_RATE'].idxmax()} has the highest average occupancy rate ({cluster_summary['OCCUPANCY_RATE'].max():.2f}),
   suggesting these facilities are most efficiently utilized.
3. Cluster {cluster_summary['TURNOVER_RATE'].idxmax()} shows the highest turnover rate ({cluster_summary['TURNOVER_RATE'].max():.2f}),
   indicating short-term parking is more common in these facilities.
4. Cluster {cluster_summary['AVG_PARKING_DURATION'].idxmax()} has the longest average parking duration ({cluster_summary['AVG_PARKING_DURATION'].max():.2f} hours),
   which might be suitable for long-term or overnight parking.
5. Cluster {cluster_summary['AVG_DAILY_USAGE'].idxmax()} demonstrates the highest average daily usage ({cluster_summary['AVG_DAILY_USAGE'].max():.2f} transactions/day),
   suggesting these are the busiest facilities.
6. There appears to be a trade-off between occupancy rate and turnover rate,
   as facilities with high occupancy tend to have lower turnover and vice versa.
"""

fig.add_annotation(
    xref="paper", yref="paper",
    x=0.5, y=-0.1,
    text=calculation_methods + "\n" + observations,
    showarrow=False,
    align="left",
    font=dict(size=10)
)

# Show the figure
fig.show()

# If you want to save the figure as an HTML file, uncomment the following line:
# fig.write_html("facility_clustering_analysis.html")