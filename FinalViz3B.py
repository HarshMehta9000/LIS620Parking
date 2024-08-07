import pandas as pd
import plotly.graph_objects as go

# Load the Parking Transactions data
transactions = pd.read_csv('Parking Transactions from 2023-01-01.csv')

# Convert ENTRY_DATE_ONLY and ENTRY_TIME_ONLY to datetime
transactions['ENTRY_DATETIME'] = pd.to_datetime(transactions['ENTRY_DATE_ONLY'] + ' ' + transactions['ENTRY_TIME_ONLY'])

# Extract day of week and hour of day
transactions['DAY_OF_WEEK'] = transactions['ENTRY_DATETIME'].dt.day_name()
transactions['HOUR_OF_DAY'] = transactions['ENTRY_DATETIME'].dt.hour

# Define the order of days
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Group the data by day of week and hour of day
grouped_data = transactions.groupby(['DAY_OF_WEEK', 'HOUR_OF_DAY', 'FACILITY_NAME']).size().reset_index(name='COUNT')

# Create pivot table for all facilities
pivot_all = grouped_data.pivot_table(values='COUNT', index='DAY_OF_WEEK', columns='HOUR_OF_DAY', aggfunc='sum')
pivot_all = pivot_all.reindex(index=days_order)

# Calculate statistics
date_range = f"{transactions['ENTRY_DATETIME'].min().strftime('%Y-%m-%d')} to {transactions['ENTRY_DATETIME'].max().strftime('%Y-%m-%d')}"
total_transactions = grouped_data['COUNT'].sum()
max_transactions = grouped_data['COUNT'].max()
avg_transactions = grouped_data['COUNT'].mean()

# Create the figure
# Create the figure
fig = go.Figure()

# Add heatmap for all facilities
heatmap = go.Heatmap(
    z=pivot_all.values,
    x=pivot_all.columns,
    y=pivot_all.index,
    colorscale='Viridis',
    colorbar=dict(title='Number of Transactions', titleside='right', tickformat=','),
    hovertemplate='Day: %{y}<br>Hour: %{x}<br>Transactions: %{z:,}<extra></extra>'
)
fig.add_trace(heatmap)

# Update layout
fig.update_layout(
    height=900,  # Further increased height to accommodate annotations
    width=1200,
    title=dict(
        text="Weekly Parking Utilization Analysis (All Facilities)",
        font=dict(size=24, color='#333'),
        y=0.98
    ),
    xaxis=dict(title=dict(text="Hour of Day", font=dict(size=16)), tickfont=dict(size=14), dtick=1),
    yaxis=dict(title=dict(text="Day of Week", font=dict(size=16)), tickfont=dict(size=14)),
    coloraxis_colorbar=dict(title="Number of<br>Transactions", titlefont=dict(size=14), tickfont=dict(size=12)),
    plot_bgcolor='rgba(240,240,240,0.95)',
    paper_bgcolor='rgba(240,240,240,0.95)',
)


# Add dropdown for facility selection
facilities = ['All Facilities'] + sorted(grouped_data['FACILITY_NAME'].unique().tolist())
fig.update_layout(
    updatemenus=[dict(
        buttons=[dict(label=facility, method='update',
                      args=[{'z': [pivot_all.values if facility == 'All Facilities' 
                                   else grouped_data[grouped_data['FACILITY_NAME'] == facility]
                                   .pivot_table(values='COUNT', index='DAY_OF_WEEK', columns='HOUR_OF_DAY', aggfunc='sum')
                                   .reindex(index=days_order).values]}])
                 for facility in facilities],
        direction="down",
        pad={"r": 10, "t": 10},
        showactive=True,
        x=0.05,
        xanchor="left",
        y=1.1,
        yanchor="top",
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor='#888',
        font=dict(size=14)
    )]
)

# Calculate busiest and quietest times
busiest_time = grouped_data.loc[grouped_data['COUNT'].idxmax()]
quietest_time = grouped_data.loc[grouped_data['COUNT'].idxmin()]

# Add annotation for key information
info_text = (f"Date Range: {date_range} | Total Transactions: {total_transactions:,}<br>"
             f"Max Transactions (single hour): {max_transactions:,} | Avg Transactions (per hour per day): {avg_transactions:.2f}")

fig.add_annotation(
    xref="paper", yref="paper",
    x=0.5, y=-0.15,
    text=info_text,
    showarrow=False,
    font=dict(size=14),
    align="center",
    bgcolor="rgba(255,255,255,0.9)",
    bordercolor="#888",
    borderwidth=1,
    borderpad=10,
)

# Add observations below the heatmap
observations = f"""
<b>Key Observations:</b><br>
1. Peak usage: Typical business hours (8 AM - 6 PM) on weekdays<br>
2. Low activity: Early morning hours (1 AM - 5 AM) across all days<br>
3. Weekend pattern differs significantly from weekdays<br>
4. Busiest time: {busiest_time['DAY_OF_WEEK']} at {busiest_time['HOUR_OF_DAY']}:00 ({busiest_time['COUNT']:,} transactions)<br>
5. Quietest time: {quietest_time['DAY_OF_WEEK']} at {quietest_time['HOUR_OF_DAY']}:00 ({quietest_time['COUNT']:,} transactions)
"""

fig.add_annotation(
    xref="paper", yref="paper",
    x=0.5, y=-0.45,
    text=observations,
    showarrow=False,
    font=dict(size=14),
    align="left",
    bgcolor="rgba(255,255,255,0.9)",
    bordercolor="#888",
    borderwidth=1,
    borderpad=10,
)

fig.update_layout(margin=dict(t=80, b=300, l=100, r=50))

# Show the figure
fig.show()

