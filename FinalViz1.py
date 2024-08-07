import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load the T2 Warehouse EntryExit Incident data
t2_data = pd.read_csv('C:/Users/Patron/Downloads/T2_Warehouse_EntryExitIncident_cleaned.csv')

# Convert DATE column to datetime
t2_data['DATE'] = pd.to_datetime(t2_data['DATE'])

# Get date range
start_date = t2_data['DATE'].min().strftime('%Y-%m-%d')
end_date = t2_data['DATE'].max().strftime('%Y-%m-%d')

# Group the data by 'FACILITY_NAME' and 'PARKING_TYPE'
grouped_data = t2_data.groupby(['FACILITY_NAME', 'PARKING_TYPE']).size().unstack(fill_value=0)

# Combine Entry and Exit for each type
grouped_data['Credential'] = grouped_data['Valid Credential Entry'] + grouped_data['Valid Credential Exit']
grouped_data['Transient'] = grouped_data['Valid Transient Entry'] + grouped_data['Valid Transient Exit']

# Calculate total events and sort
grouped_data['Total'] = grouped_data['Credential'] + grouped_data['Transient']
grouped_data = grouped_data.sort_values('Total', ascending=True)

# Create the figure with subplots
fig = make_subplots(
    rows=3, cols=1,
    row_heights=[0.6, 0.2, 0.2],
    specs=[[{"type": "bar"}], [{"type": "table"}], [{"type": "table"}]],
    vertical_spacing=0.05,
    subplot_titles=("", "", "Observations")
)

# Add bar chart
fig.add_trace(go.Bar(
    y=grouped_data.index,
    x=grouped_data['Credential'],
    name='Credential',
    orientation='h',
    marker_color='blue',
    hovertemplate='%{y}<br>Credential: %{x:,}<br>Percentage: %{customdata:.1f}%',
    customdata=grouped_data['Credential'] / grouped_data['Total'] * 100
), row=1, col=1)

fig.add_trace(go.Bar(
    y=grouped_data.index,
    x=grouped_data['Transient'],
    name='Transient',
    orientation='h',
    marker_color='orange',
    hovertemplate='%{y}<br>Transient: %{x:,}<br>Percentage: %{customdata:.1f}%',
    customdata=grouped_data['Transient'] / grouped_data['Total'] * 100
), row=1, col=1)

# Add statistics table
fig.add_trace(go.Table(
    header=dict(values=['Statistic', 'Value'], 
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[
        ['Date Range', 'Total Events', 'Credential Events', 'Transient Events', 'Credential %', 'Transient %',
         'Busiest Facility', 'Avg Events per Facility'],
        [f"{start_date} to {end_date}",
         f"{grouped_data['Total'].sum():,}",
         f"{grouped_data['Credential'].sum():,}",
         f"{grouped_data['Transient'].sum():,}",
         f"{grouped_data['Credential'].sum() / grouped_data['Total'].sum():.1%}",
         f"{grouped_data['Transient'].sum() / grouped_data['Total'].sum():.1%}",
         f"{grouped_data.index[-1]} ({grouped_data['Total'].max():,})",
         f"{grouped_data['Total'].mean():,.0f}"]
    ],
    align='left')
), row=2, col=1)

# Add observations table
observations = [
    f"The data covers parking events from {start_date} to {end_date}.",
    f"The busiest facility ({grouped_data.index[-1]}) handles over {grouped_data['Total'].max():,} parking events.",
    f"Overall, there's a slight preference for credential parking ({grouped_data['Credential'].sum() / grouped_data['Total'].sum():.1%}) over transient parking ({grouped_data['Transient'].sum() / grouped_data['Total'].sum():.1%}).",
    "The distribution of parking events across facilities is highly uneven, with a few facilities handling a majority of the events.",
    "Some facilities show a clear preference for either credential or transient parking, while others have a more balanced mix.",
    f"The average number of events per facility is about {grouped_data['Total'].mean():,.0f}, but this is skewed by the high variability between facilities."
]

fig.add_trace(go.Table(
    header=dict(values=["Observations"],
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[observations],
               align='left')
), row=3, col=1)

# Update layout
fig.update_layout(
    title={
        'text': 'Credential vs Transient Parking by Facility<br><span style="font-size:12px">Credential = Permit Holders, Transient = Visitors</span>',
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    barmode='stack',
    height=1400,
    width=1200,
    xaxis=dict(title='Number of Parking Events'),
    yaxis=dict(title='Facility Name'),
    legend=dict(x=0.85, y=1.0),
    hovermode='closest'
)

# Show the figure
fig.show()

# If you want to save the figure as an HTML file, uncomment the following line:
# fig.write_html("credential_vs_transient_parking_complete.html")
