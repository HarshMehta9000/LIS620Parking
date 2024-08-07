import pandas as pd
import numpy as np
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
from prophet import Prophet

# Load transactions data (assuming this file exists)
transactions = pd.read_csv('Parking Transactions from 2023-01-01.csv')

# Data Preparation
transactions['ENTRY_DATETIME'] = pd.to_datetime(transactions['ENTRY_DATE_ONLY'] + ' ' + transactions['ENTRY_TIME_ONLY'])
transactions['DAY_OF_WEEK'] = transactions['ENTRY_DATETIME'].dt.day_name()
transactions['HOUR_OF_DAY'] = transactions['ENTRY_DATETIME'].dt.hour

# Initialize Dash app
app = Dash(__name__)

app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif', 'padding': '20px'}, children=[
    html.H1('Parking Facility Heatmap Analysis Dashboard'),
    
    # Heatmap of Parking Utilization by Day and Time
    html.Section([
        html.H2('1. Heatmap of Parking Utilization by Day and Time'),
        dcc.Graph(id='heatmap-chart'),
        html.P('This heatmap shows the average parking utilization for each hour of the day and each day of the week. '
               'The X-axis represents the hours of the day, and the Y-axis represents the days of the week. '
               'Darker colors indicate higher utilization levels. '
               'This visualization helps identify peak hours and days for parking demand, allowing for better scheduling of resources and pricing strategies.')
    ]),
    
    # Heatmap of Predicted Parking Utilization for 2025-2026
    html.Section([
        html.H2('2. Predicted Heatmap of Parking Utilization for 2025-2026'),
        dcc.Graph(id='heatmap-forecast-chart'),
        html.P('This heatmap shows the predicted average parking utilization for each hour of the day and each day of the week for the years 2025 and 2026. '
               'The X-axis represents the hours of the day, and the Y-axis represents the days of the week. '
               'Darker colors indicate higher predicted utilization levels. '
               'This visualization helps in planning for future parking demand and optimizing resource allocation.')
    ])
])

@app.callback(
    [Output('heatmap-chart', 'figure'),
     Output('heatmap-forecast-chart', 'figure')],
    [Input('heatmap-chart', 'id')]  # Dummy input to trigger the callback
)
def update_charts(_):
    # Heatmap of Parking Utilization by Day and Time
    heatmap_data = transactions.groupby(['DAY_OF_WEEK', 'HOUR_OF_DAY']).size().unstack(fill_value=0)
    heatmap_chart_fig = px.imshow(heatmap_data, labels=dict(x="Hour of Day", y="Day of Week", color="Parking Events"),
                                  title='Heatmap of Parking Utilization by Day and Time')
    heatmap_chart_fig.update_layout(
        yaxis={'categoryorder':'array', 'categoryarray': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']},
        xaxis=dict(tickmode='linear', tick0=0, dtick=1)
    )

    # Predicted Heatmap of Parking Utilization for 2025-2026
    model_data = transactions.resample('h', on='ENTRY_DATETIME').size().reset_index(name='y')
    model_data.rename(columns={'ENTRY_DATETIME': 'ds'}, inplace=True)
    
    model = Prophet()
    model.fit(model_data)
    
    future = model.make_future_dataframe(periods=2*365*24, freq='h')  # Extend for 2 more years (2025-2026)
    forecast = model.predict(future)
    
    forecast['day_of_week'] = forecast['ds'].dt.day_name()
    forecast['hour_of_day'] = forecast['ds'].dt.hour
    
    # Adjust the forecast to match the scale of the actual data
    scale_factor = heatmap_data.max().max() / forecast['yhat'].max()
    forecast['yhat_scaled'] = forecast['yhat'] * scale_factor
    
    heatmap_forecast_data = forecast.groupby(['day_of_week', 'hour_of_day']).yhat_scaled.mean().unstack(fill_value=0)
    
    heatmap_forecast_fig = px.imshow(heatmap_forecast_data, labels=dict(x="Hour of Day", y="Day of Week", color="Predicted Parking Events"),
                                     title='Predicted Heatmap of Parking Utilization for 2025-2026')
    heatmap_forecast_fig.update_layout(
        yaxis={'categoryorder':'array', 'categoryarray': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']},
        xaxis=dict(tickmode='linear', tick0=0, dtick=1)
    )

    return heatmap_chart_fig, heatmap_forecast_fig

if __name__ == '__main__':
    app.run_server(debug=True)