from dash import Dash, dcc, html, Output, Input
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import os

# Create Dash app — note server=None
app = Dash(__name__, url_base_pathname="/attendance/",
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

# Data fetching
api_url = "https://sheet2api.com/v1/4rN3UV6RQwat/attendance-data"

def fetch_data():
    response = requests.get(api_url)
    if response.status_code == 200:
        return pd.DataFrame(response.json())
    else:
        raise Exception(f"API error: {response.status_code}")

df = fetch_data()
weeks = [f'Week {i}' for i in range(1, 15)]
df[weeks] = df[weeks].apply(pd.to_numeric, errors='coerce')

# Preprocessed data
average_attendance = df.groupby('Course ID')[weeks].mean().reset_index()
average_absence = df.groupby('Course ID')[weeks].mean().reset_index()
final_absence_avg = df.groupby('Course ID')[weeks].mean().mean(axis=1).reset_index()
final_absence_avg.columns = ['Course ID', 'Final Absence Average']

# App Layout
app.layout = dbc.Container([
    html.H2("Attendance Dashboard", className="text-center my-4"),
    dcc.Dropdown(
        id='chart-selector',
        options=[
            {'label': 'Individual Attendance', 'value': 'individual'},
            {'label': 'Average Attendance (Line)', 'value': 'average_line'},
            {'label': 'Average Absence (Bar)', 'value': 'absence_bar'},
            {'label': 'Final Absence (Pie)', 'value': 'absence_pie'},
            {'label': 'Raw Data Table', 'value': 'table'}
        ],
        value='individual',
        className='mb-4'
    ),
    dcc.Loading(
        dcc.Graph(id='attendance-graph'),
        type="circle"
    )
], fluid=True)

# Callbacks
@app.callback(
    Output('attendance-graph', 'figure'),
    Input('chart-selector', 'value')
)
def update_graph(selected_chart):
    if selected_chart == 'individual':
        fig = go.Figure()
        for course_id in df['Course ID'].unique():
            course_data = df[df['Course ID'] == course_id]
            for _, row in course_data.iterrows():
                fig.add_trace(go.Scatter(
                    x=weeks,
                    y=row[weeks],
                    mode='lines',
                    name=f"{row['Student Name']} ({course_id})"
                ))
        fig.update_layout(title='Individual Attendance Over 14 Weeks', xaxis_title='Week', yaxis_title='Attendance (%)')

    elif selected_chart == 'average_line':
        fig = go.Figure()
        for _, row in average_attendance.iterrows():
            fig.add_trace(go.Scatter(
                x=weeks,
                y=row[weeks],
                mode='lines',
                name=f"Course {row['Course ID']}"
            ))
        fig.update_layout(title='Average Attendance by Course', xaxis_title='Week', yaxis_title='Attendance (%)')

    elif selected_chart == 'absence_bar':
        melted = average_absence.melt(id_vars=['Course ID'], value_vars=weeks, var_name='Week', value_name='Absence Percentage')
        fig = px.bar(
            melted,
            x='Week',
            y='Absence Percentage',
            color='Course ID',
            barmode='group',
            title='Average Absence by Course per Week'
        )

    elif selected_chart == 'absence_pie':
        fig = px.pie(
            final_absence_avg,
            names='Course ID',
            values='Final Absence Average',
            title='Final Absence Average per Course'
        )

    elif selected_chart == 'table':
        table_data = df[['Student ID', 'Student Name', 'Course ID', 'CRN'] + weeks]
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(table_data.columns), fill_color='paleturquoise', align='left'),
            cells=dict(values=[table_data[col] for col in table_data.columns], fill_color='lavender', align='left')
        )])
        fig.update_layout(title='Attendance Data Table')

    return fig


# Run the app

port = int(os.environ.get('PORT', 8050))
app.run_server(debug=True, host='0.0.0.0', port=port)
