import pandas as pd
import dash
from dash import html, dcc, Input, Output
import plotly.express as px
import os

# Load Excel data
df = pd.read_excel("student_data.xlsx")  # Ensure the correct path

# Extract GPA and Attendance columns dynamically
gpa_columns = [col for col in df.columns if "GPA Semester" in col]
num_semesters = len(gpa_columns)

# Ensure attendance column exists
attendance_column = "Attendance (%)" if "Attendance (%)" in df.columns else None

# Create Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Student Performance Dashboard"

# Layout
app.layout = html.Div([
    html.H1("Student Performance Dashboard", style={'textAlign': 'center'}),

    # Navigation Menu
    dcc.Tabs(id="tabs", value='gpa', children=[
        dcc.Tab(label='GPA Trend', value='gpa'),
        dcc.Tab(label='Attendance Overview', value='attendance'),
    ]),

    # Student Selector (Only for GPA page)
    html.Div(id='student-dropdown-container', children=[
        dcc.Dropdown(
            id='student-selector',
            options=[{'label': name, 'value': name} for name in df['Name'].dropna().unique()],
            value=df['Name'].dropna().iloc[0],
            clearable=False,
            style={'width': '50%', 'margin': 'auto'}
        )
    ], style={'margin-bottom': '20px'}),

    html.Div(id='tab-content')
])

# Callback to switch between tabs
@app.callback(
    Output('tab-content', 'children'),
    Input('tabs', 'value'),
    Input('student-selector', 'value')
)
def render_tab(selected_tab, selected_student):
    if selected_tab == 'gpa':
        return generate_gpa_graph(selected_student)
    elif selected_tab == 'attendance':
        return generate_attendance_graph()

# Function to generate GPA graph and table
def generate_gpa_graph(selected_student):
    student_data = df[df['Name'] == selected_student]
    
    if student_data.empty:
        return html.Div("No GPA data available", style={'textAlign': 'center'})

    gpa_values = student_data[gpa_columns].iloc[0].values.tolist()

    # Create a DataFrame for Plotly (Line Chart)
    plot_df = pd.DataFrame({
        "Semester": list(range(1, num_semesters + 1)),
        "GPA": gpa_values
    })

    fig1 = px.line(
        plot_df, 
        x="Semester", 
        y="GPA",
        labels={'x': 'Semester', 'y': 'GPA'},
        title=f'GPA Trend for {selected_student}'
    )
    fig1.update_layout(yaxis_range=[0, 4], template='plotly_dark')

    # Additional Graph - Bar Chart of GPA per Semester
    fig2 = px.bar(
        plot_df,
        x="Semester",
        y="GPA",
        labels={'x': 'Semester', 'y': 'GPA'},
        title=f'GPA per Semester for {selected_student}',
        color="GPA",
        color_continuous_scale='Viridis'
    )
    fig2.update_layout(template='plotly_dark')

    # Create GPA Table
    table_header = [html.Thead(html.Tr([html.Th("Student")] + [html.Th(f"Semester {i+1}") for i in range(num_semesters)]))]
    table_body = [html.Tbody([
        html.Tr([html.Td(student)] + [html.Td(df[df['Name'] == student][col].values[0] if col in df.columns else 'N/A') for col in gpa_columns])
        for student in df['Name'].unique()
    ])]

    gpa_table = html.Table(table_header + table_body, style={'width': '100%', 'border': '1px solid white', 'margin-top': '20px', 'textAlign': 'center', 'color': 'white'})

    return html.Div([
        dcc.Graph(figure=fig1),
        dcc.Graph(figure=fig2),
        html.H3("GPA Data for All Students", style={'textAlign': 'center', 'marginTop': '20px'}),
        gpa_table
    ])

# Function to generate Attendance graph
def generate_attendance_graph():
    if not attendance_column:
        return html.Div("Attendance data not available", style={'textAlign': 'center'})

    # Filter out students with missing attendance values
    attendance_data = df[['Name', attendance_column]].dropna()

    if attendance_data.empty:
        return html.Div("No attendance data available", style={'textAlign': 'center'})

    # Create a bar chart for attendance
    fig = px.bar(
        attendance_data, 
        x="Name", 
        y=attendance_column,
        labels={'x': 'Student', 'y': 'Attendance (%)'},
        title='Attendance Percentage for All Students',
        color=attendance_column,
        color_continuous_scale='Blues'
    )
    
    # Adjust the y-axis range to reflect attendance (0-15%)
    fig.update_yaxes(range=[0, 15])
    fig.update_layout(template='plotly_dark', xaxis={'categoryorder':'total descending'})

    return dcc.Graph(figure=fig)


port = int(os.environ.get('PORT', 8050))
app.run_server(debug=True, host='0.0.0.0', port=port)