import os
import pandas as pd
import numpy as np
import requests
import plotly.graph_objs as go
import plotly.express as px
import dash
from dash import Dash, dcc, html, Input, Output, State, callback_context, dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import sqlite3
import hashlib


# Initialize database
def init_db():
    """Initialize the authentication database"""
    try:
        conn = sqlite3.connect('auth.db')
        c = conn.cursor()
        
        # Create tables if they don't exist
        c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL
        )
        ''')
        
        c.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')
        
        # Check if we need to create a default admin user
        c.execute("SELECT COUNT(*) FROM users")
        count = c.fetchone()[0]
        
        if count == 0:
            # Create default users for each role
            default_users = [
                ('admin', 'admin', 'admin'),
                ('instructor', 'instructor', 'instructor'),
                ('advisor', 'advisor', 'advisor'),
                ('student', 'student', 'student'),
                ('campus', 'campus', 'campus')
            ]
            
            for username, password, role in default_users:
                # Hash the password
                password_hash = hashlib.sha256(password.encode()).hexdigest()
                
                # Insert user
                c.execute(
                    "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                    (username, password_hash, role)
                )
        
        conn.commit()
        conn.close()
        
        print("Database initialized successfully")
        return True
    
    except Exception as e:
        print(f"Database initialization error: {str(e)}")
        return False
    
def authenticate_user(username, password):
    """Authenticate user with username and password"""
    try:
        conn = sqlite3.connect('auth.db')
        c = conn.cursor()
        
        # Get the user
        c.execute("SELECT id, password_hash, role FROM users WHERE username = ?", (username,))
        result = c.fetchone()
        
        if not result:
            return None
        
        user_id, stored_hash, role = result
        
        # Check password
        input_hash = hashlib.sha256(password.encode()).hexdigest()
        
        if input_hash != stored_hash:
            return None
        
        # Create session token
        token = secrets.token_hex(16)
        
        # Store token
        c.execute("INSERT INTO sessions (token, user_id) VALUES (?, ?)", (token, user_id))
        conn.commit()
        
        conn.close()
        
        return {'token': token, 'role': role}
    
    except Exception as e:
        print(f"Authentication error: {str(e)}")
        return None

def validate_token(token):
    """Validate session token"""
    if not token:
        return False
    
    try:
        conn = sqlite3.connect('auth.db')
        c = conn.cursor()
        
        c.execute("SELECT user_id FROM sessions WHERE token = ?", (token,))
        result = c.fetchone()
        
        conn.close()
        
        return result is not None
    
    except Exception as e:
        print(f"Token validation error: {str(e)}")
        return False

def get_user_role(token):
    """Get user role from token"""
    if not token:
        return None
    
    try:
        conn = sqlite3.connect('auth.db')
        c = conn.cursor()
        
        c.execute("""
        SELECT u.role FROM users u
        JOIN sessions s ON u.id = s.user_id
        WHERE s.token = ?
        """, (token,))
        
        result = c.fetchone()
        
        conn.close()
        
        return result[0] if result else None
    
    except Exception as e:
        print(f"Get user role error: {str(e)}")
        return None
    
def get_login_layout():
    """Return the login page layout"""
    return html.Div([
        dbc.Container([
            html.H1("HCT Dashboard Login", className="text-center mt-5"),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Please Login"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Label("Username", width=12),
                                dbc.Col([
                                    dbc.Input(id="username-input", type="text", placeholder="Enter username")
                                ], width=12)
                            ], className="mb-3"),
                            dbc.Row([
                                dbc.Label("Password", width=12),
                                dbc.Col([
                                    dbc.Input(id="password-input", type="password", placeholder="Enter password")
                                ], width=12)
                            ], className="mb-3"),
                            html.Div(id="login-error", className="text-danger"),
                            dbc.Button("Login", id="login-button", color="primary", className="mt-3")
                        ])
                    ], className="mt-3")
                ], width=6, className="mx-auto")
            ]),
            html.Div([
                html.P("Default users for testing:"),
                html.Ul([
                    html.Li("Admin: username = admin, password = admin"),
                    html.Li("Instructor: username = instructor, password = instructor"),
                    html.Li("Advisor: username = advisor, password = advisor"),
                    html.Li("Student: username = student, password = student"),
                    html.Li("Campus: username = campus, password = campus")
                ]),
            ], className="mt-5 text-center")
        ])
    ])

def get_auth_layout():
    """Return the authenticated app layout with dcc.Store for token"""
    return html.Div([
        dcc.Location(id='url', refresh=False),
        dcc.Store(id='session-token', storage_type='session'),
        html.Div(id='page-content')
    ])

def load_api_data(api_url='https://sheet2api.com/v1/e3H9HQWFR1ao/student_data'):
    """Load data from API endpoint"""
    try:
        # Make API requests for different sheets
        students_response = requests.get(f"{api_url}/Students")
        grades_response = requests.get(f"{api_url}/Grades")
        courses_response = requests.get(f"{api_url}/Courses")
        instructor_courses_response = requests.get(f"{api_url}/Instructor_Courses")
        
        # Convert JSON responses to DataFrames
        students_df = pd.DataFrame(students_response.json())
        grades_df = pd.DataFrame(grades_response.json())
        courses_df = pd.DataFrame(courses_response.json())
        instructor_courses_df = pd.DataFrame(instructor_courses_response.json())

        # Print debugging information
        print("API data loaded successfully")
        print(f"Students: {len(students_df)} records")
        print(f"Grades: {len(grades_df)} records")
        print(f"Courses: {len(courses_df)} records")
        print(f"Instructor Courses: {len(instructor_courses_df)} records")
        
        # Set up global variables
        global students, majors, campuses, advisors, courses, course_credits, instructor_courses
        
        # Extract unique values
        students = students_df['Name'].tolist()
        majors = students_df['Major'].unique().tolist()
        campuses = students_df['Campus'].unique().tolist()
        advisors = students_df['Advisor'].unique().tolist()
        
        # Ensure numeric types for grade data
        for col in ['Test', 'Midterm', 'Project', 'Final Test', 'Attendance']:
            if col in grades_df.columns:
                grades_df[col] = pd.to_numeric(grades_df[col], errors='coerce')
        
        # Create course credits dictionary
        courses = courses_df['Course Name'].tolist()
        course_credits = dict(zip(courses_df['Course Name'], 
                                 pd.to_numeric(courses_df['Credits'], errors='coerce')))
        
        # Create instructor courses dictionary
        instructor_courses = {}
        for advisor in advisors:
            advisor_courses = instructor_courses_df[instructor_courses_df['Instructor'] == advisor]['Course Name'].tolist()
            instructor_courses[advisor] = advisor_courses
        
        # Initialize grade sheet with student info
        grade_sheet = students_df.set_index('Name')
        
        # Process grades
        unique_semesters = grades_df['Semester'].unique()
        
        for semester in unique_semesters:
            semester_grades = grades_df[grades_df['Semester'] == semester]
            
            # Extract semester number
            try:
                if 'Semester' in semester:
                    semester_num = int(semester.split()[-1])
                elif 'Fall' in semester:
                    # Handle Fall/Spring naming format
                    semester_num = 1
                elif 'Spring' in semester:
                    semester_num = 2
                else:
                    semester_num = 1
            except:
                semester_num = 1  # Default to 1 if format is different
            
            # Calculate average grades per student for this semester
            semester_averages = (
                semester_grades.groupby('Student ID')
                .agg({
                    'Test': 'mean',
                    'Midterm': 'mean',
                    'Project': 'mean',
                    'Final Test': 'mean',
                    'Attendance': 'mean'
                })
            )
            
            # Map Student IDs to Names for proper assignment
            id_to_name = students_df.set_index('ID')['Name']
            semester_averages.index = semester_averages.index.map(lambda x: id_to_name.get(x, x))
            
            # Assign averaged grades to grade sheet
            for assessment in ['Test', 'Midterm', 'Project', 'Final Test']:
                grade_sheet[f'{assessment} Semester {semester_num}'] = grade_sheet.index.map(
                    lambda x: semester_averages.loc[x, assessment] if x in semester_averages.index else 0)
            
            grade_sheet[f'Attendance Semester {semester_num}'] = grade_sheet.index.map(
                lambda x: semester_averages.loc[x, 'Attendance'] if x in semester_averages.index else 0)
            
            # Calculate GPA
            weights = {'Test': 0.20, 'Midterm': 0.25, 'Project': 0.25, 'Final Test': 0.30}
            grade_sheet[f'GPA Semester {semester_num}'] = sum(
                grade_sheet[f'{assessment} Semester {semester_num}'] * weight
                for assessment, weight in weights.items()
            )
            grade_sheet[f'GPA Semester {semester_num}'] = grade_sheet[f'GPA Semester {semester_num}'].apply(
                lambda x: min(4.0, max(2.0, x / 25))
            )
        
        # Fill any missing values
        grade_sheet = grade_sheet.fillna(0)
        
        print("\nGrade sheet processed successfully")
        
        return grade_sheet
    
    except Exception as e:
        print(f"Error loading API data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def detect_at_risk_students(grade_sheet):
    """Detect students who are at risk based on their academic performance"""
    at_risk_students = []
    for student in grade_sheet.index:
        gpa_columns = [col for col in grade_sheet.columns if 'GPA Semester' in col]
        if not gpa_columns:
            continue
            
        gpa_trend = [grade_sheet.loc[student, col] for col in gpa_columns]
        
        if any(gpa < 2.5 for gpa in gpa_trend):
            at_risk_students.append((student, grade_sheet.loc[student, 'Major'], 
                                   grade_sheet.loc[student, 'Campus'], "Low GPA"))
        elif len(gpa_trend) >= 2 and np.polyfit(range(len(gpa_trend)), gpa_trend, 1)[0] < -0.2:
            at_risk_students.append((student, grade_sheet.loc[student, 'Major'], 
                                   grade_sheet.loc[student, 'Campus'], "Downward Trend"))
        elif len(gpa_trend) >= 2 and any(gpa_trend[i] - gpa_trend[i-1] < -0.5 for i in range(1, len(gpa_trend))):
            at_risk_students.append((student, grade_sheet.loc[student, 'Major'], 
                                   grade_sheet.loc[student, 'Campus'], "Sudden Drop"))
    
    if not at_risk_students:
        return pd.DataFrame(columns=['Student', 'Major', 'Campus', 'Reason'])
    
    return pd.DataFrame(at_risk_students, columns=['Student', 'Major', 'Campus', 'Reason'])

# Helper function for GPA calculation
def grade_to_gpa(grade):
    """Convert percentage grade to GPA scale"""
    if grade >= 93:
        return 4.0
    elif grade >= 90:
        return 3.7
    elif grade >= 87:
        return 3.3
    elif grade >= 83:
        return 3.0
    elif grade >= 80:
        return 2.7
    elif grade >= 77:
        return 2.3
    elif grade >= 73:
        return 2.0
    elif grade >= 70:
        return 1.7
    elif grade >= 67:
        return 1.3
    elif grade >= 63:
        return 1.0
    elif grade >= 60:
        return 0.7
    else:
        return 0.0

# Load data from API
try:
    grade_sheet = load_api_data(api_url='https://sheet2api.com/v1/e3H9HQWFR1ao/student_data')
    if grade_sheet is None:
        raise Exception("Failed to load data from API")
except Exception as e:
    print(f"Error initializing data: {str(e)}")
    # Create an empty grade sheet as fallback
    grade_sheet = pd.DataFrame()
    students = []
    majors = []
    campuses = []
    advisors = []
    courses = []
    course_credits = {}
    instructor_courses = {}

# Initialize the Dash app with suppress_callback_exceptions
app = Dash(__name__, url_base_pathname='/ats_app/')

app.title = "HCT Dashboard"

# Set the app layout
app.layout = get_auth_layout()

# Setup authentication callbacks
@app.callback(
    [Output('session-token', 'data'),
     Output('login-error', 'children')],
    [Input('login-button', 'n_clicks')],
    [State('username-input', 'value'),
     State('password-input', 'value')]
)
def login_callback(n_clicks, username, password):
    if n_clicks is None:
        return None, ""
    
    if not username or not password:
        return None, "Please enter both username and password"
    
    auth_result = authenticate_user(username, password)
    
    if auth_result:
        return auth_result, ""
    else:
        return None, "Invalid username or password"

@app.callback(
    Output('url', 'pathname'),
    [Input('session-token', 'data')],
    [State('url', 'pathname')]
)
def update_url(token, current_path):
    if token and current_path == '/login':
        return '/'
    return current_path

@app.callback(
    Output('session-token', 'clear_data'),
    [Input('logout-link', 'n_clicks')]
)
def logout(n_clicks):
    if n_clicks:
        return True
    return False

# Page content callback
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname'),
     Input('session-token', 'data')],
    prevent_initial_call=False
)
def display_page(pathname, token):
    try:
        # Debug info
        print(f"Page content callback triggered - Path: {pathname}, Token: {token}")
        ctx = callback_context
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'No triggers'
        print(f"Callback triggered by: {triggered_id}")
        
        # If no token or token is invalid, show login page (except for login page itself)
        if pathname == '/login':
            return get_login_layout()
            
        if token is None:
            return get_login_layout()
            
        is_valid = validate_token(token.get('token', None))
        if not is_valid:
            return get_login_layout()
        
        # User is authenticated, get role and show appropriate content
        user_role = get_user_role(token.get('token', None))
        if user_role:
            return html.Div([
                dbc.NavbarSimple([
                    dbc.NavItem(dbc.NavLink("Logout", href="/login", id="logout-link"))
                ], brand="HCT Dashboard", color="primary", dark=True),
                html.Div([
                    dcc.Dropdown(
                        id='role-selector',
                        options=[{'label': r.title(), 'value': r} 
                                for r in ['admin', 'instructor', 'advisor', 'student', 'campus']
                                if r == user_role or user_role == 'admin'],
                        value=user_role,
                        className="mb-4"
                    ),
                    html.Div(id='role-content')
                ], className="container mt-4")
            ])
        
        # If we get here, something is wrong with the token or role
        return get_login_layout()
        
    except Exception as e:
        import traceback
        print(f"Error in display_page callback: {str(e)}")
        traceback.print_exc()
        return html.Div([
            dbc.Alert(f"An error occurred: {str(e)}", color="danger"),
            html.A("Return to Login", href="/login", className="btn btn-primary")
        ], className="container mt-5")
    
@app.callback(
    Output('role-content', 'children'),
    [Input('role-selector', 'value')]
)
def display_role_content(role):
    if role is None:
        raise PreventUpdate
        
    try:
        print(f"Loading role content for: {role}")
        
        if role == 'admin':
            return html.Div([
        dbc.Row([
            dbc.Col([
                html.H4("Comparative Analytics", className="text-center"),
                dcc.Dropdown(
                    id='comparative-analytics',
                    options=[
                        {'label': 'Campus Performance Benchmarking', 'value': 'campus_bench'},
                        {'label': 'Major-wise Trending', 'value': 'major_trend'},
                        {'label': 'Cross-semester Performance', 'value': 'cross_sem'}
                    ],
                    value='campus_bench',
                    className="mb-3"
                )

            ], width=12)
        ], className="mb-4"),
        dbc.Row([
            dbc.Col([
                html.H4("Campus Comparison", className="text-center"),
                dcc.Graph(id='admin-campus-comparison')
            ], width=12)
        ], className="mb-4"),
        dbc.Row([
            dbc.Col([dcc.Graph(id='admin-at-risk-by-campus')], width=6),
            dbc.Col([dcc.Graph(id='admin-avg-gpa-trend')], width=6)
        ]),
        dbc.Row([
            dbc.Col([dcc.Graph(id='admin-major-distribution')], width=6),
            dbc.Col([dcc.Graph(id='admin-advisor-workload')], width=6)
        ]),
        dbc.Row([
            dbc.Col([
                html.H4("Student-Advisor Assignments", className="text-center"),
                dash_table.DataTable(id='admin-student-advisor-table')
            ], width=12)
        ])
    ])
        elif role == 'instructor':
            return html.Div([
                dcc.Dropdown(
                    id='instructor-selector',
                    options=[{'label': instructor, 'value': instructor} for instructor in advisors],
                    value=advisors[0] if advisors else None,
                    className="mb-4"
                ),
                dbc.Row([
                    dbc.Col([dcc.Graph(id='instructor-gpa-line-graph')], width=12)
                ]),
                dbc.Row([
                    dbc.Col([
                        html.H4("My Students", className="text-center"),
                        dash_table.DataTable(id='instructor-students-table')
                    ], width=12)
                ]),
                dbc.Row([
                    dbc.Col([
                        html.H4("At-Risk Students", className="text-center"),
                        dash_table.DataTable(id='instructor-at-risk-table')
                    ], width=12)
                ])
            ])
        elif role == 'advisor':
            return html.Div([
                dcc.Dropdown(
                    id='advisor-selector',
                    options=[{'label': advisor, 'value': advisor} for advisor in advisors],
                    value=advisors[0] if advisors else None,
                    className="mb-4"
                ),
                dbc.Row([
                    dbc.Col([dcc.Graph(id='advisor-student-gpa-trend')], width=12)
                ]),
                dbc.Row([
                    dbc.Col([
                        html.H4("My Students", className="text-center"),
                        dash_table.DataTable(id='advisor-students-table')
                    ], width=12)
                ]),
                dbc.Row([
                    dbc.Col([
                        html.H4("My Courses", className="text-center"),
                        html.Div(id='advisor-courses')
                    ], width=12)
                ])
            ])
        elif role == 'student':
            return html.Div([
                dcc.Dropdown(
                    id='student-selector',
                    options=[{'label': student, 'value': student} for student in students],
                    value=students[0] if students else None,
                    className="mb-4"
                ),
                dbc.Row([
                    dbc.Col([dcc.Graph(id='student-gpa-trend')], width=6),
                    dbc.Col([
                        html.H4("Student Information", className="text-center"),
                        html.Div(id='student-info')
                    ], width=6)
                ]),
                dbc.Row([
                    dbc.Col([
                        html.H4("Grade Sheet", className="text-center"),
                        dash_table.DataTable(id='student-grade-sheet')
                    ], width=12)
                ]),
                dbc.Row([
                    dbc.Col([
                        html.H4("GPA Calculator", className="text-center mt-4"),
                        html.Div([
                            dcc.Dropdown(
                                id='semester-selector',
                                options=[{'label': f'Semester {i}', 'value': i} for i in range(1, 5)],
                                value=1,
                                className="mb-3"
                            ),
                            dash_table.DataTable(
                                id='gpa-calculator-table',
                                columns=[
                                    {'name': 'Course', 'id': 'course', 'type': 'text'},
                                    {'name': 'Credits', 'id': 'credits', 'type': 'numeric'},
                                    {'name': 'Grade', 'id': 'grade', 'type': 'numeric'}
                                ],
                                data=[],
                                editable=True,
                                style_cell={'textAlign': 'center'},
                                style_header={'fontWeight': 'bold'}
                            ),
                            html.Div([
                                html.H5("Calculated GPA:", className="mt-3"),
                                html.Div(id='calculated-gpa', className="h4")
                            ], className="mt-3 text-center")
                        ])
                    ], width=12)
                ])
            ])
        elif role == 'campus':
            return html.Div([
                dcc.Dropdown(
                    id='campus-selector',
                    options=[{'label': campus, 'value': campus} for campus in campuses],
                    value=campuses[0] if campuses else None,
                    className="mb-4"
                ),
                dbc.Row([
                    dbc.Col([dcc.Graph(id='campus-major-distribution')], width=6),
                    dbc.Col([dcc.Graph(id='campus-avg-gpa-trend')], width=6)
                ]),
                dbc.Row([
                    dbc.Col([
                        html.H4("Campus Students", className="text-center"),
                        dash_table.DataTable(id='campus-students-table')
                    ], width=12)
                ])
            ])
        else:
            return html.Div([
                html.H3("Unknown role selected"),
                html.P(f"Role: {role}")
            ])
            
    except Exception as e:
        import traceback
        print(f"Error in role_content callback: {str(e)}")
        traceback.print_exc()
        return html.Div([
            dbc.Alert(f"Error loading {role} dashboard: {str(e)}", color="danger"),
        ])

@app.callback(
    Output('admin-at-risk-by-campus', 'figure'),
    Input('admin-at-risk-by-campus', 'id')
)
def update_admin_at_risk_by_campus(_):
    try:
        at_risk_df = detect_at_risk_students(grade_sheet)
        if at_risk_df.empty:
            fig = go.Figure()
            fig.update_layout(title="No at-risk students detected")
            return fig
            
        at_risk_by_campus = at_risk_df['Campus'].value_counts()
        fig = px.pie(values=at_risk_by_campus.values, names=at_risk_by_campus.index,
                     title='At-Risk Students by Campus')
        return fig
    except Exception as e:
        print(f"Error in admin-at-risk-by-campus: {str(e)}")
        fig = go.Figure()
        fig.update_layout(title="Error loading at-risk data")
        return fig

@app.callback(
    Output('student-gpa-trend', 'figure'),
    Input('student-selector', 'value')
)
def update_student_gpa_trend(selected_student):
    try:
        if not selected_student or grade_sheet.empty:
            empty_fig = go.Figure()
            empty_fig.update_layout(title="No data available")
            return empty_fig
            
        # GPA Trend
        gpa_columns = [col for col in grade_sheet.columns if 'GPA Semester' in col]
        if not gpa_columns:
            empty_fig = go.Figure()
            empty_fig.update_layout(title="No GPA data available")
            return empty_fig
            
        gpa_columns.sort(key=lambda x: int(x.split()[-1]))
        
        gpa_trend = [grade_sheet.loc[selected_student, col] for col in gpa_columns]
        semester_nums = [int(col.split()[-1]) for col in gpa_columns]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=semester_nums, y=gpa_trend, mode='lines+markers'))
        fig.update_layout(
            title=f'GPA Trend for {selected_student}',
            xaxis_title='Semester',
            yaxis_title='GPA',
            yaxis_range=[2.0, 4.0]
        )
        return fig
    except Exception as e:
        print(f"Error in student-gpa-trend: {str(e)}")
        fig = go.Figure()
        fig.update_layout(title=f"Error: {str(e)}")
        return fig
@app.callback(
    Output('admin-avg-gpa-trend', 'figure'),
    Input('admin-avg-gpa-trend', 'id')
)
def update_admin_avg_gpa_trend(_):
    try:
        if grade_sheet.empty:
            fig = go.Figure()
            fig.update_layout(title="No data available")
            return fig
        
        # Find all GPA columns
        gpa_columns = [col for col in grade_sheet.columns if 'GPA Semester' in col]
        
        if not gpa_columns:
            fig = go.Figure()
            fig.update_layout(title="No GPA data available")
            return fig
        
        # Sort columns by semester number
        gpa_columns.sort(key=lambda x: int(x.split()[-1]))
        
        # Calculate average GPA for each semester
        avg_gpa = grade_sheet[gpa_columns].mean()
        
        # Get semester numbers for x-axis
        semester_nums = [int(col.split()[-1]) for col in gpa_columns]
        
        # Create bar chart
        fig = px.bar(
            x=semester_nums,
            y=avg_gpa.values,
            labels={'x': 'Semester', 'y': 'Average GPA'},
            title='Average GPA Trend Across Semesters'
        )
        
        # Set y-axis range
        fig.update_yaxes(range=[2.0, 4.0])
        
        return fig
        
    except Exception as e:
        print(f"Error in admin-avg-gpa-trend: {str(e)}")
        fig = go.Figure()
        fig.update_layout(title=f"Error: {str(e)}")
        return fig


@app.callback(
    Output('admin-major-distribution', 'figure'),
    Input('admin-major-distribution', 'id')
)
def update_admin_major_distribution(_):
    try:
        if grade_sheet.empty:
            fig = go.Figure()
            fig.update_layout(title="No data available")
            return fig
        
        if 'Major' not in grade_sheet.columns:
            fig = go.Figure()
            fig.update_layout(title="Major data not available")
            return fig
        
        # Get major counts
        major_counts = grade_sheet['Major'].value_counts()
        
        # Create pie chart
        fig = px.pie(
            values=major_counts.values,
            names=major_counts.index,
            title='Student Distribution by Major'
        )
        
        return fig
        
    except Exception as e:
        print(f"Error in admin-major-distribution: {str(e)}")
        fig = go.Figure()
        fig.update_layout(title=f"Error: {str(e)}")
        return fig


@app.callback(
    Output('admin-advisor-workload', 'figure'),
    Input('admin-advisor-workload', 'id')
)
def update_admin_advisor_workload(_):
    try:
        if grade_sheet.empty:
            fig = go.Figure()
            fig.update_layout(title="No data available")
            return fig
        
        if 'Advisor' not in grade_sheet.columns:
            fig = go.Figure()
            fig.update_layout(title="Advisor data not available")
            return fig
        
        # Get advisor counts
        advisor_counts = grade_sheet['Advisor'].value_counts()
        
        # Create bar chart
        fig = px.bar(
            x=advisor_counts.index,
            y=advisor_counts.values,
            title='Advisor Workload',
            labels={'x': 'Advisor', 'y': 'Number of Students'}
        )
        
        return fig
        
    except Exception as e:
        print(f"Error in admin-advisor-workload: {str(e)}")
        fig = go.Figure()
        fig.update_layout(title=f"Error: {str(e)}")
        return fig


@app.callback(
    [Output('admin-student-advisor-table', 'data'),
     Output('admin-student-advisor-table', 'columns')],
    Input('admin-student-advisor-table', 'id')
)
def update_admin_student_advisor_table(_):
    try:
        if grade_sheet.empty:
            empty_data = [{'Message': 'No data available'}]
            empty_columns = [{'name': 'Message', 'id': 'Message'}]
            return empty_data, empty_columns
        
        # Define required columns
        required_columns = ['ID', 'Major', 'Advisor', 'Campus']
        available_columns = [col for col in required_columns if col in grade_sheet.columns]
        
        # Create a copy to avoid modifying the original
        df_copy = grade_sheet.copy()
        
        # Reset index to make the student name a column
        df_with_student = df_copy.reset_index()
        
        # Rename the index column to 'Student' if it's not already named
        if df_with_student.columns[0] != 'Student':
            df_with_student = df_with_student.rename(columns={df_with_student.columns[0]: 'Student'})
        
        # Select columns that exist
        cols_to_use = ['Student'] + available_columns
        
        data = df_with_student[cols_to_use].to_dict('records')
        columns = [{'name': col, 'id': col} for col in cols_to_use]
        
        return data, columns
        
    except Exception as e:
        print(f"Error in admin-student-advisor-table: {str(e)}")
        import traceback
        traceback.print_exc()
        error_data = [{'Error': f'Error: {str(e)}'}]
        error_columns = [{'name': 'Error', 'id': 'Error'}]
        return error_data, error_columns
# Instructor dashboard
@app.callback(
    [Output('instructor-gpa-line-graph', 'figure'),
     Output('instructor-students-table', 'data'),
     Output('instructor-students-table', 'columns'),
     Output('instructor-at-risk-table', 'data'),
     Output('instructor-at-risk-table', 'columns')],
    Input('instructor-selector', 'value')
)
def update_instructor_dashboard(selected_instructor):
    try:
        if not selected_instructor or grade_sheet.empty:
            empty_fig = go.Figure()
            empty_fig.update_layout(title="No data available")
            empty_cols = [{'name': 'No Data', 'id': 'no_data'}]
            empty_data = [{'no_data': 'No data available'}]
            return empty_fig, empty_data, empty_cols, empty_data, empty_cols
        
        # Filter for the selected instructor's students
        instructor_students = grade_sheet[grade_sheet['Advisor'] == selected_instructor] if 'Advisor' in grade_sheet.columns else pd.DataFrame()
        
        if instructor_students.empty:
            empty_fig = go.Figure()
            empty_fig.update_layout(title=f"No students found for {selected_instructor}")
            empty_cols = [{'name': 'No Data', 'id': 'no_data'}]
            empty_data = [{'no_data': f'No students assigned to {selected_instructor}'}]
            return empty_fig, empty_data, empty_cols, empty_data, empty_cols
        
        # GPA Line Graph
        fig = go.Figure()
        gpa_columns = [col for col in instructor_students.columns if 'GPA Semester' in col]
        
        if not gpa_columns:
            fig.update_layout(title=f"No GPA data available for {selected_instructor}'s students")
        else:
            gpa_columns.sort(key=lambda x: int(x.split()[-1]))
            semester_nums = [int(col.split()[-1]) for col in gpa_columns]
            
            for student in instructor_students.index:
                try:
                    gpa_trend = [instructor_students.loc[student, col] for col in gpa_columns]
                    fig.add_trace(go.Scatter(
                        x=semester_nums, 
                        y=gpa_trend, 
                        mode='lines+markers', 
                        name=student
                    ))
                except Exception as e:
                    print(f"Error adding student {student} to graph: {str(e)}")
            
            fig.update_layout(
                title=f'GPA Trends for Students of {selected_instructor}',
                xaxis_title='Semester',
                yaxis_title='GPA',
                yaxis_range=[2.0, 4.0]
            )
        
        # Students Table
        required_columns = ['ID', 'Major', 'Campus']
        available_columns = [col for col in required_columns if col in instructor_students.columns]
        
        # Create a copy to avoid modifying the original
        df_for_table = instructor_students.copy()
        
        # Reset index to make the student name a column
        df_with_student = df_for_table.reset_index()
        
        # Rename the index column to 'Student' if it's not already named
        if df_with_student.columns[0] != 'Student':
            df_with_student = df_with_student.rename(columns={df_with_student.columns[0]: 'Student'})
        
        # Select columns that exist
        cols_to_use = ['Student'] + [col for col in available_columns + gpa_columns if col in df_with_student.columns]
        
        students_data = df_with_student[cols_to_use].to_dict('records')
        students_columns = [{'name': col, 'id': col} for col in cols_to_use]
        
        # At-Risk Table
        try:
            at_risk_df = detect_at_risk_students(instructor_students)
            if at_risk_df.empty:
                at_risk_data = [{'Message': 'No at-risk students detected'}]
                at_risk_columns = [{'name': 'Message', 'id': 'Message'}]
            else:
                at_risk_data = at_risk_df.to_dict('records')
                at_risk_columns = [{'name': col, 'id': col} for col in at_risk_df.columns]
        except Exception as e:
            print(f"Error detecting at-risk students: {str(e)}")
            at_risk_data = [{'Error': str(e)}]
            at_risk_columns = [{'name': 'Error', 'id': 'Error'}]
        
        return fig, students_data, students_columns, at_risk_data, at_risk_columns
        
    except Exception as e:
        print(f"Error in instructor dashboard: {str(e)}")
        import traceback
        traceback.print_exc()
        
        empty_fig = go.Figure()
        empty_fig.update_layout(title=f"Error: {str(e)}")
        error_msg = [{'Error': f'Error: {str(e)}'}]
        error_cols = [{'name': 'Error', 'id': 'Error'}]
        
        return empty_fig, error_msg, error_cols, error_msg, error_cols

# Advisor dashboard
@app.callback(
    [Output('advisor-student-gpa-trend', 'figure'),
     Output('advisor-students-table', 'data'),
     Output('advisor-students-table', 'columns'),
     Output('advisor-courses', 'children')],
    Input('advisor-selector', 'value')
)
def update_advisor_dashboard(selected_advisor):
    try:
        if not selected_advisor or grade_sheet.empty:
            empty_fig = go.Figure()
            empty_fig.update_layout(title="No data available")
            empty_cols = [{'name': 'No Data', 'id': 'no_data'}]
            empty_data = [{'no_data': 'No data available'}]
            empty_courses = html.P("No courses available")
            return empty_fig, empty_data, empty_cols, empty_courses
        
        # Filter for the selected advisor
        advisor_students = grade_sheet[grade_sheet['Advisor'] == selected_advisor]
        
        if advisor_students.empty:
            empty_fig = go.Figure()
            empty_fig.update_layout(title=f"No students found for {selected_advisor}")
            empty_cols = [{'name': 'No Data', 'id': 'no_data'}]
            empty_data = [{'no_data': 'No students available'}]
            courses_list = html.Ul([html.Li("No courses assigned")])
            return empty_fig, empty_data, empty_cols, courses_list
        
        # GPA Trend
        fig = go.Figure()
        gpa_columns = [col for col in advisor_students.columns if 'GPA Semester' in col]
        
        if not gpa_columns:
            fig.update_layout(title=f"No GPA data available for {selected_advisor}'s students")
        else:
            gpa_columns.sort(key=lambda x: int(x.split()[-1]))
            semester_nums = [int(col.split()[-1]) for col in gpa_columns]
            
            for student in advisor_students.index:
                gpa_trend = [advisor_students.loc[student, col] for col in gpa_columns]
                fig.add_trace(go.Scatter(
                    x=semester_nums, 
                    y=gpa_trend, 
                    mode='lines+markers', 
                    name=student
                ))
                
            fig.update_layout(
                title=f'GPA Trends for Students of {selected_advisor}',
                xaxis_title='Semester',
                yaxis_title='GPA',
                yaxis_range=[2.0, 4.0]
            )
        
        # Students Table
        required_columns = ['ID', 'Major', 'Campus']
        available_columns = [col for col in required_columns if col in advisor_students.columns]
        
        if not available_columns and not gpa_columns:
            data_cols = ['Student']
            data = [{'Student': student} for student in advisor_students.index]
        else:
            data_cols = available_columns + gpa_columns
            data = advisor_students[data_cols].reset_index().rename(columns={'index': 'Student'}).to_dict('records')
        
        columns = [{'name': col, 'id': col} for col in ['Student'] + data_cols]
        
        # Advisor Courses
        if selected_advisor in instructor_courses and instructor_courses[selected_advisor]:
            courses_list = html.Ul([html.Li(course) for course in instructor_courses[selected_advisor]])
        else:
            courses_list = html.P("No courses assigned to this advisor")
        
        return fig, data, columns, courses_list
        
    except Exception as e:
        print(f"Error in advisor dashboard: {str(e)}")
        import traceback
        traceback.print_exc()
        
        empty_fig = go.Figure()
        empty_fig.update_layout(title=f"Error: {str(e)}")
        empty_cols = [{'name': 'Error', 'id': 'error'}]
        empty_data = [{'error': f'Error: {str(e)}'}]
        error_msg = html.P(f"Error loading courses: {str(e)}")
        
        return empty_fig, empty_data, empty_cols, error_msg
# Student dashboard
@app.callback(
    [Output('student-info', 'children'),
     Output('student-grade-sheet', 'data'),
     Output('student-grade-sheet', 'columns')],
    Input('student-selector', 'value')
)
def update_student_info_and_grades(selected_student):
    try:
        if not selected_student or grade_sheet.empty:
            return html.P("No student selected or no data available"), \
                   [{'Message': 'No data available'}], [{'name': 'Message', 'id': 'Message'}]
        
        # Debug: print columns and index
        print("Grade sheet columns:", grade_sheet.columns.tolist())
        print("Grade sheet index:", grade_sheet.index.tolist())
        print("Selected student:", selected_student)
        
        # If there's a 'Name' column, use it as the index
        if 'Name' in grade_sheet.columns:
            df = grade_sheet.set_index('Name')
        else:
            df = grade_sheet
        
        # Ensure both sides are strings (and trimmed)
        selected_student_str = str(selected_student).strip()
        df.index = df.index.map(lambda x: str(x).strip())
        
        if selected_student_str not in df.index:
            print(f"Student '{selected_student_str}' not found in index!")
            return html.P(f"Student '{selected_student_str}' not found in data"), \
                   [{'Message': 'Student not found'}], [{'name': 'Message', 'id': 'Message'}]
        
        student_info = df.loc[selected_student_str]
        
        # Build info div
        info_items = []
        for field in ['ID', 'Major', 'Advisor', 'Campus']:
            if field in student_info:
                info_items.append(html.P(f"{field}: {student_info[field]}"))
        if not info_items:
            info_items.append(html.P(f"Student: {selected_student_str}"))
            info_items.append(html.P("No additional information available"))
        info_div = html.Div(info_items)
        
        # Determine semester numbers from columns
        gpa_prefixes = ['Test Semester', 'Midterm Semester', 'Project Semester', 
                        'Final Test Semester', 'Attendance Semester', 'GPA Semester']
        semester_nums = set()
        for prefix in gpa_prefixes:
            for col in df.columns:
                if col.startswith(prefix):
                    try:
                        semester_nums.add(int(col.split()[-1]))
                    except ValueError:
                        pass
        if not semester_nums:
            return info_div, [{'Message': 'No grade data available'}], [{'name': 'Message', 'id': 'Message'}]
        semester_nums = sorted(semester_nums)
        
        # Build grade data for each semester
        grade_data = []
        for sem in semester_nums:
            sem_data = {'Semester': f"Semester {sem}"}
            for assessment in ['Test', 'Midterm', 'Project', 'Final Test', 'Attendance', 'GPA']:
                col_name = f"{assessment} Semester {sem}"
                sem_data[assessment] = df.loc[selected_student_str, col_name] if col_name in df.columns else None
            grade_data.append(sem_data)
        
        # Build table columns
        grade_columns = [{'name': 'Semester', 'id': 'Semester'}]
        for assessment in ['Test', 'Midterm', 'Project', 'Final Test', 'Attendance', 'GPA']:
            if any(f"{assessment} Semester {num}" in df.columns for num in semester_nums):
                grade_columns.append({'name': assessment, 'id': assessment})
        
        return info_div, grade_data, grade_columns
        
    except Exception as e:
        print(f"Error in student info and grades: {str(e)}")
        import traceback
        traceback.print_exc()
        return html.P(f"Error: {str(e)}"), [{'Error': f"Error: {str(e)}"}], [{'name': 'Error', 'id': 'Error'}]
@app.callback(
    Output('calculated-gpa', 'children'),
    [Input('gpa-calculator-table', 'data'),
     Input('gpa-calculator-table', 'columns')]
)
def update_calculated_gpa(rows, columns):
    # Implementation for GPA calculator
    try:
        if not rows:
            return "No data entered"
        
        total_points = 0
        total_credits = 0
        
        for row in rows:
            if row.get('credits') and row.get('grade'):
                try:
                    credits = float(row['credits'])
                    grade = float(row['grade'])
                    gpa_value = grade_to_gpa(grade)
                    total_points += credits * gpa_value
                    total_credits += credits
                except (ValueError, TypeError):
                    pass
        
        if total_credits == 0:
            return "Enter credits and grades"
        
        calculated_gpa = total_points / total_credits
        return f"{calculated_gpa:.2f}"
        
    except Exception as e:
        print(f"Error calculating GPA: {str(e)}")
        return f"Error: {str(e)}"

# Campus dashboard
@app.callback(
    [Output('campus-major-distribution', 'figure'),
     Output('campus-avg-gpa-trend', 'figure'),
     Output('campus-students-table', 'data'),
     Output('campus-students-table', 'columns')],
    Input('campus-selector', 'value')
)
def update_campus_dashboard(selected_campus):
    # Implementation for campus dashboard
    try:
        if not selected_campus or grade_sheet.empty:
            empty_fig1 = go.Figure()
            empty_fig1.update_layout(title="No data available")
            empty_fig2 = go.Figure()
            empty_fig2.update_layout(title="No data available")
            return empty_fig1, empty_fig2, [], []
        
        # Filter for the selected campus
        campus_students = grade_sheet[grade_sheet['Campus'] == selected_campus]
        
        # Major distribution
        if 'Major' in campus_students.columns:
            major_counts = campus_students['Major'].value_counts()
            major_fig = px.pie(
                values=major_counts.values, 
                names=major_counts.index,
                title=f'Student Distribution by Major in {selected_campus}'
            )
        else:
            major_fig = go.Figure()
            major_fig.update_layout(title="Major data not available")
        
        # GPA trend
        gpa_columns = [col for col in campus_students.columns if 'GPA Semester' in col]
        if gpa_columns:
            gpa_columns.sort(key=lambda x: int(x.split()[-1]))
            avg_gpa = campus_students[gpa_columns].mean()
            semester_nums = [int(col.split()[-1]) for col in gpa_columns]
            
            gpa_fig = px.bar(
                x=semester_nums,
                y=avg_gpa.values,
                labels={'x': 'Semester', 'y': 'Average GPA'},
                title=f'Average GPA Trend Across Semesters in {selected_campus}'
            )
            gpa_fig.update_yaxes(range=[2.0, 4.0])
        else:
            gpa_fig = go.Figure()
            gpa_fig.update_layout(title="GPA data not available")
        
        # Students table
        required_columns = ['ID', 'Major', 'Advisor']
        available_columns = [col for col in required_columns if col in campus_students.columns]
        gpa_columns = [col for col in campus_students.columns if 'GPA Semester' in col]
        
        data_columns = available_columns + gpa_columns
        
        if not data_columns:
            return major_fig, gpa_fig, [], []
        
        data = campus_students[data_columns].reset_index().rename(columns={'index': 'Student'}).to_dict('records')
        columns = [{'name': col, 'id': col} for col in ['Student'] + data_columns]
        
        return major_fig, gpa_fig, data, columns
        
    except Exception as e:
        print(f"Error in campus dashboard: {str(e)}")
        empty_fig1 = go.Figure()
        empty_fig1.update_layout(title=f"Error: {str(e)}")
        empty_fig2 = go.Figure()
        empty_fig2.update_layout(title=f"Error: {str(e)}")
        return empty_fig1, empty_fig2, [], []
    
@app.callback(
    Output('admin-campus-comparison', 'figure'),
    Input('comparative-analytics', 'value')
)
def update_comparative_analytics(comp_value):
    if grade_sheet.empty:
        fig = go.Figure()
        fig.update_layout(title="No data available")
        return fig

    if comp_value == 'campus_bench':
        gpa_columns = [col for col in grade_sheet.columns if 'GPA Semester' in col]
        if not gpa_columns:
            fig = go.Figure()
            fig.update_layout(title="No GPA data available for campus comparison")
            return fig
        gpa_columns.sort(key=lambda x: int(x.split()[-1]))
        campus_comparison = {}
        for campus in grade_sheet['Campus'].unique():
            campus_data = grade_sheet[grade_sheet['Campus'] == campus]
            campus_comparison[campus] = campus_data[gpa_columns].mean()
        fig = go.Figure()
        for campus, gpa_data in campus_comparison.items():
            semester_nums = [int(col.split()[-1]) for col in gpa_columns]
            fig.add_trace(go.Scatter(
                x=semester_nums,
                y=gpa_data.values,
                mode='lines+markers',
                name=campus
            ))
        fig.update_layout(
            title='Campus Performance Benchmarking',
            xaxis_title='Semester',
            yaxis_title='Average GPA',
            yaxis_range=[2.0, 4.0]
        )
        return fig

    elif comp_value == 'major_trend':
        fig = go.Figure()
        if 'Major' not in grade_sheet.columns:
            fig.update_layout(title="Major data not available")
            return fig
        for major in grade_sheet['Major'].unique():
            major_data = grade_sheet[grade_sheet['Major'] == major]
            if major_data.empty:
                continue
            avg_gpa = major_data[[col for col in grade_sheet.columns if 'GPA Semester' in col]].mean()
            semester_nums = [int(col.split()[-1]) for col in sorted(
                [col for col in grade_sheet.columns if 'GPA Semester' in col],
                key=lambda x: int(x.split()[-1])
            )]
            fig.add_trace(go.Scatter(
                x=semester_nums,
                y=avg_gpa.values,
                mode='lines+markers',
                name=major
            ))
        fig.update_layout(title="Major-wise Trending", xaxis_title="Semester", yaxis_title="Average GPA", yaxis_range=[2.0,4.0])
        return fig

    elif comp_value == 'cross_sem':
        fig = go.Figure()
        fig.update_layout(title="Cross-semester Performance not implemented")
        return fig

    else:
        fig = go.Figure()
        fig.update_layout(title="Invalid selection")
        return fig
   
            


port = int(os.environ.get('PORT', 8050))
app.run_server(debug=True, host='0.0.0.0', port=port)