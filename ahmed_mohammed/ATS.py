# Part 1: Imports and Setup
import pandas as pd
import numpy as np
import requests
import plotly.graph_objs as go
import plotly.express as px
from dash import dash_table
from dash import Dash, dcc, html, Input, Output, State, callback_context, dash_table
from dash.dependencies import State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from auth import init_db, get_login_layout, setup_auth_callbacks, get_auth_layout, validate_token, get_user_role
import os

# Add Flask imports for REST API services
from flask import Flask, jsonify, request, make_response
from flask_cors import CORS
from flask_restful import Api, Resource

# Add machine learning imports for enhanced predictions
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
from sklearn.impute import SimpleImputer
import joblib
import os
import json
from datetime import datetime, timedelta

def load_api_data(api_url='https://sheet2api.com/v1/rVktVGeqShCE/student_data'):
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
        # New: Enhanced prediction functions
def prepare_data_for_prediction(grade_sheet):
    """Prepare data for prediction models"""
    try:
        # Get GPA columns
        gpa_columns = [col for col in grade_sheet.columns if 'GPA Semester' in col]
        if not gpa_columns or len(gpa_columns) < 2:
            print("Not enough GPA data for predictions")
            return None, None
            
        # Rest of implementation...
        return [], ([], [])  # Simplified return
    except Exception as e:
        print(f"Error preparing data for prediction: {str(e)}")
        return None, None

def train_prediction_models(grade_sheet):
    """Train machine learning models for GPA and at-risk prediction"""
    try:
        # Implementation details...
        print("Training models...")
        return None, None  # For now just return None placeholders
    except Exception as e:
        print(f"Error training prediction models: {str(e)}")
        return None, None

def predict_future_performance(grade_sheet, student_name):
    """Predict future performance for a specific student"""
    try:
        # Implementation details...
        # Return a basic prediction result structure
        return {
            'student_name': student_name,
            'current_gpa': 3.0,
            'predicted_next_gpa': 3.2,
            'at_risk_probability': 20.0,
            'intervention_needed': False,
            'weak_areas': [],
            'recommendations': ["Continue with current study habits"],
            'prediction_date': datetime.now().strftime('%Y-%m-%d')
        }
    except Exception as e:
        print(f"Error predicting future performance: {str(e)}")
        return None

def predict_graduation_outlook(grade_sheet, student_name):
    """Predict graduation outlook and timeline for a student"""
    try:
        # Implementation details...
        # Return a basic outlook structure
        return {
            'student_name': student_name,
            'current_semester': 2,
            'current_avg_gpa': 3.2,
            'gpa_trend': "Stable",
            'graduation_likelihood': "Good (75-90%)",
            'estimated_semesters_to_graduation': 6,
            'estimated_graduation_date': (datetime.now() + timedelta(days=720)).strftime('%B %Y'),
            'recommendations': ["Continue with current academic strategies."]
        }
    except Exception as e:
        print(f"Error predicting graduation outlook: {str(e)}")
        return None

def predict_cohort_trends(grade_sheet):
    """Predict trends for student cohorts by major or campus"""
    try:
        # Implementation details...
        # Return simple placeholder structures
        major_trends = {'Computer Science': {'trend_slope': 0.1, 'trend_direction': "Upward", 'avg_gpa': 3.2, 'latest_gpa': 3.3, 'gpa_values': [3.1, 3.2, 3.3]}}
        campus_trends = {'Main Campus': {'trend_slope': 0.05, 'trend_direction': "Upward", 'avg_gpa': 3.0, 'latest_gpa': 3.1, 'gpa_values': [2.9, 3.0, 3.1]}}
        
        major_cohort_predictions = {
            'major_trends': major_trends,
            'exceptional_majors': ['Computer Science'],
            'concerning_majors': []
        }
        
        campus_cohort_predictions = {
            'campus_trends': campus_trends,
            'exceptional_campuses': ['Main Campus'],
            'concerning_campuses': []
        }
        
        return major_cohort_predictions, campus_cohort_predictions
    except Exception as e:
        print(f"Error predicting cohort trends: {str(e)}")
        return None, None

# Load data from API
try:
    grade_sheet = load_api_data()
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

# Initialize Flask server for API endpoints
flask_app = Flask(__name__)
CORS(flask_app)
api = Api(flask_app)

# Define REST API resources
class StudentPredictionResource(Resource):
    def get(self, student_name):
        try:
            if student_name not in grade_sheet.index:
                return {"error": f"Student {student_name} not found"}, 404
                
            # Get predictions
            performance_prediction = predict_future_performance(grade_sheet, student_name)
            graduation_outlook = predict_graduation_outlook(grade_sheet, student_name)
            
            if performance_prediction is None or graduation_outlook is None:
                return {"error": "Could not generate predictions"}, 500
                
            # Combine results
            prediction_results = {
                "student": student_name,
                "performance_prediction": performance_prediction,
                "graduation_outlook": graduation_outlook
            }
            
            return prediction_results, 200
        except Exception as e:
            return {"error": str(e)}, 500

class CohortPredictionResource(Resource):
    def get(self):
        try:
            # Get cohort predictions
            major_predictions, campus_predictions = predict_cohort_trends(grade_sheet)
            
            if major_predictions is None or campus_predictions is None:
                return {"error": "Could not generate cohort predictions"}, 500
                
            # Return results
            prediction_results = {
                "major_cohorts": major_predictions,
                "campus_cohorts": campus_predictions
            }
            
            return prediction_results, 200
        except Exception as e:
            return {"error": str(e)}, 500

class AtRiskStudentsResource(Resource):
    def get(self):
        try:
            # Get at-risk students
            at_risk_df = detect_at_risk_students(grade_sheet)
            
            if at_risk_df.empty:
                return {"message": "No at-risk students found"}, 200
                
            # Convert to dictionary for JSON response
            at_risk_students = at_risk_df.to_dict('records')
            
            return {"at_risk_students": at_risk_students}, 200
        except Exception as e:
            return {"error": str(e)}, 500

class PredictNextGPAResource(Resource):
    def post(self):
        try:
            # Get JSON data from request
            data = request.get_json()
            
            if not data or 'student_name' not in data:
                return {"error": "Student name is required"}, 400
                
            student_name = data['student_name']
            
            if student_name not in grade_sheet.index:
                return {"error": f"Student {student_name} not found"}, 404
                
            # Get student data
            student_data = grade_sheet.loc[student_name]
            
            # Get current GPA columns
            gpa_columns = [col for col in grade_sheet.columns if 'GPA Semester' in col]
            
            if not gpa_columns:
                return {"error": "No GPA data available"}, 400
                
            gpa_columns.sort(key=lambda x: int(x.split()[-1]))
            
            # Create feature vector for prediction
            features = []
            
            # Add previous GPAs
            for col in gpa_columns:
                features.append(student_data[col])
                
            # Add major and campus as categorical features
            features.append(student_data['Major'])
            features.append(student_data['Campus'])
            
            # Try loading the model or train a new one
            model_path = 'models/gpa_prediction_model.pkl'
            
            if os.path.exists(model_path):
                model = joblib.load(model_path)
            else:
                # Train a new model
                model, _ = train_prediction_models(grade_sheet)
                
                if model is None:
                    return {"error": "Could not train prediction model"}, 500
            
            # Make prediction
            predicted_gpa = predict_future_performance(grade_sheet, student_name)
            
            if predicted_gpa is None:
                return {"error": "Could not generate prediction"}, 500
                
            return {"prediction": predicted_gpa}, 200
        except Exception as e:
            return {"error": str(e)}, 500

# Register API resources
api.add_resource(StudentPredictionResource, '/api/prediction/student/<string:student_name>')
api.add_resource(CohortPredictionResource, '/api/prediction/cohorts')
api.add_resource(AtRiskStudentsResource, '/api/at-risk-students')
api.add_resource(PredictNextGPAResource, '/api/prediction/next-gpa')

# Initialize the Dash app with suppress_callback_exceptions
app = Dash(__name__, url_base_pathname='/ats/')

app.title = "HCT Dashboard"

# Set a simple default layout as fallback
default_layout = html.Div([
    html.H1("HCT Dashboard"),
    html.Div(id='page-content')
])

# Try to get auth layout, use default if it fails
try:
    app.layout = get_auth_layout()
except Exception as e:
    print(f"Error setting auth layout: {str(e)}. Using default layout.")
    app.layout = default_layout

# Create the prediction model when the app starts
try:
    print("Training initial prediction models...")
    gpa_model, at_risk_model = train_prediction_models(grade_sheet)
    if gpa_model is not None and at_risk_model is not None:
        print("Initial prediction models trained successfully")
    else:
        print("Could not train initial prediction models - will try again when needed")
except Exception as e:
    print(f"Error training initial prediction models: {str(e)}")
    import traceback
    traceback.print_exc()

# Setup authentication callbacks first
setup_auth_callbacks(app)

# Admin dropdown component
def create_admin_dropdown():
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Label("View Type:"),
                dcc.Dropdown(
                    id='admin-view-selector',
                    options=[
                        {'label': 'Dashboard Overview', 'value': 'overview'},
                        {'label': 'Student Analytics', 'value': 'student'},
                        {'label': 'Faculty Performance', 'value': 'faculty'},
                        {'label': 'Academic Programs', 'value': 'programs'},
                        {'label': 'Campus Reports', 'value': 'campus'},
                        {'label': 'System Settings', 'value': 'settings'}
                    ],
                    value='overview',
                    clearable=False,
                    className="mb-3"
                )
            ], width=4),
            dbc.Col([
                html.Label("Data Filter:"),
                dcc.Dropdown(
                    id='admin-filter-selector',
                    options=[
                        {'label': 'All Data', 'value': 'all'},
                        {'label': 'Current Semester', 'value': 'current'},
                        {'label': 'Previous Year', 'value': 'previous'},
                        {'label': 'Custom Range', 'value': 'custom'}
                    ],
                    value='all',
                    clearable=False,
                    className="mb-3"
                )
            ], width=4),
            dbc.Col([
                html.Label("Actions:"),
                dbc.ButtonGroup([
                    dbc.Button("Generate Report", id="admin-report-btn", color="primary", className="mr-2"),
                    dbc.Button("Export Data", id="admin-export-btn", color="secondary")
                ])
            ], width=4)
        ]),
        html.Div(id='admin-view-content')
    ], className="mt-4 mb-4")

# Advisor dropdown component
def create_advisor_dropdown():
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Label("Select Advisor:"),
                dcc.Dropdown(
                    id='advisor-selector',
                    options=[{'label': advisor, 'value': advisor} for advisor in advisors],
                    value=advisors[0] if advisors else None,
                    clearable=False,
                    className="mb-3"
                )
            ], width=4),
            dbc.Col([
                html.Label("View Type:"),
                dcc.Dropdown(
                    id='advisor-view-selector',
                    options=[
                        {'label': 'Student Overview', 'value': 'students'},
                        {'label': 'At-Risk Students', 'value': 'at-risk'},
                        {'label': 'Course Management', 'value': 'courses'},
                        {'label': 'Academic Planning', 'value': 'planning'},
                        {'label': 'Office Hours', 'value': 'office'}
                    ],
                    value='students',
                    clearable=False,
                    className="mb-3"
                )
            ], width=4),
            dbc.Col([
                html.Label("Communication:"),
                dbc.ButtonGroup([
                    dbc.Button("Email Students", id="advisor-email-btn", color="primary", className="mr-2"),
                    dbc.Button("Schedule Meeting", id="advisor-meeting-btn", color="secondary")
                ])
            ], width=4)
        ]),
        html.Div(id='advisor-view-content')
    ], className="mt-4 mb-4")

# Page content callback
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname'),
     Input('session-token', 'data')],
    prevent_initial_call=False
)
def display_page(pathname, token):
    # Print debug info to see what's happening
    print(f"Display page callback: Path={pathname}, Token exists={token is not None}")
    
    # If login path or no token, show login
    if pathname == '/login' or token is None:
        return get_login_layout()
    
    # If we have token, check if it's valid
    if token:
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
    
    # Default case: show login
    return get_login_layout()

# Add this callback to handle role-specific content
@app.callback(
    Output('role-content', 'children'),
    [Input('role-selector', 'value')]
)
def display_role_content(selected_role):
    print(f"Displaying content for role: {selected_role}")
    
    if not selected_role:
        return html.Div("Please select a role")
    
    # Admin role content
    if selected_role == 'admin':
        return html.Div([
            html.H2("Admin Dashboard"),
            create_admin_dropdown(),
            html.Div(id='admin-dashboard-content', children=[
                dbc.Row([
                    dbc.Col(dcc.Graph(id='admin-at-risk-by-campus'), width=6),
                    dbc.Col(dcc.Graph(id='admin-avg-gpa-trend'), width=6)
                ]),
                dbc.Row([
                    dbc.Col(dcc.Graph(id='admin-major-distribution'), width=6),
                    dbc.Col(dcc.Graph(id='admin-advisor-workload'), width=6)
                ]),
                html.H3("Student-Advisor Assignments"),
                dash_table.DataTable(
                    id='admin-student-advisor-table',
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left'},
                    style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}
                ),
                html.H3("Cohort Predictions"),
                dcc.Graph(id='admin-cohort-predictions')
            ])
        ])
    
    # Instructor role content
    elif selected_role == 'instructor':
        return html.Div([
            html.H2("Instructor Dashboard"),
            html.Label("Select Instructor:"),
            dcc.Dropdown(
                id='instructor-selector',
                options=[{'label': instructor, 'value': instructor} for instructor in advisors],
                value=advisors[0] if advisors else None,
                className="mb-3"
            ),
            dbc.Row([
                dbc.Col([
                    html.H3("Student GPA Trends"),
                    dcc.Graph(id='instructor-gpa-line-graph')
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    html.H3("My Students"),
                    dash_table.DataTable(
                        id='instructor-students-table',
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left'},
                        style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}
                    )
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    html.H3("At-Risk Students"),
                    dash_table.DataTable(
                        id='instructor-at-risk-table',
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left'},
                        style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}
                    )
                ], width=12)
            ])
        ])
    
    # Student role content
    elif selected_role == 'student':
        return html.Div([
            html.H2("Student Dashboard"),
            html.Label("Select Student:"),
            dcc.Dropdown(
                id='student-selector',
                options=[{'label': student, 'value': student} for student in students],
                value=students[0] if students else None,
                className="mb-3"
            ),
            dbc.Row([
                dbc.Col([
                    html.H3("Student Information"),
                    html.Div(id='student-info')
                ], width=4),
                dbc.Col([
                    html.H3("GPA Trend"),
                    dcc.Graph(id='student-gpa-trend')
                ], width=8)
            ]),
            dbc.Row([
                dbc.Col([
                    html.H3("Grade Sheet"),
                    dash_table.DataTable(
                        id='student-grade-sheet',
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left'},
                        style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}
                    )
                ], width=12)
            ])
        ])
    
    # Advisor role content
    elif selected_role == 'advisor':
        return html.Div([
            html.H2("Advisor Dashboard"),
            create_advisor_dropdown(),
            html.Div(id='advisor-dashboard-content', children=[
                dbc.Row([
                    dbc.Col([
                        html.H3("Student GPA Trends"),
                        dcc.Graph(id='advisor-student-gpa-trend')
                    ], width=12)
                ]),
                dbc.Row([
                    dbc.Col([
                        html.H3("My Students"),
                        dash_table.DataTable(
                            id='advisor-students-table',
                            style_table={'overflowX': 'auto'},
                            style_cell={'textAlign': 'left'},
                            style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}
                        )
                    ], width=12)
                ]),
                dbc.Row([
                    dbc.Col([
                        html.H3("My Courses"),
                        html.Div(id='advisor-courses')
                    ], width=12)
                ])
            ])
        ])
    
    # Campus role content
    elif selected_role == 'campus':
        return html.Div([
            html.H2("Campus Dashboard"),
            html.Label("Select Campus:"),
            dcc.Dropdown(
                id='campus-selector',
                options=[{'label': campus, 'value': campus} for campus in campuses],
                value=campuses[0] if campuses else None,
                className="mb-3"
            ),
            dbc.Row([
                dbc.Col(dcc.Graph(id='campus-major-distribution'), width=6),
                dbc.Col(dcc.Graph(id='campus-avg-gpa-trend'), width=6)
            ]),
            dbc.Row([
                dbc.Col([
                    html.H3("Campus Students"),
                    dash_table.DataTable(
                        id='campus-students-table',
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left'},
                        style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}
                    )
                ], width=12)
            ])
        ])
    
    # Default case
    return html.Div(f"Content for {selected_role} role is not implemented")

# Add new callbacks for the admin view selector
@app.callback(
    Output('admin-dashboard-content', 'children'),
    [Input('admin-view-selector', 'value'),
     Input('admin-filter-selector', 'value')]
)
def update_admin_view(view_type, filter_type):
    print(f"Updating admin view: {view_type}, filter: {filter_type}")
    
    # Default view (overview)
    if view_type == 'overview' or not view_type:
        return [
            dbc.Row([
                dbc.Col(dcc.Graph(id='admin-at-risk-by-campus'), width=6),
                dbc.Col(dcc.Graph(id='admin-avg-gpa-trend'), width=6)
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='admin-major-distribution'), width=6),
                dbc.Col(dcc.Graph(id='admin-advisor-workload'), width=6)
            ]),
            html.H3("Student-Advisor Assignments"),
            dash_table.DataTable(
                id='admin-student-advisor-table',
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left'},
                style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}
            ),
            html.H3("Cohort Predictions"),
            dcc.Graph(id='admin-cohort-predictions')
        ]
    
    # Student analytics view
    elif view_type == 'student':
        return [
            html.H3("Student Performance Analytics"),
            dbc.Row([
                dbc.Col([
                    html.H4("Filter Students"),
                    dcc.Dropdown(
                        id='admin-student-major-filter',
                        options=[{'label': major, 'value': major} for major in majors],
                        multi=True,
                        placeholder="Select majors..."
                    ),
                    html.Div(className="mb-3"),
                    dcc.Dropdown(
                        id='admin-student-campus-filter',
                        options=[{'label': campus, 'value': campus} for campus in campuses],
                        multi=True,
                        placeholder="Select campuses..."
                    ),
                    html.Div(className="mb-3"),
                    dbc.Button("Apply Filters", id="admin-apply-filters", color="primary")
                ], width=3),
                dbc.Col([
                    html.H4("Performance Distribution"),
                    dcc.Graph(id='admin-student-performance-dist')
                ], width=9)
            ]),
            html.H3("Student List"),
            dash_table.DataTable(
                id='admin-filtered-students-table',
                page_size=15,
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left'},
                style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'},
                filter_action="native",
                sort_action="native"
            )
        ]
    
    # Faculty performance view  
    elif view_type == 'faculty':
        return [
            html.H3("Faculty Performance Metrics"),
            dbc.Row([
                dbc.Col(dcc.Graph(id='admin-faculty-workload'), width=6),
                dbc.Col(dcc.Graph(id='admin-faculty-student-success'), width=6)
            ]),
            html.H3("Faculty Details"),
            dash_table.DataTable(
                id='admin-faculty-details-table',
                page_size=10,
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left'},
                style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}
            )
        ]
    
    # Programs view
    elif view_type == 'programs':
        return [
            html.H3("Academic Programs Analysis"),
            dbc.Row([
                dbc.Col(dcc.Graph(id='admin-program-enrollment'), width=6),
                dbc.Col(dcc.Graph(id='admin-program-completion'), width=6)
            ]),
            html.H3("Program Details"),
            dash_table.DataTable(
                id='admin-program-details-table',
                page_size=10,
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left'},
                style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}
            )
        ]
    
    # Campus view
    elif view_type == 'campus':
        return [
            html.H3("Campus Comparison"),
            dbc.Row([
                dbc.Col(dcc.Graph(id='admin-campus-enrollment'), width=6),
                dbc.Col(dcc.Graph(id='admin-campus-performance'), width=6)
            ]),
            html.H3("Campus Resources"),
            dash_table.DataTable(
                id='admin-campus-resources-table',
                page_size=10,
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left'},
                style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}
            )
        ]
    
    # Settings view
    elif view_type == 'settings':
        return [
            html.H3("System Settings"),
            dbc.Row([
                dbc.Col([
                    html.H4("User Management"),
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Button("Add User", id="admin-add-user", color="primary", className="mr-2"),
                            dbc.Button("Manage Roles", id="admin-manage-roles", color="secondary")
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    html.H4("Data Management"),
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Button("Backup Data", id="admin-backup-data", color="primary", className="mr-2"),
                            dbc.Button("Import Data", id="admin-import-data", color="secondary")
                        ])
                    ])
                ], width=6)
            ]),
            html.H4("Notification Settings", className="mt-4"),
            dbc.Card([
                dbc.CardBody([
                    html.P("Configure system notifications:"),
                    dbc.FormGroup([
                        dbc.Checkbox(id="notify-at-risk", checked=True),
                        dbc.Label("At-risk student alerts", html_for="notify-at-risk", className="ml-2")
                    ]),
                    dbc.FormGroup([
                        dbc.Checkbox(id="notify-grade-entry", checked=True),
                        dbc.Label("Grade entry deadlines", html_for="notify-grade-entry", className="ml-2")
                    ]),
                    dbc.FormGroup([
                        dbc.Checkbox(id="notify-system", checked=True),
                        dbc.Label("System maintenance updates", html_for="notify-system", className="ml-2")
                    ])
                ])
            ])
        ]
    
    # Default case
    return html.Div("Select a view to continue")

# Add new callbacks for the advisor view selector
@app.callback(
    Output('advisor-dashboard-content', 'children'),
    [Input('advisor-selector', 'value'),
     Input('advisor-view-selector', 'value')]
)
def update_advisor_view(selected_advisor, view_type):
    print(f"Updating advisor view: {view_type}, advisor: {selected_advisor}")
    
    if not selected_advisor:
        return html.Div("Please select an advisor")
    
    # Default view (student overview)
    if view_type == 'students' or not view_type:
        return [
            dbc.Row([
                dbc.Col([
                    html.H3("Student GPA Trends"),
                    dcc.Graph(id='advisor-student-gpa-trend')
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    html.H3("My Students"),
                    dash_table.DataTable(
                        id='advisor-students-table',
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left'},
                        style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}
                    )
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    html.H3("My Courses"),
                    html.Div(id='advisor-courses')
                ], width=12)
            ])
        ]
    
    # At-risk students view
    elif view_type == 'at-risk':
        return [
            html.H3("At-Risk Students"),
            dbc.Alert(
                "These students may need additional support based on their academic performance.",
                color="warning"
            ),
            dash_table.DataTable(
                id='advisor-at-risk-table',
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left'},
                style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}
            ),
            html.H3("Intervention Recommendations", className="mt-4"),
            dbc.Row([
                dbc.Col(html.Div(id='advisor-intervention-recommendations'), width=12)
            ])
        ]
    
    # Course management view
    elif view_type == 'courses':
        return [
            html.H3("Course Management"),
            dbc.Row([
                dbc.Col([
                    html.H4("Courses Taught"),
                    html.Div(id='advisor-courses-list')
                ], width=6),
                dbc.Col([
                    html.H4("Course Performance"),
                    dcc.Graph(id='advisor-course-performance')
                ], width=6)
            ]),
            html.H3("Course Details", className="mt-4"),
            dbc.Card([
                dbc.CardHeader(
                    dcc.Dropdown(
                        id='advisor-course-selector',
                        options=[{'label': course, 'value': course} 
                                for course in instructor_courses.get(selected_advisor, [])],
                        placeholder="Select a course to view details",
                    )
                ),
                dbc.CardBody(html.Div(id='advisor-course-details'))
            ])
        ]
    
    # Academic planning view
    elif view_type == 'planning':
        return [
            html.H3("Academic Planning"),
            dbc.Row([
                dbc.Col([
                    html.H4("Student Graduation Timeline"),
                    dcc.Dropdown(
                        id='advisor-planning-student-selector',
                        placeholder="Select a student"
                    ),
                    html.Div(className="mb-3"),
                    dcc.Graph(id='advisor-student-timeline')
                ], width=12)
            ]),
            html.H3("Graduation Outlook", className="mt-4"),
            html.Div(id='advisor-graduation-outlook')
        ]
    
    # Office hours view
    elif view_type == 'office':
        return [
            html.H3("Office Hours Management"),
            dbc.Row([
                dbc.Col([
                    html.H4("Schedule"),
                    dbc.Card([
                        dbc.CardBody([
                            dbc.FormGroup([
                                dbc.Label("Day:"),
                                dcc.Dropdown(
                                    id='office-day-selector',
                                    options=[
                                        {'label': 'Monday', 'value': 'monday'},
                                        {'label': 'Tuesday', 'value': 'tuesday'},
                                        {'label': 'Wednesday', 'value': 'wednesday'},
                                        {'label': 'Thursday', 'value': 'thursday'},
                                        {'label': 'Friday', 'value': 'friday'}
                                    ],
                                    multi=True,
                                    value=['monday', 'wednesday']
                                )
                            ]),
                            dbc.FormGroup([
                                dbc.Label("Time:"),
                                dbc.Row([
                                    dbc.Col(dcc.Dropdown(
                                        id='office-start-time',
                                        options=[{'label': f"{h}:00", 'value': h} for h in range(8, 18)],
                                        value=10
                                    ), width=5),
                                    dbc.Col(html.Span("to"), width=2, className="text-center"),
                                    dbc.Col(dcc.Dropdown(
                                        id='office-end-time',
                                        options=[{'label': f"{h}:00", 'value': h} for h in range(9, 19)],
                                        value=12
                                    ), width=5)
                                ])
                            ]),
                            dbc.Button("Update Schedule", id="update-office-hours", color="primary")
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    html.H4("Upcoming Appointments"),
                    html.Div(id='advisor-appointments')
                ], width=6)
            ])
        ]
    
    # Default case
    return html.Div("Select a view to continue")

# Admin Dashboard Callbacks
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
    Output('admin-avg-gpa-trend', 'figure'),
    Input('admin-avg-gpa-trend', 'id')
)
def update_admin_avg_gpa_trend(_):
    try:
        gpa_columns = [col for col in grade_sheet.columns if 'GPA Semester' in col]
        if not gpa_columns:
            fig = go.Figure()
            fig.update_layout(title="No GPA data available")
            return fig
            
        gpa_columns.sort(key=lambda x: int(x.split()[-1]))
        
        avg_gpa = grade_sheet[gpa_columns].mean()
        semester_labels = [f'Semester {i+1}' for i in range(len(gpa_columns))]
        
        fig = px.bar(x=semester_labels, y=avg_gpa,
                     labels={'x': 'Semester', 'y': 'Average GPA'},
                     title='Average GPA Trend Across Semesters')
        fig.update_yaxes(range=[2.0, 4.0])
        return fig
    except Exception as e:
        print(f"Error in admin-avg-gpa-trend: {str(e)}")
        fig = go.Figure()
        fig.update_layout(title="Error loading GPA trend data")
        return fig

@app.callback(
    Output('admin-major-distribution', 'figure'),
    Input('admin-major-distribution', 'id')
)
def update_admin_major_distribution(_):
    try:
        if 'Major' not in grade_sheet.columns or grade_sheet.empty:
            fig = go.Figure()
            fig.update_layout(title="No major data available")
            return fig
            
        major_counts = grade_sheet['Major'].value_counts()
        fig = px.pie(values=major_counts.values, names=major_counts.index,
                     title='Student Distribution by Major')
        return fig
    except Exception as e:
        print(f"Error in admin-major-distribution: {str(e)}")
        fig = go.Figure()
        fig.update_layout(title="Error loading major distribution data")
        return fig

@app.callback(
    Output('admin-advisor-workload', 'figure'),
    Input('admin-advisor-workload', 'id')
)
def update_admin_advisor_workload(_):
    try:
        if 'Advisor' not in grade_sheet.columns or grade_sheet.empty:
            fig = go.Figure()
            fig.update_layout(title="No advisor data available")
            return fig
            
        advisor_counts = grade_sheet['Advisor'].value_counts()
        fig = px.bar(x=advisor_counts.index, y=advisor_counts.values,
                     title='Advisor Workload',
                     labels={'x': 'Advisor', 'y': 'Number of Students'})
        return fig
    except Exception as e:
        print(f"Error in admin-advisor-workload: {str(e)}")
        fig = go.Figure()
        fig.update_layout(title="Error loading advisor workload data")
        return fig

@app.callback(
    [Output('admin-student-advisor-table', 'data'),
     Output('admin-student-advisor-table', 'columns')],
    Input('admin-student-advisor-table', 'id')
)
def update_admin_student_advisor_table(_):
    try:
        if grade_sheet.empty:
            return [], []
            
        required_cols = ['ID', 'Major', 'Advisor', 'Campus']
        if not all(col in grade_sheet.columns for col in required_cols):
            return [], []
            
        data = grade_sheet[required_cols].reset_index().rename(
            columns={'index': 'Student'}).to_dict('records')
        columns = [{'name': col, 'id': col} for col in ['Student', 'ID', 'Major', 'Advisor', 'Campus']]
        return data, columns
    except Exception as e:
        print(f"Error in admin-student-advisor-table: {str(e)}")
        return [], []

# New callbacks for predictive features
@app.callback(
    Output('admin-cohort-predictions', 'figure'),
    Input('admin-cohort-predictions', 'id')
)
def update_admin_cohort_predictions(_):
    try:
        major_predictions, campus_predictions = predict_cohort_trends(grade_sheet)
        
        if major_predictions is None or campus_predictions is None:
            fig = go.Figure()
            fig.update_layout(title="Could not generate cohort predictions")
            return fig
            
        # Create a heatmap of GPA trends by major
        major_data = []
        majors = []
        semesters = []
        
        for major, data in major_predictions['major_trends'].items():
            majors.append(major)
            gpa_values = data['gpa_values']
            for i, gpa in enumerate(gpa_values):
                semester = f"Semester {i+1}"
                if semester not in semesters:
                    semesters.append(semester)
                major_data.append({
                    'Major': major,
                    'Semester': semester,
                    'GPA': gpa
                })
                
        df = pd.DataFrame(major_data)
        
        # Create heatmap
        fig = px.density_heatmap(df, x='Semester', y='Major', z='GPA',
                              title='Cohort GPA Trends by Major',
                              color_continuous_scale='RdYlGn')
        
        fig.update_layout(
            height=600,
            coloraxis_colorbar=dict(
                title="GPA",
                tickvals=[2.0, 2.5, 3.0, 3.5, 4.0],
                ticktext=["2.0", "2.5", "3.0", "3.5", "4.0"]
            )
        )
        
        return fig
    except Exception as e:
        print(f"Error in admin-cohort-predictions: {str(e)}")
        fig = go.Figure()
        fig.update_layout(title=f"Error: {str(e)}")
        return fig

# Instructor Dashboard Callbacks
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
            return empty_fig, [], [], [], []
            
        instructor_students = grade_sheet[grade_sheet['Advisor'] == selected_instructor]
        
        # Handle empty data
        if instructor_students.empty:
            empty_fig = go.Figure()
            empty_fig.update_layout(title="No students found for selected instructor")
            return empty_fig, [], [], [], []

        # GPA Line Graph
        fig = go.Figure()
        gpa_columns = [col for col in instructor_students.columns if 'GPA Semester' in col]
        gpa_columns.sort(key=lambda x: int(x.split()[-1]))
        
        for student in instructor_students.index:
            gpa_trend = [instructor_students.loc[student, col] for col in gpa_columns]
            semester_nums = [int(col.split()[-1]) for col in gpa_columns]
            fig.add_trace(go.Scatter(x=semester_nums, y=gpa_trend, mode='lines+markers', name=student))
        
        fig.update_layout(
            title=f'GPA Trends for Students of {selected_instructor}',
            xaxis_title='Semester',
            yaxis_title='GPA',
            yaxis_range=[2.0, 4.0]
        )

        # Students Table
        students_data = instructor_students[['ID', 'Major', 'Campus'] + 
                                         gpa_columns].reset_index().rename(
            columns={'index': 'Student'}).to_dict('records')
        students_columns = [{'name': col, 'id': col} for col in ['Student', 'ID', 'Major', 'Campus'] + 
                                                              gpa_columns]

        # At-Risk Table
        at_risk_df = detect_at_risk_students(instructor_students)
        at_risk_data = at_risk_df.to_dict('records') if not at_risk_df.empty else []
        at_risk_columns = [{'name': col, 'id': col} for col in at_risk_df.columns] if not at_risk_df.empty else []

        return fig, students_data, students_columns, at_risk_data, at_risk_columns
        
    except Exception as e:
        print(f"Error in instructor dashboard: {str(e)}")
        empty_fig = go.Figure()
        empty_fig.update_layout(title=f"Error: {str(e)}")
        return empty_fig, [], [], [], []

# Advisor Dashboard Callbacks
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
            return empty_fig, [], [], html.Div("No advisor selected")
            
        advisor_students = grade_sheet[grade_sheet['Advisor'] == selected_advisor]
        
        # Handle empty data
        if advisor_students.empty:
            empty_fig = go.Figure()
            empty_fig.update_layout(title="No students found for selected advisor")
            return empty_fig, [], [], html.Div("No students found")

        # GPA Trend
        fig = go.Figure()
        gpa_columns = [col for col in advisor_students.columns if 'GPA Semester' in col]
        gpa_columns.sort(key=lambda x: int(x.split()[-1]))
        
        for student in advisor_students.index:
            gpa_trend = [advisor_students.loc[student, col] for col in gpa_columns]
            semester_nums = [int(col.split()[-1]) for col in gpa_columns]
            fig.add_trace(go.Scatter(x=semester_nums, y=gpa_trend, mode='lines+markers', name=student))
        
        fig.update_layout(
            title=f'GPA Trends for Students of {selected_advisor}',
            xaxis_title='Semester',
            yaxis_title='GPA',
            yaxis_range=[2.0, 4.0]
        )

        # Students Table
        data = advisor_students[['ID', 'Major', 'Campus'] + 
                              gpa_columns].reset_index().rename(
            columns={'index': 'Student'}).to_dict('records')
        columns = [{'name': col, 'id': col} for col in ['Student', 'ID', 'Major', 'Campus'] + 
                  gpa_columns]

        # Courses
        courses = instructor_courses.get(selected_advisor, [])
        courses_list = html.Ul([html.Li(course) for course in courses])

        return fig, data, columns, courses_list
        
    except Exception as e:
        print(f"Error in advisor dashboard: {str(e)}")
        empty_fig = go.Figure()
        empty_fig.update_layout(title=f"Error: {str(e)}")
        return empty_fig, [], [], html.Div(f"Error: {str(e)}")

# Student Dashboard Callbacks
@app.callback(
    [Output('student-gpa-trend', 'figure'),
     Output('student-info', 'children'),
     Output('student-grade-sheet', 'data'),
     Output('student-grade-sheet', 'columns')],
    Input('student-selector', 'value')
)
def update_student_dashboard(selected_student):
    try:
        if not selected_student or grade_sheet.empty:
            empty_fig = go.Figure()
            empty_fig.update_layout(title="No data available")
            return empty_fig, html.Div("No student selected"), [], []
        
        # GPA Trend
        gpa_columns = [col for col in grade_sheet.columns if 'GPA Semester' in col]
        if not gpa_columns:
            empty_fig = go.Figure()
            empty_fig.update_layout(title="No GPA data available")
            return empty_fig, html.Div("No GPA data available"), [], []
            
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

        # Student Info
        info = grade_sheet.loc[selected_student]
        info_div = html.Div([
            html.P(f"ID: {info.get('ID', 'N/A')}"),
            html.P(f"Major: {info.get('Major', 'N/A')}"),
            html.P(f"Advisor: {info.get('Advisor', 'N/A')}"),
            html.P(f"Campus: {info.get('Campus', 'N/A')}")
        ])

        # Grade Sheet
        grade_columns = []
        for semester in range(1, 5):
            semester_cols = [col for col in grade_sheet.columns 
                           if f'Semester {semester}' in col]
            grade_columns.extend(semester_cols)
        
        # Only keep columns that exist in the data
        grade_columns = [col for col in grade_columns if col in grade_sheet.columns]
        
        # If no grade columns exist, return empty data
        if not grade_columns:
            return fig, info_div, [], []
        
        # Create a table with one row per semester
        student_grades = grade_sheet.loc[selected_student]
        
        data = []
        for semester in range(1, 5):
            semester_cols = [col for col in grade_columns 
                           if f'Semester {semester}' in col]
            if not semester_cols:
                continue
                
            row = {'Semester': f'Semester {semester}'}
            for col in semester_cols:
                assessment = col.split('Semester')[0].strip()
                row[assessment] = student_grades[col]
            data.append(row)
        
        # Get unique assessment types across all semesters
        assessment_types = sorted(set(col.split('Semester')[0].strip() 
                                    for col in grade_columns))
        
        columns = [{'name': 'Semester', 'id': 'Semester'}]
        columns.extend([{'name': assessment, 'id': assessment} 
                      for assessment in assessment_types])

        return fig, info_div, data, columns
        
    except Exception as e:
        print(f"Error in student dashboard: {str(e)}")
        empty_fig = go.Figure()
        empty_fig.update_layout(title=f"Error: {str(e)}")
        return empty_fig, html.Div(f"Error: {str(e)}"), [], []

# Campus Dashboard Callbacks
@app.callback(
    [Output('campus-major-distribution', 'figure'),
     Output('campus-avg-gpa-trend','figure'),
     Output('campus-students-table', 'data'),
     Output('campus-students-table', 'columns')],
    Input('campus-selector', 'value')
)
def update_campus_dashboard(selected_campus):
    try:
        if not selected_campus or grade_sheet.empty:
            empty_fig = go.Figure()
            empty_fig.update_layout(title="No data available")
            return empty_fig, empty_fig, [], []
            
        campus_students = grade_sheet[grade_sheet['Campus'] == selected_campus]
        
        # Handle empty data
        if campus_students.empty:
            empty_fig = go.Figure()
            empty_fig.update_layout(title="No students found for selected campus")
            return empty_fig, empty_fig, [], []

        # Major Distribution
        major_counts = campus_students['Major'].value_counts()
        major_fig = px.pie(
            values=major_counts.values,
            names=major_counts.index,
            title=f'Student Distribution by Major in {selected_campus}'
        )

        # Average GPA Trend
        gpa_columns = [col for col in campus_students.columns if 'GPA Semester' in col]
        gpa_columns.sort(key=lambda x: int(x.split()[-1]))
        
        avg_gpa = campus_students[gpa_columns].mean()
        semester_labels = [f'Semester {i+1}' for i in range(len(gpa_columns))]
        
        gpa_fig = px.bar(
            x=semester_labels,
            y=avg_gpa,
            labels={'x': 'Semester', 'y': 'Average GPA'},
            title=f'Average GPA Trend Across Semesters in {selected_campus}'
        )
        gpa_fig.update_yaxes(range=[2.0, 4.0])

        # Students Table
        data = campus_students[['ID', 'Major', 'Advisor'] + 
                             gpa_columns].reset_index().rename(
            columns={'index': 'Student'}).to_dict('records')
        columns = [{'name': col, 'id': col} for col in ['Student', 'ID', 'Major', 'Advisor'] + 
                  gpa_columns]

        return major_fig, gpa_fig, data, columns
        
    except Exception as e:
        print(f"Error in campus dashboard: {str(e)}")
        empty_fig = go.Figure()
        empty_fig.update_layout(title=f"Error: {str(e)}")
        return empty_fig, empty_fig, [], []

# Callbacks for Student Predictions
@app.callback(
    Output('student-next-semester-prediction', 'children'),
    Input('student-selector', 'value')
)
def update_student_next_semester_prediction(selected_student):
    if not selected_student or grade_sheet.empty:
        return html.Div("No data available")
        
    try:
        # Get performance prediction
        performance = predict_future_performance(grade_sheet, selected_student)
        
        if not performance:
            return html.Div("Could not generate prediction")
            
        # Create color-coded risk indicator
        if performance['at_risk_probability'] > 70:
            risk_color = "danger"
            risk_text = "High Risk"
        elif performance['at_risk_probability'] > 30:
            risk_color = "warning"
            risk_text = "Moderate Risk"
        else:
            risk_color = "success"
            risk_text = "On Track"
            
        # Create content
        return [
            dbc.Row([
                dbc.Col([
                    html.H5("Your Current GPA:"),
                    html.H3(f"{performance['current_gpa']}", className="text-primary")
                ], width=4, className="text-center"),
                dbc.Col([
                    html.H5("Predicted Next GPA:"),
                    html.H3(f"{performance['predicted_next_gpa']}", 
                          className="text-success" if performance['predicted_next_gpa'] >= performance['current_gpa'] 
                                 else "text-danger")
                ], width=4, className="text-center"),
                dbc.Col([
                    html.H5("Status:"),
                    dbc.Badge(risk_text, color=risk_color, className="p-2", style={"font-size": "1.2rem"})
                ], width=4, className="text-center d-flex align-items-center justify-content-center")
            ])
        ]
    except Exception as e:
        print(f"Error in student-next-semester-prediction: {str(e)}")
        return html.Div(f"Error generating prediction: {str(e)}")

# Callbacks for At-Risk Students
@app.callback(
    Output('advisor-at-risk-table', 'data'),
    [Input('advisor-selector', 'value'),
     Input('advisor-view-selector', 'value')]
)
def update_advisor_at_risk_table(selected_advisor, view_type):
    if not selected_advisor or grade_sheet.empty or view_type != 'at-risk':
        return []
        
    try:
        advisor_students = grade_sheet[grade_sheet['Advisor'] == selected_advisor]
        
        if advisor_students.empty:
            return []
            
        at_risk_df = detect_at_risk_students(advisor_students)
        
        if at_risk_df.empty:
            return []
            
        # Add additional intervention data
        at_risk_data = []
        for _, row in at_risk_df.iterrows():
            student = row['Student']
            prediction = predict_future_performance(grade_sheet, student)
            
            if prediction:
                at_risk_data.append({
                    'Student': student,
                    'Major': row['Major'],
                    'Campus': row['Campus'],
                    'Reason': row['Reason'],
                    'Risk Level': f"{prediction['at_risk_probability']}%",
                    'Current GPA': prediction['current_gpa'],
                    'Predicted GPA': prediction['predicted_next_gpa'],
                    'Intervention Needed': 'Yes' if prediction['intervention_needed'] else 'No'
                })
            else:
                at_risk_data.append({
                    'Student': student,
                    'Major': row['Major'],
                    'Campus': row['Campus'],
                    'Reason': row['Reason'],
                    'Risk Level': 'N/A',
                    'Current GPA': 'N/A',
                    'Predicted GPA': 'N/A',
                    'Intervention Needed': 'N/A'
                })
                
        return at_risk_data
    except Exception as e:
        print(f"Error in advisor-at-risk-table: {str(e)}")
        return []

@app.callback(
    Output('advisor-intervention-recommendations', 'children'),
    [Input('advisor-selector', 'value'),
     Input('advisor-view-selector', 'value')]
)
def update_advisor_intervention_recommendations(selected_advisor, view_type):
    if not selected_advisor or grade_sheet.empty or view_type != 'at-risk':
        return html.Div("No data available")
        
    try:
        # Get students advised by this advisor
        advisor_students = grade_sheet[grade_sheet['Advisor'] == selected_advisor]
        
        if advisor_students.empty:
            return html.Div("No students found for this advisor")
            
        # Get predictions and identify students needing intervention
        intervention_students = []
        
        for student in advisor_students.index:
            prediction = predict_future_performance(grade_sheet, student)
            if prediction and prediction['intervention_needed']:
                intervention_students.append({
                    'student': student,
                    'prediction': prediction
                })
                
        if not intervention_students:
            return html.Div("No students currently need intervention")
            
        # Create recommendation cards
        cards = []
        
        for student_data in intervention_students:
            student = student_data['student']
            prediction = student_data['prediction']
            
            # Create a card for each student
            cards.append(
                dbc.Card([
                    dbc.CardHeader(html.H5(student)),
                    dbc.CardBody([
                        html.P(f"Risk Level: {prediction['at_risk_probability']}%"),
                        html.P(f"Current GPA: {prediction['current_gpa']} → Predicted: {prediction['predicted_next_gpa']}"),
                        html.P(f"Weak Areas: {', '.join(prediction['weak_areas']) if prediction['weak_areas'] else 'None identified'}"),
                        html.P("Recommended Actions:"),
                        html.Ul([html.Li(rec) for rec in prediction['recommendations']])
                    ])
                ], className="mb-3")
            )
        
        return html.Div(cards)
    except Exception as e:
        print(f"Error in advisor-intervention-recommendations: {str(e)}")
        return html.Div(f"Error generating recommendations: {str(e)}")

# Courses Management Callbacks
@app.callback(
    [Output('advisor-courses-list', 'children'),
     Output('advisor-course-performance', 'figure')],
    [Input('advisor-selector', 'value'),
     Input('advisor-view-selector', 'value')]
)
def update_advisor_courses(selected_advisor, view_type):
    if not selected_advisor or grade_sheet.empty or view_type != 'courses':
        empty_fig = go.Figure()
        empty_fig.update_layout(title="No data available")
        return html.Div("No data available"), empty_fig
        
    try:
        # Get courses taught by this advisor
        courses = instructor_courses.get(selected_advisor, [])
        
        if not courses:
            empty_fig = go.Figure()
            empty_fig.update_layout(title="No courses found for this advisor")
            return html.Div("No courses found"), empty_fig
            
        # Create courses list with credits
        courses_list = html.Ul([
            html.Li([
                f"{course} ", 
                html.Span(f"({course_credits.get(course, 3)} credits)", 
                        className="text-muted")
            ]) for course in courses
        ])
        
        # Create course performance chart (placeholder)
        fig = go.Figure()
        
        for course in courses:
            # Simulate random performance data (in a real app, this would use actual grades)
            # This is just for demonstration purposes
            course_hash = hash(course) % 100
            avg_grades = [70 + (course_hash % 20)]
            for i in range(1, 5):
                # Add some random variation to simulate grade changes over time
                prev = avg_grades[-1]
                change = np.random.randint(-5, 6)
                new_val = max(60, min(100, prev + change))
                avg_grades.append(new_val)
            
            # Add the course data to the chart
            semesters = [f"Semester {i+1}" for i in range(len(avg_grades))]
            fig.add_trace(go.Scatter(
                x=semesters,
                y=avg_grades,
                mode='lines+markers',
                name=course
            ))
            
        fig.update_layout(
            title="Course Performance Trends",
            xaxis_title="Semester",
            yaxis_title="Average Grade",
            yaxis_range=[60, 100]
        )
        
        return courses_list, fig
    except Exception as e:
        print(f"Error in advisor-courses: {str(e)}")
        empty_fig = go.Figure()
        empty_fig.update_layout(title=f"Error: {str(e)}")
        return html.Div(f"Error: {str(e)}"), empty_fig

@app.callback(
    Output('advisor-course-details', 'children'),
    [Input('advisor-course-selector', 'value'),
     Input('advisor-selector', 'value')]
)
def update_advisor_course_details(selected_course, selected_advisor):
    if not selected_course or not selected_advisor:
        return html.Div("Select a course to view details")
        
    try:
        # Simulate course data (in a real app, this would use the API)
        return html.Div([
            html.H5(f"Course: {selected_course}"),
            html.P(f"Credits: {course_credits.get(selected_course, 3)}"),
            html.P(f"Instructor: {selected_advisor}"),
            html.H6("Student Performance Summary:"),
            dbc.Row([
                dbc.Col([
                    html.H6("Grade Distribution"),
                    dcc.Graph(figure=px.pie(
                        values=[20, 35, 25, 15, 5],
                        names=['A', 'B', 'C', 'D', 'F'],
                        title="Grade Distribution"
                    ), config={'displayModeBar': False})
                ], width=6),
                dbc.Col([
                    html.H6("Assessment Averages"),
                    dcc.Graph(figure=px.bar(
                        x=['Tests', 'Midterm', 'Project', 'Final'],
                        y=[78, 82, 85, 76],
                        title="Assessment Averages"
                    ), config={'displayModeBar': False})
                ], width=6)
            ])
        ])
    except Exception as e:
        print(f"Error in advisor-course-details: {str(e)}")
        return html.Div(f"Error loading course details: {str(e)}")

# Academic Planning Callbacks
@app.callback(
    Output('advisor-planning-student-selector', 'options'),
    [Input('advisor-selector', 'value'),
     Input('advisor-view-selector', 'value')]
)
def update_advisor_planning_students(selected_advisor, view_type):
    if not selected_advisor or grade_sheet.empty or view_type != 'planning':
        return []
        
    try:
        advisor_students = grade_sheet[grade_sheet['Advisor'] == selected_advisor]
        
        if advisor_students.empty:
            return []
            
        return [{'label': student, 'value': student} for student in advisor_students.index]
    except Exception as e:
        print(f"Error in advisor-planning-student-selector: {str(e)}")
        return []

@app.callback(
    [Output('advisor-student-timeline', 'figure'),
     Output('advisor-graduation-outlook', 'children')],
    [Input('advisor-planning-student-selector', 'value'),
     Input('advisor-selector', 'value'),
     Input('advisor-view-selector', 'value')]
)
def update_advisor_student_planning(selected_student, selected_advisor, view_type):
    if not selected_student or not selected_advisor or grade_sheet.empty or view_type != 'planning':
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Select a student to view timeline")
        return empty_fig, html.Div("Select a student to view graduation outlook")
        
    try:
        # Get graduation outlook
        outlook = predict_graduation_outlook(grade_sheet, selected_student)
        
        if not outlook:
            empty_fig = go.Figure()
            empty_fig.update_layout(title="Could not generate timeline")
            return empty_fig, html.Div("Could not generate graduation outlook")
            
        # Create timeline figure
        fig = go.Figure()
        
        # Get GPA columns
        gpa_columns = [col for col in grade_sheet.columns if 'GPA Semester' in col]
        gpa_columns.sort(key=lambda x: int(x.split()[-1]))
        
        # Past semesters GPA
        semesters = []
        gpa_values = []
        
        for i, col in enumerate(gpa_columns):
            semester_num = i + 1
            semesters.append(f"Semester {semester_num}")
            gpa_values.append(grade_sheet.loc[selected_student, col])
            
        # Add past semesters
        fig.add_trace(go.Scatter(
            x=semesters,
            y=gpa_values,
            mode='lines+markers',
            name='Past GPA',
            line=dict(color='blue', width=3),
            marker=dict(size=10)
        ))
        
        # Add predicted future semesters
        future_semesters = []
        future_gpas = []
        
        current_semester = len(gpa_columns)
        remaining_semesters = outlook['estimated_semesters_to_graduation']
        
        # Get last GPA or prediction for next GPA
        if gpa_values:
            last_gpa = gpa_values[-1]
        else:
            last_gpa = 3.0  # Default if no GPA history
            
        # Get predicted GPA
        prediction = predict_future_performance(grade_sheet, selected_student)
        if prediction:
            next_gpa = prediction['predicted_next_gpa']
        else:
            next_gpa = last_gpa  # Use last GPA if prediction not available
            
        # Generate future GPA trend
        for i in range(1, remaining_semesters + 1):
            semester_num = current_semester + i
            future_semesters.append(f"Semester {semester_num}")
            
            if i == 1:
                future_gpas.append(next_gpa)
            else:
                # Simple projection based on graduation likelihood
                if 'Excellent' in outlook['graduation_likelihood']:
                    trend = 0.1
                elif 'Good' in outlook['graduation_likelihood']:
                    trend = 0.05
                elif 'Fair' in outlook['graduation_likelihood']:
                    trend = 0
                elif 'Concerning' in outlook['graduation_likelihood']:
                    trend = -0.05
                else:  # At Risk
                    trend = -0.1
                    
                future_gpas.append(max(2.0, min(4.0, future_gpas[-1] + trend)))
                
        # Add future semesters
        fig.add_trace(go.Scatter(
            x=future_semesters,
            y=future_gpas,
            mode='lines+markers',
            name='Projected GPA',
            line=dict(color='green', dash='dash', width=3),
            marker=dict(size=10)
        ))
        
        # Add threshold line for graduation requirement
        fig.add_shape(
            type="line",
            x0=semesters[0] if semesters else future_semesters[0],
            y0=2.5,
            x1=future_semesters[-1] if future_semesters else semesters[-1],
            y1=2.5,
            line=dict(color="red", width=2, dash="dash")
        )
        
        fig.update_layout(
            title=f'Academic Progression Timeline for {selected_student}',
            xaxis_title='Semester',
            yaxis_title='GPA',
            yaxis_range=[2.0, 4.0],
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        
        # Create graduation outlook card
        outlook_card = dbc.Card([
            dbc.CardHeader(html.H5("Graduation Outlook Summary")),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.P("Current Status:", className="font-weight-bold"),
                        html.P(f"Completed Semesters: {outlook['current_semester']}"),
                        html.P(f"Current Average GPA: {outlook['current_avg_gpa']}")
                    ], width=6),
                    dbc.Col([
                        html.P("Graduation Projection:", className="font-weight-bold"),
                        html.P(f"Likelihood: {outlook['graduation_likelihood']}"),
                        html.P(f"Estimated Graduation: {outlook['estimated_graduation_date']}")
                    ], width=6)
                ]),
                html.Hr(),
                html.P("Academic Trajectory:", className="font-weight-bold"),
                html.P(f"GPA Trend: {outlook['gpa_trend']}"),
                html.P("Faculty Recommendations:", className="font-weight-bold"),
                html.Ul([html.Li(rec) for rec in outlook['recommendations']])
            ])
        ])
        
        return fig, outlook_card
    except Exception as e:
        print(f"Error in advisor-student-planning: {str(e)}")
        empty_fig = go.Figure()
        empty_fig.update_layout(title=f"Error: {str(e)}")
        return empty_fig, html.Div(f"Error: {str(e)}")

# Office Hours Callbacks
@app.callback(
    Output('advisor-appointments', 'children'),
    [Input('advisor-selector', 'value'),
     Input('advisor-view-selector', 'value'),
     Input('office-day-selector', 'value')]
)
def update_advisor_appointments(selected_advisor, view_type, selected_days):
    if not selected_advisor or view_type != 'office' or not selected_days:
        return html.Div("No appointments to display")
        
    try:
        # In a real application, this would fetch from a database
        # For this example, we'll create sample appointments
        
        # Get advisor's students
        if grade_sheet.empty:
            return html.Div("No students found")
            
        advisor_students = grade_sheet[grade_sheet['Advisor'] == selected_advisor]
        
        if advisor_students.empty:
            return html.Div("No students found for this advisor")
            
        # Simulate some appointments
        appointments = []
        days_map = {
            'monday': 'Monday',
            'tuesday': 'Tuesday',
            'wednesday': 'Wednesday',
            'thursday': 'Thursday',
            'friday': 'Friday'
        }
        
        # Generate a few random appointments
        import random
        random.seed(hash(selected_advisor))  # Use advisor name as seed for consistent results
        
        times = ['9:00 AM', '10:30 AM', '1:00 PM', '2:30 PM', '4:00 PM']
        topics = ['Course Planning', 'Academic Progress', 'Career Advice', 'Graduation Requirements', 'Research Opportunities']
        
        for day in selected_days:
            day_name = days_map.get(day, day)
            
            # Add 1-3 appointments per day
            num_appointments = random.randint(1, 3)
            for _ in range(num_appointments):
                student = random.choice(list(advisor_students.index))
                time = random.choice(times)
                topic = random.choice(topics)
                
                appointments.append({
                    'day': day_name,
                    'time': time,
                    'student': student,
                    'topic': topic
                })
        
        # Sort appointments by day and time
        day_order = {day: i for i, day in enumerate(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])}
        appointments.sort(key=lambda x: (day_order.get(x['day'], 0), x['time']))
        
        # Create the appointments list
        if not appointments:
            return html.Div("No appointments scheduled")
            
        appointment_cards = []
        for appt in appointments:
            appointment_cards.append(
                dbc.Card([
                    dbc.CardBody([
                        html.H5(f"{appt['day']} - {appt['time']}"),
                        html.P([
                            html.Strong("Student: "), appt['student']
                        ]),
                        html.P([
                            html.Strong("Topic: "), appt['topic']
                        ]),
                        dbc.ButtonGroup([
                            dbc.Button("Reschedule", color="secondary", size="sm", className="mr-1"),
                            dbc.Button("Cancel", color="danger", size="sm")
                        ])
                    ])
                ], className="mb-2")
            )
        
        return html.Div(appointment_cards)
    except Exception as e:
        print(f"Error in advisor-appointments: {str(e)}")
        return html.Div(f"Error loading appointments: {str(e)}")



port = int(os.environ.get('PORT', 8050))
app.run_server(debug=True, host='0.0.0.0', port=port)