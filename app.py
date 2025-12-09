from flask import Flask, request, render_template, jsonify, redirect, url_for, flash
from flask_login import LoginManager, current_user, login_required
from werkzeug.security import generate_password_hash
import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime

# Import multimodal database models
from models_multimodal import db, User, PredictionHistory, MediaFile, MultimodalPrediction

# Import authentication routes
from auth_routes import auth_bp

# Import multimodal routes
from multimodal_routes import multimodal_bp

# Import image routes
# NOTE: Requires torch installation: pip install torch torchvision
# Uncomment below after installing torch
# from image_routes import image_bp

# Import processing modules
# from image_processing import process_medical_image, analyze_image_quality
# Note: audio_processing and multimodal_fusion will be implemented in Week 3-4
# from audio_processing import process_medical_audio
# from multimodal_fusion import fuse_medical_predictions

# Flask app initialization
app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = 'your-secret-key-change-this-in-production'  # Change this!
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///mrs_database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# File upload configuration
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize database
db.init_app(app)

# Create database tables
with app.app_context():
    db.create_all()
    print("Database tables created/verified!")

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth.login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Register blueprints
app.register_blueprint(auth_bp, url_prefix='')  # Remove /auth prefix to use /login and /register directly
app.register_blueprint(multimodal_bp)
# Uncomment after installing torch
# app.register_blueprint(image_bp)

# Load datasets
script_dir = os.path.dirname(os.path.abspath(__file__))
sym_des = pd.read_csv(os.path.join(script_dir, "Dataset", "symtoms_df.csv"))
precautions = pd.read_csv(os.path.join(script_dir, "Dataset", "precautions_df.csv"))
workout = pd.read_csv(os.path.join(script_dir, "Dataset", "workout_df.csv"))
description = pd.read_csv(os.path.join(script_dir, "Dataset", "description.csv"))
medications = pd.read_csv(os.path.join(script_dir, "Dataset", "medications.csv"))
diets = pd.read_csv(os.path.join(script_dir, "Dataset", "diets.csv"))

# Load model
svc = pickle.load(open(os.path.join(script_dir, 'models', 'svc.pkl'), 'rb'))

# Helper function
def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]

    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values]

    wrkout = workout[workout['disease'] == dis]['workout']

    return desc, pre, med, die, wrkout

# Symptoms dictionary
symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}

diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

# Model prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]

# Routes
@app.route("/")
def index():
    # Show landing page
    return render_template('landing.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')

        if not symptoms or symptoms.strip() == "":
            message = "Please enter your symptoms."
            return render_template('index.html', message=message)
        elif symptoms == "Symptoms":
            message = "Please either write symptoms or you have written misspelled symptoms"
            return render_template('index.html', message=message)
        else:
            # Split the user's input into a list of symptoms
            user_symptoms = [s.strip() for s in symptoms.split(',')]
            user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]
            predicted_disease = get_predicted_value(user_symptoms)
            dis_des, precautions_list, medications_list, rec_diet, workout_list = helper(predicted_disease)

            my_precautions = []
            for i in precautions_list[0]:
                my_precautions.append(i)

            # Save to database if user is logged in
            if current_user.is_authenticated:
                prediction = PredictionHistory(
                    user_id=current_user.id,
                    symptoms=symptoms,
                    predicted_disease=predicted_disease,
                    description=dis_des,
                    precautions=str(my_precautions),
                    medications=str(medications_list),
                    diet=str(rec_diet),
                    workout=str(workout_list)
                )
                db.session.add(prediction)
                db.session.commit()

            return render_template('index.html',
                                   predicted_disease=predicted_disease,
                                   dis_des=dis_des,
                                   my_precautions=my_precautions,
                                   medications=medications_list,
                                   my_diet=rec_diet,
                                   workout=workout_list)

    return render_template('index.html')

@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard showing prediction history"""
    # Get text-only predictions (legacy)
    text_predictions = PredictionHistory.query.filter_by(user_id=current_user.id).order_by(PredictionHistory.created_at.desc()).all()

    # Get multimodal predictions (image + text)
    multimodal_predictions = MultimodalPrediction.query.filter_by(user_id=current_user.id).order_by(MultimodalPrediction.created_at.desc()).all()

    # Combine both types for display
    all_predictions = []

    # Add text predictions
    for pred in text_predictions:
        all_predictions.append({
            'type': 'text',
            'predicted_disease': pred.predicted_disease,
            'symptoms': pred.symptoms,
            'description': pred.description,
            'created_at': pred.created_at,
            'confidence': pred.confidence if pred.confidence else None
        })

    # Add multimodal predictions
    for pred in multimodal_predictions:
        import json
        recommendations = json.loads(pred.recommendations) if pred.recommendations else {}

        all_predictions.append({
            'type': 'multimodal',
            'predicted_disease': pred.final_prediction,
            'symptoms': pred.text_symptoms,
            'description': recommendations.get('description', ''),
            'created_at': pred.created_at,
            'confidence': pred.combined_confidence,
            'fusion_method': pred.fusion_method
        })

    # Sort by date descending
    all_predictions.sort(key=lambda x: x['created_at'], reverse=True)

    return render_template('dashboard.html', predictions=all_predictions)

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/developer')
def developer():
    return render_template("developer.html")

@app.route('/blog')
def blog():
    return render_template("blog.html")

@app.route('/symptoms')
def symptoms():
    return render_template("symptoms.html")

# Multimodal prediction routes
# NOTE: This old route is DEPRECATED and uses undefined functions.
# Use the new multimodal blueprint routes instead:
# - /multimodal/predict (POST) - Main prediction endpoint
# - /multimodal/upload_page (GET) - Upload interface
# - /multimodal/compare (POST) - Compare fusion methods
# 
# @app.route('/multimodal-predict', methods=['GET', 'POST'])
# @login_required
# def multimodal_predict():
#     """Multimodal disease prediction with text, image, and audio"""
#     # This route requires:
#     # - process_medical_image() from image_processing
#     # - process_medical_audio() from audio_processing  
#     # - fuse_medical_predictions() from multimodal_fusion
#     # These are not currently imported/available.
#     # Use /multimodal/predict instead which uses multimodal_fusion_derm.py
#     pass

@app.route('/multimodal-history')
@login_required
def multimodal_history():
    """View multimodal prediction history"""
    predictions = MultimodalPrediction.query.filter_by(user_id=current_user.id).order_by(
        MultimodalPrediction.created_at.desc()
    ).all()
    return render_template('multimodal_history.html', predictions=predictions)

# Database initialization
def init_db():
    """Initialize the database"""
    with app.app_context():
        db.create_all()
        print("Database tables created successfully!")

        # Create a demo user if none exists
        if User.query.count() == 0:
            demo_user = User(
                username='demo',
                email='demo@example.com',
                first_name='Demo',
                last_name='User'
            )
            demo_user.set_password('demo123')
            db.session.add(demo_user)
            db.session.commit()
            print("Demo user created: username='demo', password='demo123'")

if __name__ == '__main__':
    # Initialize database on first run
    init_db()

    # Run the app
    app.run(host='127.0.0.1', port=8000, debug=True)
