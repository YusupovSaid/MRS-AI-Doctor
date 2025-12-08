"""
Multimodal Prediction Routes
API endpoints for text + image fusion predictions
"""

from flask import Blueprint, request, jsonify, render_template
from flask_login import current_user, login_required
from werkzeug.utils import secure_filename
import os
import pandas as pd
from datetime import datetime

# Import fusion engine
from multimodal_fusion_derm import (
    get_fusion_engine,
    predict_with_fusion,
    compare_fusion_methods
)

# Load recommendation datasets
script_dir = os.path.dirname(os.path.abspath(__file__))
description = pd.read_csv(os.path.join(script_dir, "Dataset", "description.csv"))
precautions = pd.read_csv(os.path.join(script_dir, "Dataset", "precautions_df.csv"))
medications = pd.read_csv(os.path.join(script_dir, "Dataset", "medications.csv"))
diets = pd.read_csv(os.path.join(script_dir, "Dataset", "diets.csv"))
workout = pd.read_csv(os.path.join(script_dir, "Dataset", "workout_df.csv"))

# Load skin diseases recommendations (NEW)
try:
    skin_diseases = pd.read_csv(os.path.join(script_dir, "Dataset", "skin_diseases_recommendations.csv"))
    print("✅ Skin diseases recommendations loaded")
except Exception as e:
    print(f"⚠️  Skin diseases recommendations not found: {e}")
    skin_diseases = None

def get_disease_recommendations(disease_name):
    """Get full recommendations for a disease (supports both general and skin diseases)"""

    # Try skin diseases first (new dataset)
    if skin_diseases is not None:
        skin_match = skin_diseases[skin_diseases['Disease'] == disease_name]
        if not skin_match.empty:
            row = skin_match.iloc[0]
            return {
                'description': row['Description'],
                'precautions': [
                    row['Precaution_1'],
                    row['Precaution_2'],
                    row['Precaution_3'],
                    row['Precaution_4']
                ],
                'medications': [row['Medications']] if pd.notna(row.get('Medications')) else ["Consult doctor for prescription"],
                'diet': [row['Diet']] if pd.notna(row.get('Diet')) else ["Balanced diet recommended"],
                'workout': [row['Workout']] if pd.notna(row.get('Workout')) else ["Follow doctor's activity recommendations"]
            }

    # Fallback to general diseases (original dataset)
    # Description
    desc = description[description['Disease'] == disease_name]['Description']
    desc = " ".join([w for w in desc]) if not desc.empty else f"Medical condition: {disease_name}"

    # Precautions
    pre = precautions[precautions['Disease'] == disease_name][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre_list = []
    if not pre.empty:
        for col in pre.values[0]:
            if pd.notna(col):
                pre_list.append(col)

    if not pre_list:
        pre_list = ["Consult a healthcare professional", "Follow prescribed treatment", "Get adequate rest"]

    # Medications
    med = medications[medications['Disease'] == disease_name]['Medication']
    med_list = [m for m in med.values] if not med.empty else ["Consult doctor for prescription"]

    # Diet
    die = diets[diets['Disease'] == disease_name]['Diet']
    die_list = [d for d in die.values] if not die.empty else ["Follow balanced diet", "Stay hydrated"]

    # Workout
    wrk = workout[workout['disease'] == disease_name]['workout']
    wrk_list = [w for w in wrk.values] if not wrk.empty else ["Light exercise as recommended", "Adequate rest"]

    return {
        'description': desc,
        'precautions': pre_list,
        'medications': med_list,
        'diet': die_list,
        'workout': wrk_list
    }

# Create blueprint
multimodal_bp = Blueprint('multimodal', __name__, url_prefix='/multimodal')

# Configuration
UPLOAD_FOLDER = os.path.join(script_dir, 'uploads', 'multimodal')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@multimodal_bp.route('/predict', methods=['POST'])
def multimodal_predict():
    """
    Multimodal prediction endpoint
    Accepts text symptoms and/or image

    Request:
        - symptoms (form): Comma-separated symptoms (optional)
        - image (file): Medical image (optional)

    Returns:
        JSON with prediction results
    """
    try:
        # Get inputs
        symptoms_text = request.form.get('symptoms', None)
        image_file = request.files.get('image', None)

        # DEBUG: Print received form data
        print(f"\n🔍 DEBUG /multimodal/predict endpoint:")
        print(f"   symptoms_text: {repr(symptoms_text)}")
        print(f"   image_file: {repr(image_file)}")
        if image_file:
            print(f"   image_file.filename: {image_file.filename}")

        # Validate inputs
        if not symptoms_text and not image_file:
            return jsonify({
                'success': False,
                'error': 'Please provide symptoms and/or image'
            }), 400

        # Process image if provided
        image_path = None
        if image_file and allowed_file(image_file.filename):
            filename = secure_filename(image_file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            image_path = os.path.join(UPLOAD_FOLDER, filename)
            image_file.save(image_path)
            print(f"   ✅ Image saved to: {image_path}")

        # Run prediction
        result = predict_with_fusion(
            symptoms_text=symptoms_text,
            image_path=image_path
        )

        # Check for errors
        if 'error' in result:
            return jsonify({
                'success': False,
                'error': result['error'],
                'fallback_needed': result.get('fallback_needed', False)
            }), 500

        # Get disease info
        predicted_disease = result.get('predicted_disease', 'Unknown')

        # Get full recommendations from datasets
        recommendations = get_disease_recommendations(predicted_disease)

        # Build response with FULL recommendations
        response = {
            'success': True,
            'predicted_disease': predicted_disease,
            'confidence': result.get('confidence', 0.0),
            'fusion_method': result.get('fusion_method', 'unknown'),
            'model_used': result.get('model', 'unknown'),
            'modality': result.get('modality', 'unknown'),

            # Additional info
            'fusion_info': result.get('fusion_info', {}),
            'top3_predictions': result.get('top3_predictions', []),
            'severity': result.get('severity'),
            'urgency': result.get('urgency'),
            'condition_type': result.get('condition_type'),
            'is_severe': result.get('is_severe', False),

            # FULL Recommendations from trained model
            'description': recommendations['description'],
            'precautions': recommendations['precautions'],
            'medications': recommendations['medications'],
            'diet': recommendations['diet'],
            'workout': recommendations['workout']
        }

        # Save to database if user is logged in
        if current_user.is_authenticated:
            from models_multimodal import db, MultimodalPrediction, MediaFile
            import json

            try:
                # Create multimodal prediction record
                prediction_record = MultimodalPrediction(
                    user_id=current_user.id,
                    text_symptoms=symptoms_text,
                    final_prediction=predicted_disease,
                    fusion_method=result.get('fusion_method', 'unknown'),
                    combined_confidence=result.get('confidence', 0.0),
                    recommendations=json.dumps({
                        'description': recommendations['description'],
                        'precautions': recommendations['precautions'],
                        'medications': recommendations['medications'],
                        'diet': recommendations['diet'],
                        'workout': recommendations['workout']
                    })
                )

                # Add image predictions if available
                if result.get('image_prediction'):
                    prediction_record.image_prediction = json.dumps(result['image_prediction'])

                # Add text predictions if available
                if result.get('text_prediction'):
                    prediction_record.text_prediction = json.dumps(result['text_prediction'])

                db.session.add(prediction_record)

                # Create media file record if image was uploaded
                if image_path:
                    media_record = MediaFile(
                        user_id=current_user.id,
                        file_type='image',
                        file_path=image_path,
                        storage_backend='local',
                        original_filename=image_file.filename if image_file else 'unknown',
                        image_type='skin',
                        processing_status='completed'
                    )
                    db.session.add(media_record)

                db.session.commit()
                print(f"✅ Saved consultation to database for user {current_user.id}")

            except Exception as db_error:
                db.session.rollback()
                print(f"⚠️ Failed to save consultation to database: {db_error}")
                # Don't fail the request if database save fails

        return jsonify(response), 200

    except Exception as e:
        import traceback
        traceback.print_exc()  # Print full error to console
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Alias for /predict (frontend compatibility)
@multimodal_bp.route('/analyze', methods=['POST'])
def multimodal_analyze():
    """Alias for multimodal_predict for frontend compatibility"""
    return multimodal_predict()


@multimodal_bp.route('/compare', methods=['POST'])
def compare_fusion():
    """
    Compare Late Fusion vs Embedding Fusion
    Requires both symptoms and image

    Request:
        - symptoms (form): Comma-separated symptoms
        - image (file): Medical image

    Returns:
        JSON with comparison results
    """
    try:
        # Get inputs
        symptoms_text = request.form.get('symptoms', None)
        image_file = request.files.get('image', None)

        # Validate inputs
        if not symptoms_text or not image_file:
            return jsonify({
                'success': False,
                'error': 'Both symptoms and image are required for comparison'
            }), 400

        # Process image
        if not allowed_file(image_file.filename):
            return jsonify({
                'success': False,
                'error': 'Invalid file type'
            }), 400

        filename = secure_filename(image_file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        image_file.save(image_path)

        # Compare fusion methods
        comparison = compare_fusion_methods(symptoms_text, image_path)

        return jsonify({
            'success': True,
            'comparison': comparison
        }), 200

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@multimodal_bp.route('/config', methods=['GET'])
def get_config():
    """
    Test endpoint to verify multimodal system is working

    Returns:
        JSON with system status
    """
    engine = get_fusion_engine()

    return jsonify({
        'status': 'ok',
        'image_model': 'Tanishq Skin Classifier (95.60% accuracy)',
        'svc_model_loaded': engine.svc_model is not None,
        'endpoints': {
            'predict': '/multimodal/predict',
            'compare': '/multimodal/compare',
            'config': '/multimodal/config',
            'upload_page': '/multimodal/upload_page'
        }
    }), 200


@multimodal_bp.route('/upload_page', methods=['GET'])
@login_required
def upload_page():
    """
    Render multimodal upload page
    Requires user authentication

    Returns:
        HTML page for multimodal predictions
    """
    return render_template('multimodal_upload.html')


# Error handlers
@multimodal_bp.errorhandler(413)
def file_too_large(error):
    """Handle file too large error"""
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size is 16MB'
    }), 413


@multimodal_bp.errorhandler(500)
def internal_error(error):
    """Handle internal server error"""
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


if __name__ == "__main__":
    print("Multimodal routes blueprint created")
    print("Endpoints:")
    print("  POST /multimodal/predict - Main prediction endpoint")
    print("  POST /multimodal/compare - Compare fusion methods")
    print("  GET  /multimodal/config - Get configuration")
    print("  GET  /multimodal/test - Test system status")
    print("  GET  /multimodal/upload_page - Upload page UI")
