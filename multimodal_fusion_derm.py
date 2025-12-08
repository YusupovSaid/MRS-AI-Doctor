"""
Multimodal Fusion Engine
Uses Tanishq Skin Classifier (95.60% accuracy) for images
Uses SVC model for text symptoms
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import pickle
import os
import pandas as pd

# Load symptom dictionary for SVC model
SYMPTOMS_DICT = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anus': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}

# Disease mapping (41 diseases)
DISEASES_LIST = {
    15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis',
    14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ',
    17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ',
    30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)',
    28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid',
    40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D',
    22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis',
    10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)',
    18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism',
    24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis',
    5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne',
    38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'
}


class MultimodalFusionEngine:
    """
    Engine for fusing predictions from multiple modalities
    Uses Tanishq Skin Classifier (95.60% accuracy) for images
    Uses SVC model for text symptoms
    """

    def __init__(self):
        # Load existing SVC model for text
        self.svc_model = self._load_svc_model()

    def _load_svc_model(self):
        """Load existing SVC model for text symptom analysis"""
        try:
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'svc.pkl')
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print("✓ SVC model loaded for text analysis")
            return model
        except Exception as e:
            print(f"Warning: Could not load SVC model: {e}")
            return None

    def predict_text_only(self, symptoms_text: str) -> Dict:
        """
        Predict using text symptoms only (SVC model)

        Args:
            symptoms_text: Comma-separated symptoms

        Returns:
            Prediction results
        """
        if not self.svc_model:
            return {
                'modality': 'text',
                'error': 'SVC model not available',
                'fallback_needed': True
            }

        try:
            # Parse symptoms from comma-separated text
            user_symptoms = [s.strip().lower().replace(' ', '_') for s in symptoms_text.split(',')]
            user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]

            print(f"\n🔍 DEBUG predict_text_only:")
            print(f"   Raw symptoms: {user_symptoms}")

            # Create input vector
            input_vector = np.zeros(len(SYMPTOMS_DICT))
            matched_symptoms = 0
            matched_list = []
            unmatched_list = []

            for symptom in user_symptoms:
                if symptom in SYMPTOMS_DICT:
                    input_vector[SYMPTOMS_DICT[symptom]] = 1
                    matched_symptoms += 1
                    matched_list.append(symptom)
                else:
                    unmatched_list.append(symptom)

            print(f"   Matched symptoms ({matched_symptoms}): {matched_list}")
            print(f"   Unmatched symptoms: {unmatched_list}")

            if matched_symptoms == 0:
                return {
                    'modality': 'text',
                    'error': 'No valid symptoms recognized',
                    'fallback_needed': True
                }

            # Use REAL SVC model prediction
            predicted_class = self.svc_model.predict([input_vector])[0]
            predicted_disease = DISEASES_LIST.get(predicted_class, 'Unknown')

            # Get prediction probabilities for confidence
            # SVC model has probability=True, so we ALWAYS use predict_proba
            try:
                proba = self.svc_model.predict_proba([input_vector])[0]
                confidence = float(np.max(proba))

                # Get top 3 predictions for debugging
                top3_idx = np.argsort(proba)[-3:][::-1]
                print(f"   ✅ Using predict_proba (TRUE probabilities):")
                print(f"      Predicted: {predicted_disease} ({confidence:.1%})")
                for idx in top3_idx:
                    disease_name = DISEASES_LIST.get(idx, f'Class_{idx}')
                    print(f"      {idx}: {disease_name} = {proba[idx]:.1%}")
            except Exception as e:
                # Fallback to decision_function (shouldn't happen)
                print(f"   ⚠️  predict_proba failed: {e}")
                decision = self.svc_model.decision_function([input_vector])[0]
                max_decision = np.max(decision)
                confidence = min(0.95, max(0.5, (max_decision + 3) / 6))
                print(f"   Using decision_function fallback: max={max_decision:.4f}, normalized={confidence:.4f}")

            return {
                'modality': 'text',
                'model': 'svc',
                'fusion_method': 'text_only',
                'confidence': float(confidence),
                'predicted_class': int(predicted_class),
                'predicted_disease': predicted_disease,
                'matched_symptoms': matched_symptoms,
                'total_symptoms': len(user_symptoms)
            }

        except Exception as e:
            print(f"Error in text prediction: {e}")
            return {
                'modality': 'text',
                'error': str(e),
                'fallback_needed': True
            }

    def predict_image_only(self, image_path: str) -> Dict:
        """
        Predict using image only with Tanishq Skin Classifier
        Uses Tanishq77/skin-condition-classifier (95.60% accuracy)

        Args:
            image_path: Path to medical image

        Returns:
            Prediction results
        """
        try:
            # Import Tanishq classifier
            from tanishq_skin_classifier import get_tanishq_classifier

            # Get Tanishq instance
            classifier = get_tanishq_classifier()

            # Run prediction
            result = classifier.predict(image_path)

            if not result.get('success', False):
                print("⚠️  Tanishq model prediction failed")
                return self._predict_image_fallback(image_path)

            # Extract prediction results
            predicted_disease = result['predicted_disease']
            confidence = result['confidence']
            severity = result.get('severity', 'unknown')
            urgency = result.get('urgency', 'unknown')
            top3_predictions = result.get('top3_predictions', [])

            # Return in expected format
            return {
                'modality': 'image',
                'model': 'tanishq_skin',
                'fusion_method': 'image_only',
                'confidence': float(confidence),
                'predicted_class': -1,  # Not using old class system
                'predicted_disease': predicted_disease,
                'severity': severity,
                'urgency': urgency,
                'condition_type': result.get('condition_type'),
                'is_severe': result.get('is_severe', False),
                'top3_predictions': top3_predictions,
                'model_accuracy': 0.956,
                'architecture': 'EfficientNetV2B0',
                'source': 'tanishq_skin_classifier'
            }

        except ImportError as e:
            print(f"⚠️  Tanishq classifier not available: {e}")
            return self._predict_image_fallback(image_path)
        except Exception as e:
            print(f"Error in Tanishq prediction: {e}")
            import traceback
            traceback.print_exc()
            return self._predict_image_fallback(image_path)

    def _predict_image_fallback(self, image_path: str) -> Dict:
        """
        Fallback image prediction (basic stub)
        Returns error since Tanishq model should always be available

        Args:
            image_path: Path to image

        Returns:
            Error dict
        """
        return {
            'modality': 'image',
            'error': 'Tanishq skin classifier not available and no fallback',
            'fallback_needed': True,
            'predicted_disease': 'Unknown',
            'confidence': 0.0
        }

    def late_fusion(
        self,
        text_prediction: Dict,
        image_prediction: Dict,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Late Fusion: Combine predictions from separate models

        Args:
            text_prediction: Text model prediction
            image_prediction: Image model prediction
            weights: Optional weights for each modality

        Returns:
            Fused prediction
        """
        if weights is None:
            weights = {'text': 0.5, 'image': 0.5}

        # Weighted average of confidences
        text_conf = text_prediction.get('confidence', 0.5)
        image_conf = image_prediction.get('confidence', 0.5)

        combined_confidence = (
            weights['text'] * text_conf +
            weights['image'] * image_conf
        )

        # Choose prediction with higher confidence
        if text_conf > image_conf:
            final_prediction = text_prediction.copy()
            final_prediction['fusion_info'] = {
                'method': 'late_fusion',
                'text_conf': text_conf,
                'image_conf': image_conf,
                'winner': 'text'
            }
        else:
            final_prediction = image_prediction.copy()
            final_prediction['fusion_info'] = {
                'method': 'late_fusion',
                'text_conf': text_conf,
                'image_conf': image_conf,
                'winner': 'image'
            }

        final_prediction['combined_confidence'] = combined_confidence
        final_prediction['fusion_method'] = 'late_fusion'

        return final_prediction

    def embedding_fusion(
        self,
        symptoms_text: str,
        image_path: str
    ) -> Dict:
        """
        Embedding-Level Fusion: NOT SUPPORTED (removed Google Derm Foundation)
        Falls back to late fusion

        Args:
            symptoms_text: Comma-separated symptoms
            image_path: Path to medical image

        Returns:
            Error dict indicating fallback needed
        """
        print("⚠️  Embedding fusion not supported, use late fusion instead")
        return {
            'error': 'Embedding fusion removed (Google Derm Foundation)',
            'fallback_needed': True
        }

    def predict_multimodal(
        self,
        symptoms_text: Optional[str] = None,
        image_path: Optional[str] = None
    ) -> Dict:
        """
        Main multimodal prediction function
        Uses Tanishq model for images and SVC model for text

        Args:
            symptoms_text: Comma-separated symptoms (optional)
            image_path: Path to medical image (optional)

        Returns:
            Prediction results with recommendations
        """
        # DEBUG: Print received inputs
        print(f"\n🔍 DEBUG predict_multimodal:")
        print(f"   symptoms_text: {repr(symptoms_text)}")
        print(f"   image_path: {repr(image_path)}")

        # Handle single modality
        if symptoms_text and not image_path:
            print("   → Using TEXT-ONLY prediction")
            return self.predict_text_only(symptoms_text)

        if image_path and not symptoms_text:
            print("   → Using IMAGE-ONLY prediction")
            return self.predict_image_only(image_path)

        # Both modalities available - use late fusion
        if symptoms_text and image_path:
            print("   → Using LATE FUSION (text + image)")
            text_pred = self.predict_text_only(symptoms_text)
            image_pred = self.predict_image_only(image_path)
            return self.late_fusion(text_pred, image_pred)

        print("   ❌ ERROR: No input data provided")
        return {
            'error': 'No symptoms text or image provided'
        }


# Global engine instance
_engine = None

def get_fusion_engine() -> MultimodalFusionEngine:
    """Get global fusion engine instance"""
    global _engine
    if _engine is None:
        _engine = MultimodalFusionEngine()
    return _engine


def predict_with_fusion(
    symptoms_text: Optional[str] = None,
    image_path: Optional[str] = None
) -> Dict:
    """
    Convenience function for multimodal prediction

    Args:
        symptoms_text: Comma-separated symptoms
        image_path: Path to medical image

    Returns:
        Prediction with disease, confidence, and recommendations
    """
    engine = get_fusion_engine()
    return engine.predict_multimodal(symptoms_text, image_path)


# Comparison function
def compare_fusion_methods(symptoms_text: str, image_path: str) -> Dict:
    """
    Compare Late Fusion vs Embedding Fusion

    Args:
        symptoms_text: Comma-separated symptoms
        image_path: Path to medical image

    Returns:
        Comparison results
    """
    engine = get_fusion_engine()

    # Late fusion
    late_result = engine.predict_multimodal(
        symptoms_text, image_path, fusion_mode='late'
    )

    # Embedding fusion
    embedding_result = engine.predict_multimodal(
        symptoms_text, image_path, fusion_mode='embedding'
    )

    return {
        'late_fusion': late_result,
        'embedding_fusion': embedding_result,
        'comparison': {
            'late_confidence': late_result.get('combined_confidence', 0),
            'embedding_confidence': embedding_result.get('confidence', 0),
            'winner': 'embedding' if embedding_result.get('confidence', 0) >
                                    late_result.get('combined_confidence', 0)
                                    else 'late'
        }
    }


if __name__ == "__main__":
    # Test the fusion engine
    print("=" * 60)
    print("MULTIMODAL FUSION ENGINE TEST")
    print("=" * 60)

    engine = get_fusion_engine()

    print(f"Image model: Tanishq Skin Classifier (95.60% accuracy)")
    print(f"Text model: SVC loaded = {engine.svc_model is not None}")
    print(f"Supported: Single modality (text/image) or both")

    print("=" * 60)
