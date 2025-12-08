"""
Tanishq Skin Condition Classifier
High-accuracy skin disease detection using EfficientNetV2B0
Accuracy: 95.60% | 6 common skin conditions
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple
import logging

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

class TanishqSkinClassifier:
    """
    Tanishq Skin Condition Classifier
    - 6 dermatological conditions
    - 95.60% test accuracy
    - 224x224 input resolution
    - EfficientNetV2B0 backbone
    """

    # Disease classes (6 skin conditions)
    DISEASE_CLASSES = [
        "Acne",
        "Carcinoma",
        "Eczema",
        "Keratosis",
        "Milia",
        "Rosacea"
    ]

    # Severity levels for each condition
    DISEASE_INFO = {
        "Acne": {"severity": "mild", "type": "inflammatory", "urgency": "routine"},
        "Carcinoma": {"severity": "high", "type": "malignant", "urgency": "immediate"},
        "Eczema": {"severity": "moderate", "type": "chronic", "urgency": "monitor"},
        "Keratosis": {"severity": "moderate", "type": "benign/precancerous", "urgency": "monitor"},
        "Milia": {"severity": "mild", "type": "benign", "urgency": "cosmetic"},
        "Rosacea": {"severity": "moderate", "type": "chronic", "urgency": "monitor"}
    }

    def __init__(self, model_name: str = "Tanishq77/skin-condition-classifier"):
        """
        Initialize Tanishq skin classifier

        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        self.model = None
        self.is_loaded = False

        print(f"🔬 Initializing Tanishq Skin Classifier...")
        print(f"   Model: {model_name}")
        print(f"   Accuracy: 95.60%")
        print(f"   Classes: {len(self.DISEASE_CLASSES)}")
        print(f"   Architecture: EfficientNetV2B0")

    def load_model(self) -> bool:
        """
        Load model from HuggingFace

        Returns:
            True if successful, False otherwise
        """
        try:
            from huggingface_hub import hf_hub_download

            print(f"📥 Loading Tanishq model from HuggingFace...")

            # Download model file
            model_path = hf_hub_download(
                repo_id=self.model_name,
                filename="skin_model.keras"
            )

            # Load Keras model
            self.model = tf.keras.models.load_model(model_path)

            self.is_loaded = True
            print(f"✅ Tanishq model loaded successfully!")
            print(f"   Input size: 224x224")
            print(f"   Conditions: {', '.join(self.DISEASE_CLASSES)}")

            return True

        except Exception as e:
            print(f"❌ Error loading Tanishq model: {e}")
            self.is_loaded = False
            return False

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess image for Tanishq model

        Args:
            image_path: Path to image file

        Returns:
            Preprocessed image array
        """
        from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

        # Load image
        image = Image.open(image_path).convert('RGB')

        # Resize to 224x224 (EfficientNetV2 requirement)
        image = image.resize((224, 224), Image.LANCZOS)

        # Convert to array
        image_array = np.array(image, dtype=np.float32)

        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)

        # EfficientNet preprocessing
        image_array = preprocess_input(image_array)

        return image_array

    def predict(self, image_path: str) -> Dict:
        """
        Predict skin condition from image

        Args:
            image_path: Path to skin image

        Returns:
            Prediction results with condition and confidence
        """
        if not self.is_loaded:
            if not self.load_model():
                return {
                    'success': False,
                    'error': 'Model not loaded',
                    'model': 'tanishq_skin',
                    'fallback': True
                }

        try:
            # Preprocess image
            image_array = self.preprocess_image(image_path)

            # Get predictions
            predictions = self.model.predict(image_array, verbose=0)[0]

            # DEBUG: Print all predictions
            print(f"\n🔍 DEBUG - Tanishq Predictions:")
            for i, (disease, prob) in enumerate(zip(self.DISEASE_CLASSES, predictions)):
                print(f"   {i}. {disease}: {prob:.4f} ({prob*100:.2f}%)")

            # Get top prediction
            predicted_idx = np.argmax(predictions)
            predicted_disease = self.DISEASE_CLASSES[predicted_idx]
            confidence = float(predictions[predicted_idx])

            print(f"\n✅ FINAL: {predicted_disease} (confidence: {confidence:.4f})")

            # Get disease info
            disease_info = self.DISEASE_INFO[predicted_disease]

            # Get top 3 predictions
            top3_indices = np.argsort(predictions)[-3:][::-1]
            top3_predictions = [
                {
                    'disease': self.DISEASE_CLASSES[idx],
                    'confidence': float(predictions[idx]),
                    'info': self.DISEASE_INFO[self.DISEASE_CLASSES[idx]]
                }
                for idx in top3_indices
            ]

            # All predictions
            all_predictions = {
                self.DISEASE_CLASSES[i]: float(predictions[i])
                for i in range(len(self.DISEASE_CLASSES))
            }

            # Check if high severity
            is_severe = disease_info['severity'] == 'high'

            return {
                'success': True,
                'predicted_disease': predicted_disease,
                'confidence': confidence,
                'severity': disease_info['severity'],
                'condition_type': disease_info['type'],
                'urgency': disease_info['urgency'],
                'is_severe': is_severe,
                'top3_predictions': top3_predictions,
                'all_predictions': all_predictions,
                'model': 'tanishq_skin',
                'model_accuracy': 0.956,
                'input_size': '224x224',
                'architecture': 'EfficientNetV2B0'
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'model': 'tanishq_skin'
            }


# Global instance
_tanishq_classifier = None

def get_tanishq_classifier() -> TanishqSkinClassifier:
    """Get singleton instance of Tanishq classifier"""
    global _tanishq_classifier
    if _tanishq_classifier is None:
        _tanishq_classifier = TanishqSkinClassifier()
        _tanishq_classifier.load_model()
    return _tanishq_classifier


if __name__ == "__main__":
    # Test the classifier
    print("=" * 60)
    print("TANISHQ SKIN CONDITION CLASSIFIER - TEST")
    print("=" * 60)

    classifier = TanishqSkinClassifier()

    # Load model
    if classifier.load_model():
        print("\n✅ Model loaded successfully!")
        print(f"   Classes: {classifier.DISEASE_CLASSES}")
        print(f"   Accuracy: 95.60%")
        print(f"   Architecture: EfficientNetV2B0")
    else:
        print("\n❌ Failed to load model")

    print("\n" + "=" * 60)
