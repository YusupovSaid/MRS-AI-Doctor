"""
Extended database models for multimodal Medical Recommendation System
Adds support for images, audio, and multimodal predictions
"""

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from sqlalchemy.dialects.postgresql import JSON

db = SQLAlchemy()

# ============================================================================
# EXISTING MODELS (Updated for PostgreSQL)
# ============================================================================

class User(UserMixin, db.Model):
    """User model with authentication and profile information"""
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    date_of_birth = db.Column(db.Date, nullable=True)
    gender = db.Column(db.String(10), nullable=True)
    phone = db.Column(db.String(20), nullable=True)
    address = db.Column(db.Text, nullable=True)
    medical_history = db.Column(db.Text, nullable=True)
    allergies = db.Column(db.Text, nullable=True)
    current_medications = db.Column(db.Text, nullable=True)
    emergency_contact = db.Column(db.String(100), nullable=True)
    emergency_phone = db.Column(db.String(20), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    prediction_history = db.relationship('PredictionHistory', backref='user', lazy=True, cascade='all, delete-orphan')
    media_files = db.relationship('MediaFile', backref='user', lazy=True, cascade='all, delete-orphan')
    multimodal_predictions = db.relationship('MultimodalPrediction', backref='user', lazy=True, cascade='all, delete-orphan')

    def set_password(self, password):
        """Hash and set password"""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """Check if provided password matches hash"""
        return check_password_hash(self.password_hash, password)

    def get_id(self):
        """Required by Flask-Login"""
        return str(self.id)

    def __repr__(self):
        return f'<User {self.username}>'


class PredictionHistory(db.Model):
    """Legacy prediction history for text-only symptom predictions"""
    __tablename__ = 'prediction_history'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    symptoms = db.Column(db.Text, nullable=False)
    predicted_disease = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=True)
    precautions = db.Column(db.Text, nullable=True)
    medications = db.Column(db.Text, nullable=True)
    diet = db.Column(db.Text, nullable=True)
    workout = db.Column(db.Text, nullable=True)
    confidence = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)

    def __repr__(self):
        return f'<PredictionHistory {self.predicted_disease} for User {self.user_id}>'


class UserSession(db.Model):
    """User session tracking for security and authentication"""
    __tablename__ = 'user_sessions'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    session_token = db.Column(db.String(255), unique=True, nullable=False, index=True)
    ip_address = db.Column(db.String(45), nullable=True)
    user_agent = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    last_activity = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    expires_at = db.Column(db.DateTime, nullable=False, index=True)

    def __repr__(self):
        return f'<UserSession for User {self.user_id}>'


# ============================================================================
# NEW MULTIMODAL MODELS
# ============================================================================

class MediaFile(db.Model):
    """Storage metadata for uploaded medical images and audio files"""
    __tablename__ = 'media_files'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True, index=True)

    # File metadata
    file_type = db.Column(db.String(20), nullable=False, index=True)  # 'image', 'audio', 'video'
    file_path = db.Column(db.String(500), nullable=False)  # S3 URL or local path
    storage_backend = db.Column(db.String(20), default='local')  # 'local', 's3', 'azure'
    original_filename = db.Column(db.String(255), nullable=False)
    file_size = db.Column(db.Integer, nullable=True)  # bytes
    mime_type = db.Column(db.String(100), nullable=True)

    # Image-specific metadata
    image_width = db.Column(db.Integer, nullable=True)
    image_height = db.Column(db.Integer, nullable=True)
    image_format = db.Column(db.String(10), nullable=True)  # 'jpg', 'png', 'dicom', 'nii'
    image_type = db.Column(db.String(50), nullable=True)  # 'skin', 'xray', 'retinal', 'mri', etc.

    # Audio-specific metadata
    audio_duration = db.Column(db.Float, nullable=True)  # seconds
    audio_sample_rate = db.Column(db.Integer, nullable=True)  # Hz
    audio_channels = db.Column(db.Integer, nullable=True)  # 1 for mono, 2 for stereo
    audio_type = db.Column(db.String(50), nullable=True)  # 'respiratory', 'cough', 'heart', etc.

    # Feature extraction results (stored as JSON)
    extracted_features = db.Column(JSON, nullable=True)

    # Processing status
    processing_status = db.Column(db.String(20), default='pending')  # 'pending', 'processing', 'completed', 'failed'
    processing_error = db.Column(db.Text, nullable=True)

    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)
    processed_at = db.Column(db.DateTime, nullable=True)

    # Relationships
    multimodal_predictions = db.relationship('MultimodalPrediction',
                                            secondary='prediction_media',
                                            backref=db.backref('media_files_rel', lazy='dynamic'))

    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'file_type': self.file_type,
            'file_path': self.file_path,
            'original_filename': self.original_filename,
            'file_size': self.file_size,
            'mime_type': self.mime_type,
            'image_width': self.image_width,
            'image_height': self.image_height,
            'image_format': self.image_format,
            'image_type': self.image_type,
            'audio_duration': self.audio_duration,
            'audio_sample_rate': self.audio_sample_rate,
            'audio_type': self.audio_type,
            'processing_status': self.processing_status,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

    def __repr__(self):
        return f'<MediaFile {self.file_type} {self.original_filename}>'


class MultimodalPrediction(db.Model):
    """Multimodal disease predictions combining text, image, and audio data"""
    __tablename__ = 'multimodal_predictions'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True, index=True)

    # Input data
    text_symptoms = db.Column(db.Text, nullable=True)  # Comma-separated symptoms

    # Model predictions (stored as JSON)
    # Each contains: {'predicted_class': int, 'confidence': float, 'probabilities': [...]}
    text_prediction = db.Column(JSON, nullable=True)
    image_prediction = db.Column(JSON, nullable=True)
    audio_prediction = db.Column(JSON, nullable=True)

    # Final combined prediction
    final_prediction = db.Column(db.String(200), nullable=False)
    fusion_method = db.Column(db.String(50), nullable=True)  # 'early', 'late', 'attention', 'adaptive'
    combined_confidence = db.Column(db.Float, nullable=True)

    # Recommendations (stored as JSON)
    recommendations = db.Column(JSON, nullable=True)
    # Structure: {
    #   'description': str,
    #   'precautions': [str],
    #   'medications': [str],
    #   'diet': [str],
    #   'workout': [str]
    # }

    # Processing metadata
    modalities_used = db.Column(JSON, nullable=True)  # ['text', 'image', 'audio']
    processing_time = db.Column(db.Float, nullable=True)  # seconds

    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)

    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'text_symptoms': self.text_symptoms,
            'text_prediction': self.text_prediction,
            'image_prediction': self.image_prediction,
            'audio_prediction': self.audio_prediction,
            'final_prediction': self.final_prediction,
            'fusion_method': self.fusion_method,
            'combined_confidence': self.combined_confidence,
            'recommendations': self.recommendations,
            'modalities_used': self.modalities_used,
            'processing_time': self.processing_time,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'media_files': [mf.to_dict() for mf in self.media_files_rel]
        }

    def __repr__(self):
        return f'<MultimodalPrediction {self.final_prediction} (User {self.user_id})>'


# Association table for many-to-many relationship between predictions and media files
prediction_media = db.Table('prediction_media',
    db.Column('prediction_id', db.Integer, db.ForeignKey('multimodal_predictions.id'), primary_key=True),
    db.Column('media_file_id', db.Integer, db.ForeignKey('media_files.id'), primary_key=True),
    db.Column('created_at', db.DateTime, default=datetime.utcnow)
)


class ModelVersion(db.Model):
    """Track different versions of ML models for A/B testing and rollback"""
    __tablename__ = 'model_versions'

    id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(100), nullable=False, index=True)  # 'text_svc', 'image_resnet', 'audio_cnn', etc.
    version = db.Column(db.String(50), nullable=False)  # 'v1.0', 'v2.3', etc.
    model_path = db.Column(db.String(500), nullable=False)  # Path to model file
    model_type = db.Column(db.String(50), nullable=False)  # 'sklearn', 'pytorch', 'tensorflow'

    # Model metadata
    architecture = db.Column(db.String(100), nullable=True)  # 'SVC', 'ResNet50', 'CNN', etc.
    input_shape = db.Column(JSON, nullable=True)  # Model input dimensions
    output_classes = db.Column(db.Integer, nullable=True)  # Number of output classes

    # Training metadata
    training_dataset = db.Column(db.String(200), nullable=True)
    training_samples = db.Column(db.Integer, nullable=True)
    training_accuracy = db.Column(db.Float, nullable=True)
    validation_accuracy = db.Column(db.Float, nullable=True)

    # Performance metrics
    performance_metrics = db.Column(JSON, nullable=True)
    # Structure: {
    #   'precision': float,
    #   'recall': float,
    #   'f1_score': float,
    #   'confusion_matrix': [[...]],
    # }

    # Deployment status
    is_active = db.Column(db.Boolean, default=False, index=True)
    is_default = db.Column(db.Boolean, default=False)

    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    deployed_at = db.Column(db.DateTime, nullable=True)
    deprecated_at = db.Column(db.DateTime, nullable=True)

    def __repr__(self):
        return f'<ModelVersion {self.model_name} {self.version}>'


class PredictionFeedback(db.Model):
    """User feedback on predictions for continuous model improvement"""
    __tablename__ = 'prediction_feedback'

    id = db.Column(db.Integer, primary_key=True)
    prediction_id = db.Column(db.Integer, db.ForeignKey('multimodal_predictions.id'), nullable=False, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True, index=True)

    # Feedback data
    is_correct = db.Column(db.Boolean, nullable=True)  # Was the prediction correct?
    actual_disease = db.Column(db.String(200), nullable=True)  # If prediction was wrong
    feedback_text = db.Column(db.Text, nullable=True)  # Additional comments
    rating = db.Column(db.Integer, nullable=True)  # 1-5 stars

    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Relationships
    prediction = db.relationship('MultimodalPrediction', backref='feedback')

    def __repr__(self):
        return f'<PredictionFeedback for Prediction {self.prediction_id}>'


# ============================================================================
# DATABASE INITIALIZATION HELPERS
# ============================================================================

def init_db(app):
    """Initialize database with app context"""
    db.init_app(app)

def create_tables(app):
    """Create all database tables"""
    with app.app_context():
        db.create_all()
        print("✅ All database tables created successfully!")

def drop_tables(app):
    """Drop all database tables (use with caution!)"""
    with app.app_context():
        db.drop_all()
        print("⚠️  All database tables dropped!")
