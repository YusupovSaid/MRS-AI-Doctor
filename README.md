# MRS Health - Medicine Recommendation System

An AI-powered healthcare web application that predicts diseases based on symptoms (text) and skin conditions (images). Built with Flask and Machine Learning models as a Minimum Viable Product (MVP) for educational purposes.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange)
![License](https://img.shields.io/badge/License-Educational-yellow)

## Features

- **User Authentication**: Sign up, login, logout, and profile management
- **Text-Based Disease Prediction**: Analyze 132 symptoms to predict 41 different diseases
- **Image-Based Skin Analysis**: Upload skin images to detect 19 skin conditions
- **Multimodal Fusion**: Combine text and image inputs for comprehensive diagnosis
- **Health Recommendations**: Get personalized diet, medication, precautions, and workout suggestions
- **Consultation History**: Track past diagnoses in user dashboard
- **Responsive Design**: Modern UI with animated backgrounds

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend                                 │
│  (HTML/CSS/Bootstrap/JavaScript)                                │
├─────────────────────────────────────────────────────────────────┤
│                      Flask Web Server                           │
│                         (app.py)                                │
├──────────────┬──────────────────┬───────────────────────────────┤
│   Auth       │   Multimodal     │    Core Routes                │
│  Routes      │    Routes        │  (Pages & API)                │
├──────────────┴──────────────────┴───────────────────────────────┤
│                    Machine Learning Models                       │
├─────────────────┬─────────────────┬─────────────────────────────┤
│  SVC Classifier │  Skin Disease   │   Hugging Face             │
│  (Text-based)   │   Detector      │   Image Classifier         │
│   41 diseases   │  (TensorFlow)   │   (Deep Learning)          │
├─────────────────┴─────────────────┴─────────────────────────────┤
│                      Datasets (CSV)                             │
│  Training.csv │ skin_diseases_recommendations.csv │ diets.csv  │
└─────────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Backend** | Flask 3.0, Python 3.11 |
| **Database** | SQLite with Flask-SQLAlchemy |
| **Authentication** | Flask-Login |
| **ML (Text)** | Scikit-learn (SVC Classifier) |
| **ML (Image)** | TensorFlow 2.16, Hugging Face Transformers |
| **Image Processing** | OpenCV, Pillow |
| **Frontend** | HTML5, CSS3, Bootstrap 5, JavaScript |

## Project Structure

```
MRS/
├── app.py                    # Main Flask application
├── auth_routes.py            # Authentication routes (login, register, etc.)
├── multimodal_routes.py      # Image & text analysis endpoints
├── multimodal_fusion_derm.py # Multimodal fusion logic
├── models_multimodal.py      # ML model definitions
├── tanishq_skin_classifier.py # Skin disease classifier
├── requirements.txt          # Python dependencies
├── start_with_arm.sh         # Startup script for Apple Silicon
│
├── models/
│   ├── svc.pkl                      # SVC classifier for symptom-based prediction
│   ├── huggingface_image_classifier.pkl  # Pre-trained image classifier
│   ├── feature_names.pkl            # Feature names for ML model
│   └── label_encoder.pkl            # Label encoder for diseases
│
├── Dataset/
│   ├── Training.csv                 # Symptom-disease training data
│   ├── skin_diseases_recommendations.csv  # Skin disease info
│   ├── description.csv              # Disease descriptions
│   ├── precautions_df.csv           # Disease precautions
│   ├── medications.csv              # Recommended medications
│   ├── diets.csv                    # Diet recommendations
│   ├── workout_df.csv               # Exercise recommendations
│   └── Symptom-severity.csv         # Symptom severity scores
│
├── templates/                # HTML templates
│   ├── landing.html          # Home page
│   ├── multimodal_upload.html # Consultation page
│   ├── symptoms.html         # Symptoms & diseases list
│   ├── dashboard.html        # User dashboard
│   ├── about.html            # About page
│   ├── contact.html          # Contact page
│   ├── developer.html        # Developer info
│   ├── blog.html             # Blog page
│   ├── login.html            # Login page
│   ├── register.html         # Registration page
│   └── profile.html          # User profile
│
├── static/                   # Static assets (CSS, JS, images)
├── uploads/                  # User uploaded images
└── instance/                 # SQLite database
```

## Supported Conditions

### Text-Based Disease Prediction (41 diseases)
Fungal Infection, Allergy, GERD, Diabetes, Hypertension, Migraine, Jaundice, Malaria, Dengue, Typhoid, Hepatitis (A-E), Tuberculosis, Pneumonia, Heart Attack, and more...

### Image-Based Skin Analysis (19 conditions)
Acne, Eczema, Melanoma, Carcinoma, Rosacea, Keratosis, Milia, Dermatofibroma, Nevus, Psoriasis, and more...

## Setup Instructions

### Prerequisites
- Python 3.11+
- pip or conda

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YusupovSaid/MRS-Health.git
   cd MRS-Health
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the application**
   ```
   http://127.0.0.1:8000
   ```

### For Apple Silicon (M1/M2/M3)
```bash
chmod +x start_with_arm.sh
./start_with_arm.sh
```

## Usage

1. **Register/Login**: Create an account to access consultation features
2. **Consultation**:
   - Upload a skin image for analysis, OR
   - Type your symptoms for text-based diagnosis
3. **View Results**: Get disease prediction with:
   - Description
   - Precautions
   - Medications
   - Diet recommendations
   - Exercise tips
4. **Dashboard**: View your consultation history

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Landing page |
| `/about` | GET | About page |
| `/symptoms` | GET | List of symptoms & diseases |
| `/contact` | GET | Contact page |
| `/multimodal/upload_page` | GET | Consultation page |
| `/multimodal/predict` | POST | Submit symptoms/image for prediction |
| `/dashboard` | GET | User consultation history |
| `/auth/login` | GET/POST | User login |
| `/auth/register` | GET/POST | User registration |
| `/auth/logout` | GET | User logout |

## Important Disclaimer

**This project is for educational purposes only.** MRS Health is a Minimum Viable Product (MVP) developed as part of machine learning studies at Woosong University.

- This application is **NOT** intended for real-world medical diagnosis
- Predictions should **NOT** replace professional medical advice
- Always consult a qualified healthcare provider for health concerns
- ML models are trained on limited datasets

## Author

**Abbosjonov SaidAkbar**
AI & Big Data Student at Woosong University, South Korea

- GitHub: [YusupovSaid](https://github.com/YusupovSaid)
- Instagram: [@abbosjonov_said](https://www.instagram.com/abbosjonov_said/)
- Telegram: [@abbosjonov_said](https://t.me/abbosjonov_said)

## License

This project is for educational purposes only. Not licensed for commercial use.

## Acknowledgments

- Woosong University for academic support
- Hugging Face for pre-trained models
- Open-source medical datasets contributors
