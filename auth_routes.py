from flask import Blueprint, render_template, request, redirect, url_for, session, flash
from flask_login import login_user, logout_user, current_user, login_required
from urllib.parse import urlparse
from models_multimodal import db, User
from datetime import datetime

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if current_user.is_authenticated:
        return redirect(url_for('multimodal.upload_page'))

    if request.method == 'POST':
        # Get form data - SIMPLIFIED (only essential fields)
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        full_name = request.form.get('full_name', '').strip()

        # Validation
        errors = []

        if not email:
            errors.append('Email is required')
        if not password:
            errors.append('Password is required')
        if password != confirm_password:
            errors.append('Passwords do not match')
        if len(password) < 6:
            errors.append('Password must be at least 6 characters')
        if not full_name:
            errors.append('Full name is required')

        # Check if email already exists
        if User.query.filter_by(email=email).first():
            errors.append('Email already exists')

        if errors:
            for error in errors:
                flash(error, 'danger')
            return render_template('register.html')

        # Split full name into first and last
        name_parts = full_name.split(' ', 1)
        first_name = name_parts[0]
        last_name = name_parts[1] if len(name_parts) > 1 else ''

        # Generate unique username from email
        base_username = email.split('@')[0]
        username = base_username

        # If username exists, add number suffix
        counter = 1
        while User.query.filter_by(username=username).first():
            username = f"{base_username}{counter}"
            counter += 1

        # Create new user with simplified fields
        user = User(
            username=username,
            email=email,
            first_name=first_name,
            last_name=last_name
        )

        # Set password
        user.set_password(password)

        # Save to database
        db.session.add(user)
        db.session.commit()

        # Auto-login the user after registration
        login_user(user)
        flash('Registration successful! Welcome to MRS Health.', 'success')
        return redirect(url_for('multimodal.upload_page'))

    return render_template('register.html')

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if current_user.is_authenticated:
        return redirect(url_for('multimodal.upload_page'))

    if request.method == 'POST':
        username_or_email = request.form.get('username_or_email', '').strip()
        password = request.form.get('password', '')
        remember = request.form.get('remember') == 'on'

        # Find user by username or email
        user = User.query.filter(
            (User.username == username_or_email) | (User.email == username_or_email)
        ).first()

        if user and user.check_password(password):
            # Login user
            login_user(user, remember=remember)

            # Redirect to next page or multimodal consultation page
            next_page = request.args.get('next')
            if not next_page or urlparse(next_page).netloc != '':
                next_page = url_for('multimodal.upload_page')
            return redirect(next_page)
        else:
            flash('Invalid username/email or password', 'danger')

    return render_template('login.html')

@auth_bp.route('/logout')
@login_required
def logout():
    """User logout"""
    logout_user()
    flash('You have been logged out successfully', 'info')
    return redirect(url_for('index'))

@auth_bp.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    """User profile management"""
    if request.method == 'POST':
        # Update user information
        first_name = request.form.get('first_name', '').strip()
        last_name = request.form.get('last_name', '').strip()
        email = request.form.get('email', '').strip().lower()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()
        date_of_birth = request.form.get('date_of_birth', '')
        gender = request.form.get('gender', '')
        medical_history = request.form.get('medical_history', '').strip()
        allergies = request.form.get('allergies', '').strip()
        current_medications = request.form.get('current_medications', '').strip()
        emergency_contact = request.form.get('emergency_contact', '').strip()
        emergency_phone = request.form.get('emergency_phone', '').strip()

        # Check if email is already taken by another user
        if email != current_user.email:
            if User.query.filter_by(email=email).first():
                flash('Email already exists', 'danger')
                return render_template('profile.html')

        # Update user
        current_user.first_name = first_name
        current_user.last_name = last_name
        current_user.email = email
        current_user.phone = phone
        current_user.address = address
        current_user.medical_history = medical_history
        current_user.allergies = allergies
        current_user.current_medications = current_medications
        current_user.emergency_contact = emergency_contact
        current_user.emergency_phone = emergency_phone

        # Parse date of birth
        if date_of_birth:
            try:
                current_user.date_of_birth = datetime.strptime(date_of_birth, '%Y-%m-%d').date()
            except ValueError:
                pass

        # Update gender
        if gender in ['Male', 'Female', 'Other', '']:
            current_user.gender = gender

        db.session.commit()
        flash('Profile updated successfully', 'success')
        return redirect(url_for('auth.profile'))

    return render_template('profile.html')

@auth_bp.route('/change-password', methods=['GET', 'POST'])
@login_required
def change_password():
    """Change user password"""
    if request.method == 'POST':
        current_password = request.form.get('current_password', '')
        new_password = request.form.get('new_password', '')
        confirm_password = request.form.get('confirm_password', '')

        if not current_user.check_password(current_password):
            flash('Current password is incorrect', 'danger')
            return render_template('change_password.html')

        if new_password != confirm_password:
            flash('New passwords do not match', 'danger')
            return render_template('change_password.html')

        if len(new_password) < 6:
            flash('New password must be at least 6 characters', 'danger')
            return render_template('change_password.html')

        current_user.set_password(new_password)
        db.session.commit()
        flash('Password changed successfully', 'success')
        return redirect(url_for('auth.profile'))

    return render_template('change_password.html')
