from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
db = SQLAlchemy(app)

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load ML Model
MODEL_PATH = "model/skin_disease_model.h5"
model = tf.keras.models.load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

# Class Names and Recommendations
class_names = [
    'Acne', 'Actinic keratosis', 'Atopic Dermatitis', 'Basal Cell Carcinoma',
    'Eczema', 'Melanoma', 'Nevus', 'Psoriasis', 'Rosacea',
    'Seborrheic Keratosis', 'Tinea Ringworm Candidiasis', 'Urticaria Hives',
    'Vitiligo'
]

recommendations = {
    'Acne': ('Benzoyl peroxide', 'Consult dermatologist if severe.'),
    'Actinic keratosis': ('Fluorouracil cream', 'See a skin specialist.'),
    'Atopic Dermatitis': ('Hydrocortisone cream', 'Visit an allergy specialist.'),
    'Basal Cell Carcinoma': ('Surgical removal', 'Consult an oncologist.'),
    'Eczema': ('Moisturizers, antihistamines', 'Consult dermatologist.'),
    'Melanoma': ('Biopsy and treatment', 'Urgent dermatologist visit.'),
    'Nevus': ('Observation', 'Consult dermatologist for changes.'),
    'Psoriasis': ('Steroid creams, phototherapy', 'See a rheumatologist.'),
    'Rosacea': ('Topical metronidazole', 'Consult dermatologist.'),
    'Seborrheic Keratosis': ('Cryotherapy', 'Consult skin specialist.'),
    'Tinea Ringworm Candidiasis': ('Antifungal cream', 'General physician consult.'),
    'Urticaria Hives': ('Antihistamines', 'Allergy testing advised.'),
    'Vitiligo': ('Topical corticosteroids', 'Dermatologist consultation.')
}

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    if User.query.filter_by(username=data['username']).first():
        return jsonify({'message': 'User already exists'}), 400

    new_user = User(username=data['username'])
    new_user.set_password(data['password'])
    db.session.add(new_user)
    db.session.commit()
    return jsonify({'message': 'User registered successfully'}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    user = User.query.filter_by(username=data['username']).first()
    if user and user.check_password(data['password']):
        session['user'] = user.username
        return jsonify({'message': 'Login successful'}), 200
    return jsonify({'message': 'Invalid credentials'}), 401

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home'))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def predict_skin_disease(image_path):
    if model is None:
        return "Model not found", "N/A", "N/A"

    try:
        img = Image.open(image_path).resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions)
        disease = class_names[predicted_index]
        confidence = float(predictions[0][predicted_index]) * 100

        medicine, doctor = recommendations[disease]
        return disease, medicine, doctor
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return "Prediction Error", "N/A", "N/A"

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    disease, medicine, doctor = predict_skin_disease(file_path)

    return jsonify({
        "disease": disease,
        "medicine": medicine,
        "doctor": doctor,
        "file_path": file_path
    })

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
