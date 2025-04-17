import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load trained model
model = load_model("D:\project\major-project\model\skin_disease_model.h5")

# Class names (must match the folder names used during training)
class_names = [
    'Acne', 'Actinic keratosis', 'Atopic Dermatitis', 'Basal Cell Carcinoma',
    'Eczema', 'Melanoma', 'Nevus', 'Psoriasis', 'Rosacea',
    'Seborrheic Keratosis', 'Tinea Ringworm Candidiasis', 'Urticaria Hives',
    'Vitiligo'
]

# Medicine and doctor suggestions (you can customize these)
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

# Function to predict
def predict_disease(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    disease = class_names[predicted_index]
    confidence = predictions[0][predicted_index] * 100

    medicine, doctor = recommendations[disease]

    print(f"\n✅ Predicted Disease: {disease}")
    print(f"📊 Confidence: {confidence:.2f}%")
    print(f"💊 Recommended Medicine: {medicine}")
    print(f"👨‍⚕️ Doctor Advice: {doctor}")

# Example usage
if __name__ == "__main__":
    img_path = input("Enter the path to the image: ")
    if os.path.exists(img_path):
        predict_disease(img_path)
    else:
        print("❌ Image path not found.")