from flask import Flask, request, jsonify
import cv2
import numpy as np  # Don't forget to import numpy
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/detect-face', methods=['POST'])
def detect_face():
    try:
        # Get the image file from the POST request
        file = request.files['image']
        print("Received file:", file.filename)  # Log the filename

        # Convert the image to a format OpenCV can process
        img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"message": "Error processing image."}), 400

        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            return jsonify({"message": "Human face detected!"})
        else:
            return jsonify({"message": "No face detected."})

    except Exception as e:
        print("Error:", e)
        return jsonify({"message": "An error occurred while processing the image."}), 500

if __name__ == '__main__':
    app.run(debug=True)
