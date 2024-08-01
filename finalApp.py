from flask import Flask, request, jsonify
from facenet_pytorch import MTCNN
import cv2
import numpy as np
from PIL import Image
import io

# Initialize Flask app and MTCNN model
app = Flask(__name__)
mtcnn = MTCNN(keep_all=True)

def is_blurry(image, threshold=100):
    """Check if the image is blurry using the Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold

def analyze_image(image):
    """Analyze the uploaded image for face detection and clarity."""
    image_np = np.array(image)
    
    # Detect faces and landmarks in the image
    boxes, _ = mtcnn.detect(image)

    # If no faces are detected, return appropriate message
    if boxes is None or len(boxes) == 0:
        return {"result": "No face detected","auth":"false"}

    
    # If multiple faces are detected, return appropriate message
    if len(boxes) > 1:
        return {"result": "Multiple faces detected","auth":"false"}
    
    # For a single detected face, check if it's a full face and clear
    x1, y1, x2, y2 = map(int, boxes[0])
    face_region = image_np[y1:y2, x1:x2]

    # Heuristic check for a "full" face (occupying a significant portion of the image)
    face_area = (x2 - x1) * (y2 - y1)
    image_area = image_np.shape[0] * image_np.shape[1]
    if face_area / image_area < 0.2:
        return {"result": "Face detected but it may not be a full face","auth":"false"}
    
    # Check if the detected face region is blurry
    if is_blurry(face_region):
        return {"result": "Image is blurry","auth":"false"}
    
    return {"result": "Face detected","auth":"true"}

@app.route('/check_image', methods=['POST'])
def check_image():
    """Endpoint to check the image for face detection and clarity."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    
    try:
        # Read the image using PIL
        image = Image.open(io.BytesIO(image_file.read()))
        result = analyze_image(image)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)