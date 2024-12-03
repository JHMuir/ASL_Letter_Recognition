import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model # For some reason, I need to add this in order for it to work for Mac
import numpy as np

# Load the trained model
model = load_model('asl_model.h5')

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Set camera resolution for performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# List of ASL letters corresponding to model predictions
letterpred = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the image horizontally for a selfie-view display
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Draw hand landmarks and make predictions
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Calculate the bounding box around the hand
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]

            h, w, c = frame.shape
            x_min = int(min(x_coords) * w)
            x_max = int(max(x_coords) * w)
            y_min = int(min(y_coords) * h)
            y_max = int(max(y_coords) * h)

            # Add dynamic padding to the bounding box
            padding = int(0.1 * (x_max - x_min))  # 10% of the bounding box width
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)

            # Crop and preprocess the hand region
            try:
                hand_region = frame[y_min:y_max, x_min:x_max]
                hand_region_gray = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
                hand_region_resized = cv2.resize(hand_region_gray, (28, 28))  # Resize to model input size
                hand_region_normalized = hand_region_resized / 255.0  # Normalize pixel values
                hand_region_input = hand_region_normalized.reshape(1, 28, 28, 1)  # Reshape for the model

                # Predict the gesture
                prediction = model.predict(hand_region_input)
                confidence = np.max(prediction)
                if confidence > 0.7:  # Use a confidence threshold
                    predicted_letter = letterpred[np.argmax(prediction)]
                else:
                    predicted_letter = "Unknown"

                # Display the predicted letter on the frame
                cv2.putText(
                    frame, 
                    f"Prediction: {predicted_letter}", 
                    (x_min, y_min - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 255, 0), 
                    2
                )
            except Exception as e:
                print(f"Error processing hand region: {e}")

            # Draw the bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame
    cv2.imshow('Hand Gesture Recognition', frame)

    # Press 'q' to quit
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
