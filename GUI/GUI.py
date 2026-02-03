import cv2
import numpy as np
import tensorflow as tf
import pyttsx3

# Initialize text-to-speech engine
engine = pyttsx3.init()
# your speech code
engine.say("Hello")
engine.runAndWait()
engine.stop()  # Explicitly stop engine to help cleanup

import atexit

def cleanup():
    try:
        engine.stop()
    except Exception:
        pass

atexit.register(cleanup)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Load your trained model locally
model = tf.keras.models.load_model('isl_mobilenetv2_finetuned.h5')

IMG_SIZE = (224, 224)  # Model input size

def preprocess_frame(frame):
    img = cv2.resize(frame, IMG_SIZE)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)  # Batch dimension
    return img

# Mapping from class ids to labels (update as per your model)
label_map = {
    0: "Hello",
    1: "Thank You",
    2: "Yes",
    3: "No",
    4: "Peace",
}

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

predicted_gesture = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Flip frame for mirror view
    frame = cv2.flip(frame, 1)

    # Preprocess frame for prediction
    input_tensor = preprocess_frame(frame)

    # Run prediction
    preds = model.predict(input_tensor)
    class_id = int(np.argmax(preds[0]))
    confidence = float(np.max(preds[0]))

    # Use threshold to filter predictions
    if confidence > 0.2:
        gesture = label_map.get(class_id, f"Unknown {class_id}")
        display_text = f"{gesture}: {confidence:.2f}"

        if predicted_gesture != gesture:
            predicted_gesture = gesture
            speak(display_text)
    else:
        display_text = "No confident prediction"

    # Show prediction on frame
    cv2.putText(frame, display_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    cv2.imshow("ISL Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
