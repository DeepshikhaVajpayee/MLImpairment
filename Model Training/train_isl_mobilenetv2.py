
#import tensorflow as tf
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.callbacks import ReduceLROnPlateau
#import pickle

#datagen = ImageDataGenerator(
#    rescale=1./255,
#    rotation_range=20,
#    shear_range=0.2,
#    zoom_range=0.2,
#    horizontal_flip=True,
#    validation_split=0.2
#)

#train_generator = datagen.flow_from_directory(
#    r"C:\Users\hp\Documents\INNOVATIVE PROJECT\SignDataset\Indian",
#    target_size=(96, 96),
#    batch_size=32,
#    class_mode='categorical',
#    subset='training'
#)

#validation_generator = datagen.flow_from_directory(
#    r"C:\Users\hp\Documents\INNOVATIVE PROJECT\SignDataset\Indian",
#    target_size=(96, 96),
#    batch_size=32,
#    class_mode='categorical',
#    subset='validation',
#    shuffle=False
#)

#num_classes = train_generator.num_classes

#base_model = tf.keras.applications.MobileNetV2(
#    input_shape=(96, 96, 3),
#    include_top=False,
#    weights='imagenet'
#)
#base_model.trainable = False

#model = Sequential([
#    base_model,
#    GlobalAveragePooling2D(),
#   Dense(128, activation='relu'),
#    Dropout(0.4),
#    Dense(num_classes, activation='softmax')
#])

#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#model.summary()

# Disable EarlyStopping to allow full epoch run
#lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1)

#history = model.fit(
#    train_generator,
#    epochs=15,
#    validation_data=validation_generator,
#    callbacks=[lr_reduction]
#)

# Unfreeze last 20 layers for fine-tuning
#base_model.trainable = True
#for layer in base_model.layers[:-20]:
#    layer.trainable = False

#model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

#fine_tune_history = model.fit(
#    train_generator,
#    epochs=15,
#    validation_data=validation_generator,
#    callbacks=[lr_reduction]
#)

#loss, accuracy = model.evaluate(validation_generator)
#print(f"Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

#model.save('isl_mobilenetv2_finetuned.h5')#

#with open('mobilenetv2_history.pkl', 'wb') as f:
#    pickle.dump(history.history, f)

#with open('mobilenetv2_finetune_history.pkl', 'wb') as f:
#    pickle.dump(fine_tune_history.history, f)

#SERVER
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify


app = Flask(__name__)  # Corrected __name__


# -----------------------------
# Load Model Once at Startup
# -----------------------------
model = tf.keras.models.load_model('isl_mobilenetv2_finetuned.h5')
model.make_predict_function()   # Ensures thread-safe prediction in Flask


# Image size for model (change if different)
IMG_SIZE = (224, 224)


# -----------------------------
# Prediction Route
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Request received")
        if 'image' not in request.files:
            print("No image in request")
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        img_bytes = file.read()
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print("Failed to decode image!")
            with open("bad_image_dump.jpg", "wb") as f:
                f.write(img_bytes)
            return jsonify({'error': 'Invalid image file'}), 400

        print("Image shape:", img.shape)
        img = cv2.resize(img, (224, 224))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        print("Predicting...")
        preds = model.predict(img)
        print("Predictions:", preds)
        class_id = int(np.argmax(preds[0]))
        confidence = float(np.max(preds[0]))

        print("Prediction complete:", class_id, confidence)
        return jsonify({
            "class": class_id,
            "confidence": confidence
        })

    except Exception as e:
        print("Exception occurred:", e)
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Server error", "details": str(e)}), 500


@app.route("/")
def home():
    return "ISL Prediction Server Running"


if __name__ == '__main__':  # Corrected __name__ and __main__
    app.run(host='0.0.0.0', port=5000, threaded=True)
