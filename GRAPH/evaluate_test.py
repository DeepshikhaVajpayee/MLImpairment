import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

test_data_dir = r"C:\Users\hp\Documents\INNOVATIVE PROJECT\SignDataset"

# Check if test data directory exists and has class subfolders with images
if not os.path.exists(test_data_dir):
    raise FileNotFoundError(f"Test data directory not found: {test_data_dir}")

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(96, 96),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

if test_generator.samples == 0:
    print("No images found in test directory.")
else:
    model = tf.keras.models.load_model("isl_mobilenetv2_model.h5")
    steps = test_generator.samples // test_generator.batch_size + 1

    predictions = model.predict(test_generator, steps=steps)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    print("Classification Report:\n")
    print(classification_report(true_classes, predicted_classes, target_names=class_labels))

    print("Confusion Matrix:\n")
    print(confusion_matrix(true_classes, predicted_classes))
