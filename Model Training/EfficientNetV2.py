import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import pickle

data_dir = r"C:\Users\hp\Documents\INNOVATIVE PROJECT\SignDataset\Indian"

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8, 1.2],
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

num_classes = train_generator.num_classes

base_model = tf.keras.applications.EfficientNetV2B0(
    input_shape=(128, 128, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True) # high patience

history = model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator,
    callbacks=[lr_reduction, early_stopping]
)

base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

fine_tune_history = model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator,
    callbacks=[lr_reduction, early_stopping]
)

loss, accuracy = model.evaluate(val_generator)
print(f"Final Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

model.save('isl_efficientnetv2_finetuned.h5')

with open('efficientnetv2_initial_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)

with open('efficientnetv2_finetune_history.pkl', 'wb') as f:
    pickle.dump(fine_tune_history.history, f)
