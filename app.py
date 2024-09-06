import sys
import os
import io
import numpy as np
from PIL import Image
from flask import request, jsonify, Flask, render_template
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from sklearn.utils import class_weight
import tensorflow as tf

# Handle terminal encoding issues on Windows
if os.name == 'nt':
    sys.stdout.reconfigure(encoding='utf-8')

# Create Flask app
app = Flask(__name__)

# Path to our dataset
dataset_dir = 'static/dataset'
# Categories for classification
class_labels = ['rock', 'paper', 'scissors']


def create_model():
    # Start with the VGG16 model, exclude its top layers
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

    # We won't train the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add a custom classifier on top
    model = Sequential([
        base_model,
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')  # We have 3 output categories
    ])

    # Compile the model with Adam optimizer and categorical crossentropy
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def process_image(image_bytes):
    # Open the image from bytes and convert it to RGB format
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    # Resize image to 150x150 (input size for the model)
    img = img.resize((150, 150))
    # Convert image to numpy array and scale pixel values to [0, 1]
    img_array = np.array(img) / 255.0
    # Add a batch dimension (required for model input)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


@app.route("/predict", methods=["POST"])
def predict():
    # Load our pre-trained model
    model = create_model()
    model.load_weights('rock_paper_scissors_classifier.h5')

    try:
        # Check if the user uploaded a file
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        # Read the file as bytes
        file_bytes = request.files['file'].read()
        # Preprocess the image for prediction
        image = process_image(file_bytes)
        # Get the model's prediction
        prediction = model.predict(image)
        # Get the predicted class
        predicted_class = np.argmax(prediction)

        # Map the class index to the actual label
        class_map = {0: 'paper', 1: 'rock', 2: 'scissors'}
        prediction_label = class_map[predicted_class]

        return jsonify({"prediction": prediction_label}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def get_data_generators():
    # Data augmentation and preprocessing for training images
    train_data_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    # Preprocessing for validation images (no augmentation)
    valid_data_gen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

    # Generate training data
    train_generator = train_data_gen.flow_from_directory(
        directory=dataset_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    # Generate validation data
    valid_generator = valid_data_gen.flow_from_directory(
        directory=dataset_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    return train_generator, valid_generator


def calculate_class_weights(train_generator):
    # Get the classes and calculate class weights to handle imbalance
    labels = train_generator.classes
    class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    return dict(enumerate(class_weights))


@app.route("/train")
def train():
    try:
        # Get data generators
        train_gen, valid_gen = get_data_generators()
        # Calculate class weights for training
        class_weights = calculate_class_weights(train_gen)

        # Build the model
        model = create_model()
        model.summary()

        # Train the model
        model.fit(
            train_gen,
            steps_per_epoch=train_gen.samples // train_gen.batch_size,
            validation_data=valid_gen,
            validation_steps=valid_gen.samples // valid_gen.batch_size,
            epochs=1,
            class_weight=class_weights,
            verbose=1
        )

        # Save the trained model
        model.save('rock_paper_scissors_classifier.h5')

        # Evaluate the model on validation data
        loss, accuracy = model.evaluate(valid_gen, steps=valid_gen.samples // valid_gen.batch_size, verbose=0)

        return f"Training complete! Validation accuracy: {accuracy * 100:.2f}%. Model saved as 'rock_paper_scissors_classifier.h5'."

    except Exception as e:
        return f"Error during training: {str(e)}"


@app.route("/")
def home():
    return render_template('index.html')


# Start the Flask app
if __name__ == '__main__':
    # Enable dynamic memory growth for GPUs
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
        except:
            pass

    app.run()
