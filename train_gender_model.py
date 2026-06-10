#imporrt libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import load_img, img_to_array

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout


# Load the dataset
train_dir = r"C:\Users\adhit\OneDrive\Desktop\age_detect_face\gender\Training"
test_dir = r"C:\Users\adhit\OneDrive\Desktop\age_detect_face\gender\Testing"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)



#building the model
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')
])


#train the model
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])



history = model.fit(
    train_data,
    validation_data = test_data,
    epochs = 5
)

model.save("gender_model.h5")


#prediction on test images

img_path = r"C:\Users\adhit\OneDrive\Desktop\age_detect_face\dileep.jpeg"
img = load_img(img_path, target_size=(224,224))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
predicted_index = np.argmax(prediction)

class_labels = train_data.class_indices

# Reverse mapping
index_to_label = {v: k for k, v in class_labels.items()}
predicted_label = index_to_label[predicted_index]

print("Predicted gender Group:", predicted_label)