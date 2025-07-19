import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Set dataset path (customize this!)
DATASET_PATH = r"C:\Users\RISHITA\Downloads\archive (3)\food_dataset"

# Load images and labels
def load_images(folder, size=(100, 100)):
    images = []
    labels = []
    for label in os.listdir(folder):
        label_folder = os.path.join(folder, label)
        for file in os.listdir(label_folder):
            img_path = os.path.join(label_folder, file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, size)
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)

# Load image data
print("[🔄] Loading images...")
images, labels = load_images(DATASET_PATH)

# Preprocess
images = images / 255.0
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)

# CNN model
print("[🔧] Building model...")
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(100,100,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("[🚀] Training...")
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save model
model.save("food_model.h5")

# Load calorie info
calorie_df = pd.read_csv("calorie_data.csv")
calorie_dict = dict(zip(calorie_df["food"], calorie_df["calories"]))

# Predict one sample image
sample_img = X_test[0]
sample_label = y_test[0]
sample_input = np.expand_dims(sample_img, axis=0)

prediction = model.predict(sample_input)
predicted_class = le.inverse_transform([np.argmax(prediction)])[0]
true_class = le.inverse_transform([sample_label])[0]
calories = calorie_dict.get(predicted_class, "Unknown")

# Output
print(f"\n🎯 Predicted: {predicted_class} | 🔥 Estimated Calories: {calories}")
print(f"✅ Actual Label: {true_class}")

# Display image
plt.imshow(sample_img)
plt.title(f"Predicted: {predicted_class}\nCalories: {calories}")
plt.axis("off")
plt.show()
