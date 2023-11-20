import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os

# Duplicating constants...
DATA_DIR = 'data/PetImages'
IMAGE_SIZE = (180, 180)
MODEL = 'save_at_17.keras'

model = keras.models.load_model(MODEL)

img = keras.utils.load_img(
    os.path.join(DATA_DIR, "Cat/6779.jpg"), target_size=IMAGE_SIZE
)
plt.imshow(img)

img_array = keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)
score = float(predictions[0])
print(f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")
