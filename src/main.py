# Adapted from an example at keras.io [1], with some minor changes:
# - factor into separate functions
# - add some docstrings
# - tweak some settings to make it run at all
#
# Please note that this script is optimized for running on a CPU rather
# than a GPU. See the section _Two options to preprocess the data_ in [1]
# and adapt appropriately if you have a beefy GPU at hand.
#
# [1]: https://keras.io/examples/vision/image_classification_from_scratch/

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os

# Constants
DATA_DIR = 'data/PetImages'
IMAGE_SIZE = (180, 180)
# `BATCH_SIZE` was 128 in the keras.io tutorial, but it made my computer
# complain and sometimes freeze. Reducing to a lower number resolved this.
BATCH_SIZE = 32

def filter_corrupt_images():
    '''Walk through images in `DATA_DIR/{Cat, Dog}` and remove corrupt images'''

    print('Filtering out corrupt images...')
    num_skipped = 0
    for folder_name in ('Cat', 'Dog'):
        folder_path = os.path.join(DATA_DIR, folder_name)
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            with open(fpath, 'rb') as fobj:
                is_jfif = tf.compat.as_bytes('JFIF') in fobj.peek(10)
                if not is_jfif:
                    num_skipped += 1
                    # Delete corrupted image
                    os.remove(fpath)
    print(f'- Deleted {num_skipped} images')

def generate_dataset():
    '''Generate dataset, using global constants DATA_DIR, IMAGE_SIZE and BATCH_SIZE
    as inputs. Use 1/5 of the data set for validation.
    '''

    # https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory
    return keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split = 0.2,
        subset = 'both',
        seed = 1337,
        image_size = IMAGE_SIZE,
        batch_size = BATCH_SIZE,
    )

def visualize_data(train_ds):
    '''Make a plot of some objects in the data set and show in new window'''

    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype('uint8'))
            plt.title(str(int(labels[i])))
            plt.axis('off')
        plt.show()

def make_model(input_shape, num_classes):
    '''Create the classification model'''
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 728]:
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(size, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(size, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding='same')(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = 'sigmoid'
        units = 1
    else:
        activation = 'softmax'
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)

if __name__ == '__main__':
    # filter_corrupt_images()
    train_ds, val_ds = generate_dataset()
    # visualize_data(train_ds)

    # https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip('horizontal'),
            layers.RandomRotation(0.1),
        ]
    )

    # Apply `data_augmentation` to the training images.
    train_ds = train_ds.map(
        lambda img, label: (data_augmentation(img), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    # Prefetching samples in GPU memory helps maximize GPU utilization.
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    # Generate model
    model = make_model(input_shape=IMAGE_SIZE + (3,), num_classes=2)

    # Plot model, generating `model.png` in the directory the script is run
    keras.utils.plot_model(model, show_shapes=True)

    epochs = 25

    callbacks = [
        keras.callbacks.ModelCheckpoint('save_at_{epoch}.keras'),
    ]

    # https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    # https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
    model.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_ds,
    )

