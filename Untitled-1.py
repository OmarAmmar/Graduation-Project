# %%
# Importing the important packages
import tensorflow as tf
import keras
import matplotlib as plt
import numpy as np


# %%
from sklearn.model_selection import KFold

# %%
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# %%
from tensorflow.keras import layers
import matplotlib.pyplot as plt


# %%
# Include the data.
import os
from PIL import Image
import cv2

# %%


# A function to load all the images from a certain folder
def load_images_from_folder(folder,type):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            img = tf.keras.preprocessing.image.load_img(os.path.join(folder, filename),color_mode='grayscale',target_size=[360,180,1],interpolation='bilinear')
            # summarize some details about the image
            if img is not None:
                # I changed this part manually for loading the Cancer and Normal Images
                images.append((img,type))

    return images

# %%
# Load all the images from the files
Normal_Images = load_images_from_folder("/Users/omarammar/Downloads/Normal_generated",0)


# %%
np.shape(Normal_Images[0][0])

# %%
# Preparing Both datasets with labels and shuffling them
Precancer_Images = load_images_from_folder("/Users/omarammar/Downloads/Precancer_generated",1)

# %% [markdown]
# # Preparing the datasets

# %%
# We will add the two lists to each other generating one combined list of all the examples
print("The length of Normal Images is {} and the length of Precancer Images list is {}".format(len(Normal_Images),len(Precancer_Images)))

# %%
Images = Normal_Images+Precancer_Images
print("The length of the combined set of images is {}".format(len(Images)))


# %%
import random as rand
# Shuffling the images to mix the precancer and normal Images
rand.shuffle(Images)

# %%
Normal_Images[0][0]

# %% [markdown]
# # Start of Data Augmentation Using GANS

# %%
# Data Augmentation using GANs
# source:  https://towardsdatascience.com/generative-adversarial-network-gan-for-dummies-a-step-by-step-tutorial-fdefff170391
# source: tensorflow

# %%

BUFFER_SIZE = 60000
BATCH_SIZE = 8

"""_summary_
new_rows = ((rows - 1) * strides[0] + kernel_size[0] - 2 * padding[0] + output_padding[0])
new_cols = ((cols - 1) * strides[1] + kernel_size[1] - 2 * padding[1] + output_padding[1])
"""

IMAGE_SIZE = (360, 180)

def make_generator_model():
    model = tf.keras.Sequential(name = 'Generator')
    model.add(layers.Dense(90*45*128, use_bias=False, input_shape=(200,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((90, 45, 128)))
    assert model.output_shape == (None, 90, 45, 128)  # Note: None is the batch size
    
    model.add(layers.Conv2DTranspose(filters = 64,kernel_size = (5, 5), strides=(2, 2), padding='same', use_bias=False)) 
    assert model.output_shape == (None, 180, 90, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(filters = 1,kernel_size = (5, 5), strides=(2, 2), padding='same', use_bias=False)) 
    assert model.output_shape == (None, 360, 180, 1)  # Note: None is the batch size
        
    return model



# %%
generator = make_generator_model()

noise = tf.random.normal([1, 200])
generated_image = generator(noise,training=False)
print(generated_image.shape)
plt.imshow(generated_image[0, :, :, 0], cmap='gray')

# %%
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

# %%
def make_discriminator_model():
    model = tf.keras.Sequential(name='Discriminator')
    model.add(tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), activation='relu', input_shape=(360, 180, 1)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=120, activation='relu'))
    model.add(tf.keras.layers.Dense(units=84, activation='relu'))
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

    return model

# %%
discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print (decision)

# %%
# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# %%
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

# %%
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# %%
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# %%
import os
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# %%
EPOCHS = 250
noise_dim = 200
num_examples_to_generate = 16

# You will reuse this seed overtime (so it's easier)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# %%
# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

# %%
def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()

# %%
import time
def train(dataset, epochs):
  for epoch in range(epochs):
    print('Training Epoch #', epoch)
    start = time.time()

    i=1
    for image_batch in dataset:
      print('--->Training on image batch #', i)
      i+=1
      train_step(image_batch)

    # Produce images for the GIF as you go
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epochs,
                           seed)

# %%
X_train = []
y_train = []

for i in range(len(Normal_Images)):
    img = np.asarray(Normal_Images[i][0],dtype="uint8")
    X_train.append(img)
    y_train.append(Normal_Images[i][1])

X_train = np.asarray(X_train)
X_train = [x.reshape((1,360, 180, 1)) for x in X_train]


# %%
train(X_train, EPOCHS)


# %%
# Display a single image using the epoch number
def display_image(epoch_no):
  return Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

display_image(EPOCHS)


# %%
noise = tf.random.normal([1, 200])
generated_image = generator(noise,training=False)
print(generated_image.shape)
plt.imshow(generated_image[0, :, :, 0], cmap='gray')

