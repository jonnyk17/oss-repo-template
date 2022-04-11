# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from PIL import ImageOps
import numpy as np
def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
size= 28, 28
im = Image.open("shirt1.jpg")
im2=Image.open("shirt2.jpg")
im3=Image.open("shirt3.jpg")
np_im = np.array(im)
np_im = np_im - 18
new_im = Image.fromarray(np_im)
new_im.thumbnail(size)
ImageOps.autocontrast(new_im)
new_im.save("new1.jpg")
np_im = np.array(im2)
np_im = np_im - 18
new_im2 = Image.fromarray(np_im)
new_im2.thumbnail(size)
ImageOps.autocontrast(new_im2)
new_im2.save("new2.jpg")
np_im = np.array(im3)
np_im = np_im - 18
new_im3 = Image.fromarray(np_im)
new_im3.thumbnail(size)
ImageOps.autocontrast(new_im3)
new_im3.save("new3.jpg")
new_list=['new1.jpg', 'new2.jpg', 'new3.jpg']
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_image, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10)
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_image)
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(10,10))
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, new_list)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

