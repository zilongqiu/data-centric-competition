
import sys

from keras.preprocessing import image
import numpy as np
from keras.models import load_model

model = load_model('./model.h5', compile=True)

# dimensions of our images
img_width, img_height = 32, 32

class_names = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X']

# predicting images
imagePath= sys.argv[1]
img = image.load_img(imagePath, target_size=(img_width, img_height))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = np.argmax(model.predict(images), axis=1)
print(class_names[classes[0]])
