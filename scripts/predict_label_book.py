
import sys
import os
import pathlib

from keras.preprocessing import image
import numpy as np
from keras.models import load_model

modelPath = sys.argv[1]
model = load_model(modelPath, compile=True)

# dimensions of our images
img_width, img_height = 32, 32

class_names = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X']

total = 0
correct = 0
wrong = 0
lastFolder = ''

for subdir, dirs, files in os.walk('./../label_book/'):
    for file in files:
        filepath = subdir + os.sep + file

        if filepath.endswith(".png"):
            path = pathlib.PurePath(os.path.dirname(os.path.join(subdir, file)))
            folderName = path.name.upper()

            if lastFolder != folderName:
                print(folderName)
                lastFolder = folderName

            # predicting images
            print('    - ' + filepath)
            img = image.load_img(filepath, target_size=(img_width, img_height))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)

            images = np.vstack([x])
            classes = np.argmax(model.predict(images), axis=1)
            prediction = class_names[classes[0]]

            total += 1
            if prediction == folderName:
                print('      CORRECT')
                correct += 1
            else:
                print('      WRONG ' + prediction)
                wrong += 1

print('==============================')
print('TOTAL: ' + str(total))
print('CORRECT: ' + str(correct))
print('WRONG: ' + str(wrong))
print('RESULT: ' + str(correct / total * 100) + '%')
print('==============================')