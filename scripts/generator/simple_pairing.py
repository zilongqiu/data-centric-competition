
import random
from PIL import Image
import numpy as np
import os
import uuid

def simplePairing(backgrounds, numberImage):
    img_a_array = np.asarray(numberImage)

    # pick one image from the pool
    img_b = random.choice(backgrounds)
    img_b_array = np.asarray(img_b)

    # mix two images
    mean_img = np.mean([img_a_array, img_b_array], axis=0)
    outputImage = Image.fromarray(np.uint8(mean_img))

    outputImage.save('pairing_'+numberImage+''+uuid.uuid4().hex+".png")

backgrounds = []
for subdir, dirs, files in os.walk('./backgrounds'):
    for file in files:
        filepath = subdir + os.sep + file

        if filepath.endswith(".png"):
            image = Image.open(filepath)
            backgrounds.append(file)

numberImage = Image.open('./numbers/I_0.png')
simplePairing(backgrounds, numberImage)
