
import random
from PIL import Image
import numpy as np
import os

def simplePairing(backgrounds, numberImage):
    img_a_array = np.asarray(numberImage)

    # pick one image from the pool
    img_b, _ = random.choice(backgrounds)
    img_b_array = np.asarray(img_b.resize((240, 240)))

    # mix two images
    mean_img = np.mean([img_a_array, img_b_array], axis=0)
    img = Image.fromarray(np.uint8(mean_img))

    return img

backgrounds = []
for subdir, dirs, files in os.walk('./backgrounds'):
    for file in files:
        filepath = subdir + os.sep + file

        if filepath.endswith(".png"):
            backgrounds.append(filepath)

