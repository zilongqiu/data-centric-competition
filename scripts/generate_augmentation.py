
import sys
import os
import pathlib

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

total = 0
lastFolder = ''

for subdir, dirs, files in os.walk('./../results/v2/train'):
    for file in files:
        filepath = subdir + os.sep + file

        if filepath.endswith(".png"):
            path = pathlib.PurePath(os.path.dirname(os.path.join(subdir, file)))
            folderName = path.name.upper()

            if lastFolder != folderName:
                print(folderName)
                lastFolder = folderName

            print('    - ' + filepath)
            datagen = ImageDataGenerator(
                    rotation_range=40,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    fill_mode='nearest')

            img = load_img(filepath)  # this is a PIL image
            x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
            x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

            # the .flow() command below generates batches of randomly transformed images
            # and saves the results to the `preview/` directory
            i = 0
            for batch in datagen.flow(x, batch_size=1,
                                      save_to_dir='preview/'+folderName+'/', save_prefix='image', save_format='png'):
                i += 1
                total += 1
                if i > 2:
                    break  # otherwise the generator would loop indefinitely

print('==============================')
print('TOTAL GENERATED: ' + str(total))
print('==============================')
