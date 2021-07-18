# import the necessary packages
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def compare_images(imageA, imageB, display, title):
    # compute the mean squared error and structural similarity
    # index for the images
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB)

    if display:
        # setup the figure
        fig = plt.figure(title)
        plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
        # show first image
        ax = fig.add_subplot(1, 2, 1)
        plt.imshow(imageA, cmap=plt.cm.gray)
        plt.axis("off")
        # show the second image
        ax = fig.add_subplot(1, 2, 2)
        plt.imshow(imageB, cmap=plt.cm.gray)
        plt.axis("off")
        # show the images
        plt.show()

    return {'mse': m, 'ssim': s}

def similarity_for_folder(original, targetFolderPath):
    similarityScore = 0
    mseScore = 0
    similarityImagePath = ""
    highScores = []
    for subdir, dirs, files in os.walk(targetFolderPath):
        for file in files:
            filepath = subdir + os.sep + file

            if filepath.endswith(".png"):
                # print(filepath)
                compare = cv2.imread(filepath)
                compare = cv2.cvtColor(compare, cv2.COLOR_BGR2GRAY)
                compare = cv2.resize(compare, (original.shape[1], original.shape[0]))
                data = compare_images(original, compare, False, "")

                if data['ssim'] > similarityScore:
                    similarityImagePath = filepath
                    similarityScore = data['ssim']
                    mseScore = data['mse']

                if data['ssim'] > 0.75:
                    highScores.append({'score': data['ssim'], 'mse': data['mse'], 'path': filepath})

        return {'similarityScore': similarityScore,
                'similarityImagePath': similarityImagePath,
                'mseScore': mseScore,
                'similarityHighScores': highScores}


originalImagePath = "./../label_book/IX/a7967416-ce5d-11eb-b317-38f9d35ea60f.png"
original = cv2.imread(originalImagePath)
original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

data = similarity_for_folder(original, "./../best/train/iii")
similarityScore = data['similarityScore']
similarityImagePath = data['similarityImagePath']
mseScore = data['mseScore']
similarityHighScores = data['similarityHighScores']

if similarityScore != 0:
    mostSimilarImage = cv2.imread(similarityImagePath)
    mostSimilarImage = cv2.cvtColor(mostSimilarImage, cv2.COLOR_BGR2GRAY)
    mostSimilarImage = cv2.resize(mostSimilarImage, (original.shape[1], original.shape[0]))
    compare_images(original, mostSimilarImage, True, "Original vs. Most similar")
    print("SSIM: %.2f" % (similarityScore))
    print("MSE: %.2f" % (mseScore))
    print("Original: %s" % (originalImagePath))
    print("Most similar image: %s" % (similarityImagePath))

#if len(similarityHighScores):
#    # initialize the figure
#    fig = plt.figure("High similarity images")
    # loop over the images
#    for (i, (data)) in enumerate(similarityHighScores):
#        # show the image
#        ax = fig.add_subplot(1, 1, 1)
#        ax.set_title(data['mse'])
#        image = cv2.imread(data['path'])
#        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#        image = cv2.resize(image, (original.shape[1], original.shape[0]))
#        plt.imshow(image, cmap = plt.cm.gray)
#        plt.axis("off")
    # show the figure
#       plt.show()