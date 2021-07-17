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

# load the images -- the original, the original + contrast,
# and the original + photoshop
originalImagePath = "./../label_book/II/a3e33dcc-ce5d-11eb-b317-38f9d35ea60f.png"
classBlocked = "II"
original = cv2.imread(originalImagePath)
original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
#contrast = cv2.imread("./../label_book/vii/a5ade382-ce5d-11eb-b317-38f9d35ea60f.png")
def similarity_for_folder(original, targetFolderPath):
    similarityScore = 0
    mseScore = 0
    similarityImagePath = ""
    highScores = []
    for subdir, dirs, files in os.walk(targetFolderPath):
        for file in files:
            filepath = subdir + os.sep + file

            if filepath.endswith(".png"):
                #print(filepath)
                compare = cv2.imread(filepath)
                compare = cv2.cvtColor(compare, cv2.COLOR_BGR2GRAY)
                compare = cv2.resize(compare, (original.shape[1], original.shape[0]))
                data = compare_images(original, compare, False, "")

                if data['ssim'] > similarityScore:
                    similarityImagePath = filepath
                    similarityScore = data['ssim']
                    mseScore = data['mse']

                if data['ssim'] > 0.82:
                    highScores.append({'score': data['ssim'], 'mse': data['mse'], 'path': filepath})

        return {'similarityScore': similarityScore,
                'similarityImagePath': similarityImagePath,
                'mseScore': mseScore,
                'similarityHighScores': highScores}

#data = similarity_for_folder(original, "./../best/train/iii")
#similarityScore = data['similarityScore']
#similarityImagePath = data['similarityImagePath']
#mseScore = data['mseScore']
#similarityHighScores = data['similarityHighScores']

#if similarityScore != 0:
#    mostSimilarImage = cv2.imread(similarityImagePath)
#    mostSimilarImage = cv2.cvtColor(mostSimilarImage, cv2.COLOR_BGR2GRAY)
#    mostSimilarImage = cv2.resize(mostSimilarImage, (original.shape[1], original.shape[0]))
#    compare_images(original, mostSimilarImage, False, "Original vs. Most similar")
#    print("\n\nSSIM: %.2f" % (similarityScore))
#    print("\n\nMSE: %.2f" % (mseScore))
#    print("Original: %s" % (originalImagePath))
#    print("Most similar image: %s" % (similarityImagePath))

def averageHighScores(similarityHighScores):
    print("Similar count: %d" % (len(similarityHighScores)))
    totalScore = 0
    totalMse = 0
    for data in similarityHighScores:
        #print("SSIM: %.2f | MSE: %.2f for %s" % (data['score'], data['mse'], data['path']))
        totalScore += data['score']
        totalMse += data['mse']
    
    if totalScore != 0:
        print("\nAverage SSIM: %.2f | Average MSE: %.2f" % (totalScore/len(similarityHighScores), totalMse/len(similarityHighScores)))

    if len(similarityHighScores) != 0:
        return {'averageSSIM': totalScore / len(similarityHighScores),
                'averageMSE': totalMse / len(similarityHighScores)}
    else:
        return {'averageSSIM': -1,
                'averageMSE': 999999999999}

def highestScores(similarityHighScores):
    print("Similar count: %d" % (len(similarityHighScores)))
    bestSSIM = 0
    bestMSE = 1
    for data in similarityHighScores:
        # print("SSIM: %.2f | MSE: %.2f for %s" % (data['score'], data['mse'], data['path']))
        if data['mse'] < bestMSE:
            bestMSE += data['mse']
            bestSSIM += data['ssim']

    return {'bestSSIM': bestSSIM, 'bestMSE': bestMSE}

#averageHighScores(similarityHighScores)

class_names = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X']

bestMSEScore = 999999999
bestClassName = ''
for number in class_names:
    for subdir, dirs, files in os.walk('./../best/train/' + number):
        print('\n=========')
        print(number + ' (' +subdir+')')
        print('=========')
        data = similarity_for_folder(original, subdir)
        similarityScore = data['similarityScore']
        similarityImagePath = data['similarityImagePath']
        mseScore = data['mseScore']
        similarityHighScores = data['similarityHighScores']

        if similarityScore != 0:
            mostSimilarImage = cv2.imread(similarityImagePath)
            mostSimilarImage = cv2.cvtColor(mostSimilarImage, cv2.COLOR_BGR2GRAY)
            mostSimilarImage = cv2.resize(mostSimilarImage, (original.shape[1], original.shape[0]))
            compare_images(original, mostSimilarImage, False, "Original vs. Most similar")
            print("SSIM: %.2f" % (similarityScore))
            print("MSE: %.2f" % (mseScore))
            print("Original: %s" % (originalImagePath))
            print("Most similar image: %s" % (similarityImagePath))
        #averageScores = averageHighScores(similarityHighScores)
        #averageSSIM = averageScores['averageSSIM']
        #averageMSE = averageScores['averageMSE']

        if number != classBlocked:
            for similar in similarityHighScores:
                os.remove(similar['path'])

        #if (averageMSE < bestMSEScore):
        #    bestMSEScore = averageMSE
        #    bestClassName = number

        bestScores = highestScores(similarityHighScores)
        bestMSE = bestScores['bestMSE']

        if (bestScores['bestSSIM'] < bestMSEScore):
            bestMSEScore = bestScores['bestSSIM']
            bestClassName = number

print("\n\nBest matching class: %s" % (bestClassName))
print("Best class MSE: %.2f" % (bestMSEScore))

#contrast = cv2.imread("./../data/train/iii/b0a6c0e2-ce5d-11eb-b317-38f9d35ea60f_0_.png")
#shopped = cv2.imread("./../data/train/i/ab9fb784-ce5d-11eb-b317-38f9d35ea60f_0_.png")
# convert the images to grayscale
#original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
#contrast = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)
#contrast = cv2.resize(contrast, (original.shape[1],original.shape[0]))
#shopped = cv2.cvtColor(shopped, cv2.COLOR_BGR2GRAY)

# initialize the figure
#fig = plt.figure("Images")
#images = ("Original", original), ("Contrast", contrast), ("Photoshopped", shopped)
# loop over the images
#for (i, (name, image)) in enumerate(images):
    # show the image
    #ax = fig.add_subplot(1, 3, i + 1)
    #ax.set_title(name)
    #plt.imshow(image, cmap = plt.cm.gray)
    #plt.axis("off")
# show the figure
#plt.show()
# compare the images
#compare_images(original, original, "Original vs. Original")
#compare_images(original, contrast, "Original vs. Contrast")
#compare_images(original, shopped, "Original vs. Photoshopped")