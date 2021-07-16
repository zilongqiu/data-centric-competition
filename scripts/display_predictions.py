
import sys
import os
import pathlib
import webbrowser

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
original_stdout = sys.stdout

with open('./wrong_predictions.html', 'w') as f:
    sys.stdout = f
    print('<html>', file=f)
    print('<head>', file=f)
    print('<meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">', file=f)
    print('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css">', file=f)
    print('<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.5.0/font/bootstrap-icons.css">', file=f)
    print('</head>', file=f)
    print('<body>', file=f)
    print('<div class="container" style="margin-bottom: 200px;">', file=f)

    for number in class_names:
        for subdir, dirs, files in os.walk('./../label_book/'+number):
            print('<div class="row results">', file=f)
            for file in files:
                filepath = subdir + os.sep + file

                if filepath.endswith(".png"):
                    path = pathlib.PurePath(os.path.dirname(os.path.join(subdir, file)))
                    folderName = path.name.upper()

                    if lastFolder != folderName:
                        print('<h1>'+folderName+'</h1><br>', file=f)
                        lastFolder = folderName

                    # predicting images
                    img = image.load_img(filepath, target_size=(img_width, img_height))
                    x = image.img_to_array(img)
                    x = np.expand_dims(x, axis=0)

                    images = np.vstack([x])
                    classes = np.argmax(model.predict(images), axis=1)
                    prediction = class_names[classes[0]]

                    total += 1
                    if prediction == folderName:
                        print('<div class="col-lg-4 corrects" style="height:200px; text-align:center;">', file=f)
                        print('<i class="bi-hand-thumbs-up-fill" style="font-size: 4rem; color: green; position:absolute; left: 60px"></i>', file=f)
                        print('<img src="'+filepath+'" alt="'+filepath+'" style="max-width: 150px">', file=f)
                        print('</div>', file=f)
                        correct += 1
                    else:
                        print('<div class="col-lg-4 wrongs" style="height:200px; text-align:center;">', file=f)
                        print('<i class="bi-hand-thumbs-down-fill" style="font-size: 4rem; color: red; position:absolute; left: 60px"></i>', file=f)
                        print('<p style="position:absolute; left: 100px; font-weight:bold; font-size: 2em;">' + prediction +'</p>', file=f)
                        print('<img src="'+filepath+'" alt="404" style="max-width: 150px">', file=f)
                        print('</div>', file=f)
                        wrong += 1
            print('</div>', file=f)

    print('<div class="row" id="scores" style="height: 200px; position: fixed; bottom: 0; background: white; font-weight: bold; font-size: 20px;">', file=f)
    print('<p>TOTAL: '+str(total)+'</p>', file=f)
    print('<p>CORRECT: '+str(correct)+'</p>', file=f)
    print('<p>WRONG: '+str(wrong)+'</p>', file=f)
    print('<p>RESULT: '+str(correct / total * 100)+'%</p>', file=f)
    print('</div>', file=f)

    print('<div class="row" style="height: 200px; margin-left: 500px; position: fixed; bottom: 0; background: white; font-weight: bold; font-size: 20px;">', file=f)
    print('<button type="button" class="btn btn-danger" id="button-wrong">WRONGs</button>', file=f)
    print('<button type="button" class="btn btn-success" id="button-correct">CORRECTs</button>', file=f)
    print('<button type="button" class="btn btn-primary" id="button-all">ALL</button>', file=f)
    print('</div>', file=f)

    print('<script src="https://code.jquery.com/jquery-3.3.1.slim.min.js" integrity="sha384-q8i/X+965DzO0rT7abK41JStQIAqVgRVzpbzo5smXKp4YfRvH+8abtTE1Pi6jizo" crossorigin="anonymous"></script>', file=f)
    print('<script>', file=f)
    print("$('#button-wrong').on('click', function (e) { $('.wrongs').show(); $('.corrects').hide(); });", file=f)
    print("$('#button-correct').on('click', function (e) { $('.wrongs').hide(); $('.corrects').show(); });", file=f)
    print("$('#button-all').on('click', function (e) { $('.wrongs').show(); $('.corrects').show(); });", file=f)
    print('</script>', file=f)
    print('</div>', file=f)
    print('</body>', file=f)
    sys.stdout = original_stdout
f.close()

webbrowser.open('file:///'+os.getcwd()+'/wrong_predictions.html')