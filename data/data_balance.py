
import os

class_names = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X']

print('Training:')
for class_name in class_names:
    path, dirs, files = next(os.walk("./train/" + class_name))
    file_count = len(files)
    print(class_name + ' | ' + str(file_count))
print('')

print('Validation:')
for class_name in class_names:
    path, dirs, files = next(os.walk("./val/" + class_name))
    file_count = len(files)
    print(class_name + ' | ' + str(file_count))
print('\n')
