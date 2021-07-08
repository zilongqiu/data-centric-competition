import imgaug as ia
from imgaug import augmenters as iaa
import imageio
import glob
import os

DIR = os.getcwd()

def main():
  print('Start Augment Inputs')
  clean_output_files()
  image_paths = glob.glob(f'{DIR}/input/*')
  ia.seed(1)
  seq = iaa.Sequential([
      iaa.Crop(percent=(0, 0.2)),
      sometimes(iaa.Affine(
        scale={"x": (0.95, 1.05), "y": (0.95, 1.05)},
        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
        #rotate=(-45, 45),
        rotate=(-25, 25),
      )),
      iaa.GaussianBlur(sigma=(0, 3.0)),
      iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
      iaa.Add((-10, 10)),
  ])

  for image_path in image_paths:
    print(f'proc {image_path}')
    img = imageio.imread(image_path)
    splited_name = os.path.splitext(os.path.basename(image_path))
    for i in range(1):
      output_path = f'{DIR}/output/{splited_name[0]}_{i}_{splited_name[1]}'
      imageio.imwrite(output_path, seq.augment_image(img))

def clean_output_files():
  files = glob.glob(f'{DIR}/output/*')
  for f in files:
    os.remove(f)

def sometimes(aug):
  return iaa.Sometimes(0.3, aug)

main()
