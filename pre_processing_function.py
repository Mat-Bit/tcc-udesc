# import the necessary packages
import numpy as np
from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt


def crop_image(image_path_filename, saved_location):
    image_ext = '.' + image_path_filename.split('.')[-1]
    image_base_name = image_path_filename.split('/')[-1].strip(image_ext)
    label = image_path_filename.split('/')[-2]

    img_in = Image.open(image_path_filename)
    img_out = img_in.crop((40, 30, 1920, 1910))

    img_out_name = image_base_name + '_1_cropped' + image_ext
    img_out_path = os.path.join(saved_location, label, img_out_name)
    img_out.save(img_out_path)

    return img_out_path


# function to apply CLAHE on the image
def clahe_image(image_path_filename, saved_location):
    image_ext = '.' + image_path_filename.split('.')[-1]
    image_base_name = image_path_filename.split('/')[-1].strip(image_ext)
    label = image_path_filename.split('/')[-2]

    img_in = cv2.imread(image_path_filename)
    b, g, r = cv2.split(img_in)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe = clahe.apply(g)
    img_out = cv2.merge((b, clahe, r))

    img_out_name = image_base_name + '_2_clahe' + image_ext
    img_out_path = os.path.join(saved_location, label, img_out_name)
    cv2.imwrite(img_out_path, img_out)

    return img_out_path


# main function
def main():
    IMAGE_ROOT_PATH = '../dataset_sadalla/dataset_fo_unicas_10N_10G_JPG/'
    saved_location = IMAGE_ROOT_PATH + '_processed/'

    if os.path.exists(saved_location) is False:
        os.mkdir(saved_location)
    

    classes = os.listdir(IMAGE_ROOT_PATH)
    print(classes)

    for class_name in classes:
        class_root = os.path.join(IMAGE_ROOT_PATH, class_name)

        if os.path.isdir(os.path.join(saved_location, class_name)) is False:
            os.mkdir(os.path.join(saved_location, class_name))


        for file in os.listdir(class_root):
            if file.endswith('.jpg') or file.endswith('.jpeg'):
                file_path = os.path.join(class_root, file)
                img_cropped = crop_image(file_path, saved_location)
                img_clahe = clahe_image(img_cropped, saved_location)
                print("Image", file_path, "cropped and clahe applied")


if __name__ == '__main__':
    main()

