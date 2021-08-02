# import the necessary packages
from multiprocessing import process
from multiprocessing.context import Process
import numpy as np
from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt
import time
import multiprocessing as mp


def pre_processing(image_path_filename, saved_location):
    image_ext = '.' + image_path_filename.split('.')[-1]
    image_base_name = image_path_filename.split('/')[-1].strip(image_ext)
    label = image_path_filename.split('/')[-2]

    img_in = cv2.imread(image_path_filename)

    # Step 1: Crop the image to remove the unwanted parts
    crop_img = img_in[30:1910, 40:1920]

    # Step 2: Apply binary mask to the image
    binary_mask = cv2.imread('binary_mask.jpg')
    binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY)
    binary_mask = cv2.resize(binary_mask, (crop_img.shape[1], crop_img.shape[0]))
    binary_mask = binary_mask / 255
    binary_mask = binary_mask.astype(np.uint8)
    b, g, r = cv2.split(crop_img)
    b = b * binary_mask
    g = g * binary_mask
    r = r * binary_mask

    # Step 3: Apply CLAHE on the image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_g = clahe.apply(g)
    img_out = cv2.merge((b, clahe_g, r))
    
    # Step 4: Save the image
    img_out_name = image_base_name + '_pre_processed' + image_ext
    img_out_path = os.path.join(saved_location, label, img_out_name)
    cv2.imwrite(img_out_path, img_out)


# main function
def main():
    IMAGE_ROOT_PATH = '../dataset_sadalla/dataset_fo_unicas_10N_10G_JPG'
    saved_location = IMAGE_ROOT_PATH + '_processed_dataset'

    if os.path.exists(saved_location) is False:
        os.mkdir(saved_location)

    classes = os.listdir(IMAGE_ROOT_PATH)
    path_images = []

    for class_name in classes:
        class_root = os.path.join(IMAGE_ROOT_PATH, class_name)

        if os.path.isdir(os.path.join(saved_location, class_name)) is False:
            os.mkdir(os.path.join(saved_location, class_name))


        for file in os.listdir(class_root):
            if file.endswith('.jpg') or file.endswith('.jpeg'):
                file_path = os.path.join(class_root, file)
                path_images.append(file_path)

    print("Begining processing the images...")

    start_time = time.time()

    # Create a pool of workers
    pool = mp.Pool()
    # Execute the function in parallel
    [pool.apply_async(pre_processing, args=(image_path_filename, saved_location)) for image_path_filename in path_images]
    # Finish the task
    pool.close()
    pool.join()

    end_time = time.time()
    print("Pre processing completed in {:.2f}s".format(end_time - start_time))


if __name__ == '__main__':
    main()

