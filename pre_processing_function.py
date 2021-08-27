# import the necessary packages
import numpy as np
from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt
import time


# Create the histogram plots from of the three channels of the input image 'img_in'
# and save the histogram plots with the '_hist' suffix
def histogram_plot(img_in):
    img_in_array = cv2.imread(img_in)
    b, g, r = cv2.split(img_in_array)
    hist_b = cv2.calcHist([b], [0], None, [256], [1, 256])
    hist_g = cv2.calcHist([g], [0], None, [256], [1, 256])
    hist_r = cv2.calcHist([r], [0], None, [256], [1, 256])
    plt.plot(hist_b, color='b')
    plt.plot(hist_g, color='g')
    plt.plot(hist_r, color='r')
    plt.xlim([-5, 260])
    if img_in.endswith('.jpg') or img_in.endswith('.png'):
        plt.savefig(img_in[:-4] + '_hist.png')
    elif img_in.endswith('.jpeg'):
        plt.savefig(img_in[:-5] + '_hist.png')
    plt.clf()


# Function to crop the image to the desired size
def crop_image(image_path_filename, saved_location, label, image_base_name, image_ext):
    img_in = Image.open(image_path_filename)
    img_out = img_in.crop((40, 30, 1920, 1910))

    img_out_name = image_base_name + '_1_cropped' + image_ext
    img_out_path = os.path.join(saved_location, label, img_out_name)
    img_out.save(img_out_path)
    histogram_plot(img_out_path)

    return img_out_path


# Function to merge the input image with a defined image of binary mask
# where maintain the original image pixel value when the mask is 1
def apply_mask_image(image_path_filename, saved_location, label, image_base_name, image_ext):
    img_in = cv2.imread(image_path_filename)
    mask = cv2.imread('binary_mask.jpg')
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = cv2.resize(mask, (img_in.shape[1], img_in.shape[0]))
    mask = mask / 255
    mask = mask.astype(np.uint8)
    b, g, r = cv2.split(img_in)

    b = b * mask
    g = g * mask
    r = r * mask
    img_out = cv2.merge((b, g, r))

    img_out_name = image_base_name + '_2_binary_mask' + image_ext
    img_out_path = os.path.join(saved_location, label, img_out_name)
    cv2.imwrite(img_out_path, img_out)
    histogram_plot(img_out_path)

    return img_out_path


# Function to apply CLAHE on the image
def clahe_image(image_path_filename, saved_location, label, image_base_name, image_ext):
    img_in = cv2.imread(image_path_filename)
    b, g, r = cv2.split(img_in)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_g = clahe.apply(g)

    img_out = cv2.merge((b, clahe_g, r))

    img_out_name = image_base_name + '_3_clahe_green_channel' + image_ext
    img_out_path = os.path.join(saved_location, label, img_out_name)
    cv2.imwrite(img_out_path, img_out)
    histogram_plot(img_out_path)

    return img_out_path


# main function
def main():
    IMAGE_ROOT_PATH = '../dataset_sadalla/dataset_fo_unicas_10N_10G_JPG'
    saved_location = IMAGE_ROOT_PATH + '_processed'

    if os.path.exists(saved_location) is False:
        os.mkdir(saved_location)

    classes = os.listdir(IMAGE_ROOT_PATH)

    start_time = time.time()

    for class_name in classes:
        print("Pre processing " + class_name + " class images...")
        class_root = os.path.join(IMAGE_ROOT_PATH, class_name)

        if os.path.isdir(os.path.join(saved_location, class_name)) is False:
            os.mkdir(os.path.join(saved_location, class_name))

        for file in os.listdir(class_root):
            if file.endswith('.jpg') or file.endswith('.jpeg'):
                file_path = os.path.join(class_root, file)

                image_ext = '.' + file_path.split('.')[-1]
                image_base_name = file_path.split('/')[-1].strip(image_ext)
                label = file_path.split('/')[-2]

                img_original = cv2.imread(file_path)
                img_out_name = image_base_name + '_0_original' + image_ext
                img_out_path = os.path.join(saved_location, label, img_out_name)
                cv2.imwrite(img_out_path, img_original)
                histogram_plot(img_out_path)


                img_cropped = crop_image(file_path, saved_location, label, image_base_name, image_ext)
                img_masked = apply_mask_image(img_cropped, saved_location, label, image_base_name, image_ext)
                img_clahe = clahe_image(img_masked, saved_location, label, image_base_name, image_ext)
    
    end_time = time.time()
    print("Pre processing completed in {:.2f}s".format(end_time - start_time))


if __name__ == '__main__':
    main()

