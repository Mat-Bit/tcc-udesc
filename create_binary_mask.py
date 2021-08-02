import numpy as np
import cv2
import os
import time


# create binary mask of the images where the red channel is grathee than 28
# and aplly a opening morfology on the image to remove the noise with a kernel size of 3x3
def create_binary_mask(image_path_filename, saved_location, label, image_base_name, image_ext):
    img_in = cv2.imread(image_path_filename)
    crop_img = img_in[30:1910, 40:1920]
    b, g, r = cv2.split(crop_img)
    r = r.astype(np.uint8)

    r_binary = np.zeros_like(r)
    r_binary[r > 28] = 1
    kernel = np.ones((3, 3), np.uint8)
    r_binary = cv2.morphologyEx(r_binary, cv2.MORPH_OPEN, kernel)
    r_binary = r_binary.astype(np.uint8)
    r_binary = r_binary * 255

    img_out_name = image_base_name + '_binary_mask' + image_ext
    img_out_path = os.path.join(saved_location, label, img_out_name)
    cv2.imwrite(img_out_path, r_binary)


# main function
def main():
    IMAGE_ROOT_PATH = '../dataset_sadalla/dataset_fo_unicas_10N_10G_JPG'
    saved_location = IMAGE_ROOT_PATH + '_binary_mask'

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

                create_binary_mask(file_path, saved_location, label, image_base_name, image_ext)

    
    end_time = time.time()
    print("Pre processing completed in {:.2f}s".format(end_time - start_time))


if __name__ == '__main__':
    main()
