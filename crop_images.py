from PIL import Image
import os

RADIUS = 600
SOURCE_DIR = "../dataset_sadalla/dataset_fo_unicas_500N_500G_jpeg"
DESTINATION_DIR = f"{SOURCE_DIR}_RAD_{RADIUS}"


img_example = Image.open(f"{SOURCE_DIR}/Normal/1.2.392.200106.1651.4.2.228164113206078245.1546966790.66.jpg")

width_ref, height_ref = img_example.size

mid_width = width_ref // 2
mid_height = height_ref // 2

left = mid_width - RADIUS
right = mid_width + RADIUS

top = mid_height - RADIUS
bottom = mid_height + RADIUS


try:
    os.makedirs(DESTINATION_DIR)
except FileExistsError:
    print("O diretorio '{}' ja foi criado.\n".format(DESTINATION_DIR))


for label in os.listdir(SOURCE_DIR):
    dir_label = os.path.join(SOURCE_DIR, label)

    try:
        label_folder_destination = os.path.join(DESTINATION_DIR, label)
        os.makedirs(label_folder_destination)
    except FileExistsError:
        print("O diretorio '{}' ja foi criado.\n".format(label_folder_destination))

    for image_file in os.listdir(dir_label):
        image_path_file = os.path.join(dir_label, image_file)
        img_in = Image.open(image_path_file)

        width, height = img_in.size

        if width == width_ref and height == height_ref:
            img_out = img_in.crop((left, top, right, bottom))
            image_out_file = os.path.join(DESTINATION_DIR, label, image_file)
            img_out.save(image_out_file)
        else:
            print(f"Erro ao cortar a imagem {image_file}.")


