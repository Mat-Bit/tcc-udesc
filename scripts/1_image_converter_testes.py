#%% Importando as bibliotecas

import pydicom as dicom
import os


#%% Especificando o formato de saida e o diretorio de entrada

# Make it True if you want in PNG format
PNG = False

# Specify the .dcm folder path
input_folder_path = "Glaucoma/Glaucoma/DICOM-STORAGE/2019/"


#%% Qual metodo de extracao das imagens sera utilizado

metodo = {1 : "OpenCV_1", 2 : "Matplotlib_1", 3 : "PIL_1", 4 : "Scipy"}

n_metodo = 4

output_folder_path = "Glaucoma/Glaucoma/Amostragem/" + metodo[n_metodo] + "/"


#%% Especificando o diretorio de saida, com base no formato escolhido

# Specify the output jpg/png folder path
if PNG == True:
    output_folder_path = output_folder_path + "PNG/"
else:
    output_folder_path = output_folder_path + "JPEG/"
    

#%% Criando o diretorio de saida das imagens, caso ainda nao exista

try:
    os.makedirs(os.path.dirname(output_folder_path))
except FileExistsError:
    print("O diretorio '{}' ja foi criado.\n".format(output_folder_path))


#%% Funcao para salvar em um dicionario o caminho (chave) e os arquivos (valores)

# Setando o contador de imagem
count_img = 0

def dir_walker(root_dir):
    dict_dir = {}
    count = 0
    # Iterando entre o diretorio ano (2019)
    for r0, d0, f0 in os.walk(root_dir):
        if len(d0) > 0:
            for d in d0:
                # Iterando entre os diretorios 'meses'
                for r1, d1, f1 in os.walk(os.path.join(r0, d)):
                    if len(d1) > 0:
                        d1.sort()
                        for dd in d1:
                            # Iterando entre os diretorios 'dias'
                            for r2, d2, f2 in os.walk(os.path.join(r1, dd)):
                                if len(d2) == 0:
                                    f2.sort()
                                    dict_dir[r2] = f2
                                    print(r2)
                                    count += len(f2)
        else:
            f0.sort()
            dict_dir[r0] = f0
            print(r0)
            count += len(f0)
                                   
    print("Count files:", count)
    return dict_dir, count


#%% Aplicando a funcao para o diretorio de entrada
    
# Pegando um dicionario 'dir_img', sendo:
    # chave = diretorio_imagens
    # valor = lista com os nomes das imagens

dir_img, count_img = dir_walker(input_folder_path)


#%% Criando os diretorios de saida

for k_dir in dir_img.keys():
    
    ano = k_dir.split("/")[-4]
    mes = k_dir.split("/")[-3]
    dia = k_dir.split("/")[-2]
    
    fim_path = ano + "-" + mes + "-" + dia + "/"
    
    dir_fim = os.path.join(output_folder_path, fim_path)

    
    # Criando o diretorio do mes
    try:
        os.makedirs(os.path.dirname(dir_fim))
        print("Criado o diretorio final de saida:", fim_path)
    except FileExistsError:
        print("O diretorio '{}' ja foi criado.\n".format(dir_fim))
    

#%% Extracao Metodo 1 - OpenCV

import cv2

# --------------------------  METODO 1  ---------------------------------------
if n_metodo == 1:

    for n, k_dir in enumerate(dir_img.keys()):
        print("Dir {}: \t Startting conversion of path {} ...".format(n, k_dir))
        for v_dir in dir_img[k_dir]:
            
            img_dicom = os.path.join(k_dir, v_dir)
            # print(img_dicom)
            ds = dicom.dcmread(img_dicom)
            pixel_array_numpy = ds.pixel_array
            
            if PNG == True:
                img_out_name = v_dir.replace('.dcm', '.png')
            else:
                img_out_name = v_dir.replace('.dcm', '.jpg')
            
            img_new = os.path.join(dir_fim, img_out_name)
            cv2.imwrite(img_new, pixel_array_numpy)
            # print("Image converted: \t", img_new)
                
        print("{} files extract with sucessfully!\n".format(k_dir))


#%% Extracao Metodo 2 - Matplotlib
    
import matplotlib.pyplot as plt

# --------------------------  METODO 2  ---------------------------------------
if n_metodo == 2:
    
    for n, k_dir in enumerate(dir_img.keys()):
        print("Dir {}: \t Startting conversion of path {} ...".format(n, k_dir))
        for v_dir in dir_img[k_dir]:
            
            img_dicom = os.path.join(k_dir, v_dir)
            # print(img_dicom)
            ds = dicom.dcmread(img_dicom)
            pixel_array_numpy = ds.pixel_array
            
            if PNG == True:
                img_out_name = v_dir.replace('.dcm', '.png')
            else:
                img_out_name = v_dir.replace('.dcm', '.jpg')
            
            img_new = os.path.join(dir_fim, img_out_name)
            # plt.imshow(pixel_array_numpy)
            
            plt.imsave(img_new, pixel_array_numpy)
    
            # print("Image converted: \t", img_new)
                
        print("{} files extract with sucessfully!\n".format(k_dir))


#%% Extracao Metodo 3 - PIL

# import contrib-pydicom
from pydicom_PIL import show_PIL, save_PIL

# import pydicom.contrib.pydicom_PIL.show_PIL as show_pil

# --------------------------  METODO 3  ---------------------------------------
if n_metodo == 3:
    
    for n, k_dir in enumerate(dir_img.keys()):
        print("Dir {}: \t Startting conversion of path {} ...".format(n, k_dir))
        for v_dir in dir_img[k_dir]:
            
            img_dicom = os.path.join(k_dir, v_dir)
            print(img_dicom)
            ds = dicom.read_file(img_dicom)
            
            # show_PIL(ds)
            
            if PNG == True:
                img_out_name = v_dir.replace('.dcm', '.png')
            else:
                img_out_name = v_dir.replace('.dcm', '.jpg')
            
            img_new = os.path.join(dir_fim, img_out_name)
            # plt.imshow(pixel_array_numpy)
            
            save_PIL(ds, img_new)
            
            # plt.imsave(img_new, pixel_array_numpy)
    
            # print("Image converted: \t", img_new)
                
        print("{} files extract with sucessfully!\n".format(k_dir))


#%% Extracao Metodo 4: Usando o Scipy

import matplotlib.pyplot as plt
import scipy.misc
# import pandas as pd
# import numpy as np

# --------------------  METODO 4  ---------------------------
if n_metodo == 4:
    images_dicom = dir_img[input_folder_path]
    
    for img_dicom_name in images_dicom:
        input_image = os.path.join(input_folder_path, img_dicom_name)
        
        output_image_jpg = img_dicom_name.replace('.dcm', '.jpg')
        output_image = os.path.join(output_folder_path, output_image_jpg)
        
        ds = dicom.read_file(input_image)
        img = ds.pixel_array
        r = img[0]
        
        plt.imshow(img)
        # scipy.misc.imsave(output_image, img)
        
        print("Saving the {} at {} path.".format(output_folder_path, output_image_jpg))
    

      
            