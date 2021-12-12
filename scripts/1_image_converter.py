#%% Importando as bibliotecas

import pydicom as dicom
import os


#%% Especificando o formato de saida e o diretorio de entrada

# Make it True if you want in PNG format
PNG = False

# Specify the .dcm folder path
input_folder_path = "Glaucoma/Glaucoma/DICOM-STORAGE/2019/"


#%% Especificando o diretorio de saida, com base no formato escolhido

# Specify the output jpg/png folder path
if PNG == True:
    output_folder_path = "Glaucoma/Glaucoma/PNG-STORAGE/2019/"
else:
    output_folder_path = "Glaucoma/Glaucoma/JPG-STORAGE/2019/"
    
# Specify the text output path
text_output_path = "Glaucoma/Glaucoma/TXT-STORAGE/2019/"


#%% Criando o diretorio de saida das imagens, caso ainda nao exista

try:
    os.makedirs(os.path.dirname(output_folder_path))
except FileExistsError:
    print("O diretorio '{}' ja foi criado.\n".format(output_folder_path))
    
# Criando o diretorio de saida dos textos, caso ainda nao exista
try:
    os.makedirs(os.path.dirname(text_output_path))
except FileExistsError:
    print("O diretorio '{}' ja foi criado.\n".format(text_output_path))

# Setando o contador de imagem
count_img = 0


#%% Funcao para salvar em um dicionario o caminho (chave) e os arquivos (valores)

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
                                   
    print("Count files:", count)
    return dict_dir, count


#%% Aplicando a funcao para o diretorio de entrada

# Pegando um dicionario 'dir_img', sendo:
    # chave = diretorio_imagens
    # valor = lista com os nomes das imagens

dir_img, count_img = dir_walker(input_folder_path)


#%% Criando os diretorios de saida

for k_dir in dir_img.keys():
    
    mes = k_dir.split("/")[-2]
    dia = k_dir.split("/")[-1]
    
    dir_mes = os.path.join(output_folder_path, mes) + '/'
    dir_dia = os.path.join(dir_mes, dia) + '/'
    
    # Criando o diretorio do mes
    try:
        os.makedirs(os.path.dirname(dir_mes))
        print("Criado o diretorio do mes:", mes)
    except FileExistsError:
        print("O diretorio '{}' ja foi criado.\n".format(dir_mes))
    
    # Criando o diretorio do dia
    try:
        os.makedirs(os.path.dirname(dir_dia))
        print("Criado o diretorio do dia:", dia)
    except FileExistsError:
        print("O diretorio '{}' ja foi criado.\n".format(dir_dia))
    

#%% Para cada diretório chave, extrair as imagens para o formato informado

import cv2

# --------------------------  METODO 1  ---------------------------------------

for n, k_dir in enumerate(dir_img.keys()):
    print("Dir {}: \t Startting conversion of path {} ...".format(n, k_dir))
    for v_dir in dir_img[k_dir]:
        
        img_dicom = os.path.join(k_dir, v_dir)
        # print(img_dicom)
        ds = dicom.dcmread(img_dicom)
        pixel_array_numpy = ds.pixel_array
        
        if PNG == True:
            img_out_name = v_dir.replace('.dcm', '.png')
            dir_out_img = k_dir.replace('DICOM', 'PNG')
        else:
            img_out_name = v_dir.replace('.dcm', '.jpg')
            dir_out_img = k_dir.replace('DICOM', 'JPG')
        
        img_new = os.path.join(dir_out_img, img_out_name)
        cv2.imwrite(img_new, pixel_array_numpy)
        # print("Image converted: \t", img_new)
            
    print("{} files extract with sucessfully!\n".format(k_dir))


#%% Para cada diretório chave, extrair as imagens para o formato informado
    
import matplotlib.pyplot as plt

# --------------------------  METODO 2  ---------------------------------------

for n, k_dir in enumerate(dir_img.keys()):
    print("Dir {}: \t Startting conversion of path {} ...".format(n, k_dir))
    for v_dir in dir_img[k_dir]:
        
        img_dicom = os.path.join(k_dir, v_dir)
        # print(img_dicom)
        ds = dicom.dcmread(img_dicom)
        pixel_array_numpy = ds.pixel_array
        
        plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
        
        if PNG == True:
            img_out_name = v_dir.replace('.dcm', '.png')
            dir_out_img = k_dir.replace('DICOM', 'PNG')
        else:
            img_out_name = v_dir.replace('.dcm', '.jpg')
            dir_out_img = k_dir.replace('DICOM', 'JPG')
        
        img_new = os.path.join(dir_out_img, img_out_name)
        cv2.imwrite(img_new, pixel_array_numpy)
        # print("Image converted: \t", img_new)
            
    print("{} files extract with sucessfully!\n".format(k_dir))


#%% Extraindo as informacoes nao sensiveis sobre as imagens dicom para txt

for n, k_dir in enumerate(dir_img.keys()):
    print("Dir {}: \t Startting conversion of path {} ...".format(n, k_dir))
    for v_dir in dir_img[k_dir]:
        
        img_dicom = os.path.join(k_dir, v_dir)
        # print(img_dicom)
        ds = dicom.dcmread(img_dicom)
        
        txt_out_name = v_dir.replace('.dcm', '.txt')
        dir_out_txt = k_dir.replace('DICOM', 'TXT')      
        
        ds.remove_private_tags()
        
        txt_file = os.path.join(dir_out_txt, txt_out_name)
        with open(txt_file, "w") as arq_out:
            arq_out.write(str(ds))
            


