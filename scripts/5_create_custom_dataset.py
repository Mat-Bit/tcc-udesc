#%% Importando as libs

import os
import shutil
import pandas as pd
import csv
from random import randint


#%% Parametros de execucao

# Se = 0: move todos os arquivos dos diretorios recursivamente
# OU se > tamanho do dataset, pega a quantidade maxima do dataset
NUM_COPY_IMAGES_NORMAL = 10
NUM_COPY_IMAGES_GLAUCOMA = 10

IMAGE_EXT = 'jpg'

DESTINATION_FOLDER_ROOT = 'dataset_sadalla'

TYPE_OF_IMAGES = 'FO_Unicas'

TYPE_OF_IMAGES_LOWER = TYPE_OF_IMAGES.lower()

LABELS = ['Normal', 'Glaucoma']


#%% Lendo arquivo CSV das imagens 

csv_files_map = os.path.join('csv', f'{TYPE_OF_IMAGES_LOWER}.csv')

df_info_imgs = pd.read_csv(csv_files_map)

df_normal = df_info_imgs[df_info_imgs.Label == 'Normal']
df_normal.reset_index(inplace=True, drop=True)

df_glaucoma = df_info_imgs[df_info_imgs.Label == 'Glaucoma']
df_glaucoma.reset_index(inplace=True, drop=True)

TOTAL_NORMAL_IMGS = len(df_normal)
TOTAL_GLAUCOMA_IMGS = len(df_glaucoma)


#%% Preparando o diretorio do dataset customizado

folder_destination = f'dataset_{TYPE_OF_IMAGES_LOWER}_{NUM_COPY_IMAGES_NORMAL}N_{NUM_COPY_IMAGES_GLAUCOMA}G_{IMAGE_EXT.upper()}'

try:
    final_folder_destination = os.path.join(DESTINATION_FOLDER_ROOT, folder_destination)
    os.makedirs(final_folder_destination)
except FileExistsError:
    print("O diretorio '{}' ja foi criado.\n".format(final_folder_destination))

if NUM_COPY_IMAGES_NORMAL <= 0 or NUM_COPY_IMAGES_NORMAL > TOTAL_NORMAL_IMGS:
    NUM_COPY_IMAGES_NORMAL = TOTAL_NORMAL_IMGS
if NUM_COPY_IMAGES_GLAUCOMA <= 0 or NUM_COPY_IMAGES_GLAUCOMA > TOTAL_GLAUCOMA_IMGS:
    NUM_COPY_IMAGES_GLAUCOMA = TOTAL_GLAUCOMA_IMGS

for label in LABELS:
    try:
        label_folder_destination = os.path.join(final_folder_destination, label)
        os.makedirs(label_folder_destination)
    except FileExistsError:
        print("O diretorio '{}' ja foi criado.\n".format(label_folder_destination))


#%% Abrindo arquivo de log e colocando o cabecalho

log_file = open(f'{DESTINATION_FOLDER_ROOT}/{folder_destination}.csv', 'w', encoding='UTF8')

csv_writer = csv.writer(log_file)

header = ['Incluso_Dataset', 'Imagem', 'Glaucoma']

csv_writer.writerow(header)


#%% Copiando as imagens classificadas como 'Normal' para o Dataset personalizado

for normal_img in range(NUM_COPY_IMAGES_NORMAL):
    label = LABELS[0]
    i = randint(0, len(df_normal) - 1)

    filename_original = df_normal.loc[i, 'Imagem'].copy()
    source_file = filename_original.replace('.jpg', f'.{IMAGE_EXT}')
    
    filename = source_file.split('/')[-1]
    dest_file = os.path.join(final_folder_destination, label, filename)
    
    row = [True, source_file, 0]

    try:
        shutil.copyfile(source_file, dest_file)
        csv_writer.writerow(row)
        df_normal.drop(df_normal[df_normal.Imagem == filename_original].index, inplace=True)
        df_normal.reset_index(inplace=True, drop=True)
        print("Arquivo {} copiado para {}".format(filename, dest_file))
    except FileNotFoundError:
        print("Erro ao copiar o arquivo {}".format(source_file))


#%% Copiando as imagens classificadas como 'Glaucoma' para o Dataset personalizado

for glaucoma_img in range(NUM_COPY_IMAGES_GLAUCOMA):
    label = LABELS[1]
    i = randint(0, len(df_glaucoma) - 1)

    filename_original = df_glaucoma.loc[i, 'Imagem'].copy()
    source_file = filename_original.replace('.jpg', f'.{IMAGE_EXT}')
    
    filename = source_file.split('/')[-1]
    dest_file = os.path.join(final_folder_destination, label, filename)
    
    row = [True, source_file, 1]

    try:
        shutil.copyfile(source_file, dest_file)
        csv_writer.writerow(row)
        df_glaucoma.drop(df_glaucoma[df_glaucoma.Imagem == filename_original].index, inplace=True)
        df_glaucoma.reset_index(inplace=True, drop=True)
        # print("Arquivo {} copiado para {}".format(filename, dest_file))
    except FileNotFoundError:
        print("Erro ao copiar o arquivo {}".format(source_file))


#%% Colocando as imagens que nao foram selecionadas no arquivo de log

for i in range(len(df_normal)):
    filename_original = df_normal.loc[i, 'Imagem']
    source_file = filename_original.replace('.jpg', f'.{IMAGE_EXT}')
    
    row = [False, source_file, 0]
    csv_writer.writerow(row)

for i in range(len(df_glaucoma)):
    filename_original = df_glaucoma.loc[i, 'Imagem']
    source_file = filename_original.replace('.jpg', f'.{IMAGE_EXT}')
    
    row = [False, source_file, 1]
    csv_writer.writerow(row)
    

log_file.close()


# %%
