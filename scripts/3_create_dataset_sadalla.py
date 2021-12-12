# Script para mover as imagens ja convertidas em JPG / PNG
# para diretorios separados em Normal e Glaucoma

#%% Importando as libs

import os
import shutil
import pandas as pd
import csv


#%% Parametros de execucao

SOURCE_FOLDER_ROOT = 'Glaucoma/Glaucoma/JPG-STORAGE/2019'

DESTINATION_FOLDER_ROOT = 'dataset_sadalla'

TYPE_OF_IMAGES = ['FO_Unicas', 'Mosaico']

LABELS = ['Normal', 'Glaucoma']

CSV_GLAUCOMA_FILES = 'csv/dados_info.csv'

# Se = 0: move todos os arquivos dos diretorios recursivamente
NUM_MOVED_IMAGES = 0


#%% Criando o diretorio de saida das imagens, caso ainda nao exista

for type_img in TYPE_OF_IMAGES:
    for label in LABELS:
        type_folder_path = os.path.join(DESTINATION_FOLDER_ROOT, type_img)
        output_folder_path = os.path.join(type_folder_path, label + '/')
        try:
            os.makedirs(os.path.dirname(output_folder_path))
        except FileExistsError:
            print("O diretorio '{}' ja foi criado.\n".format(output_folder_path))


#%% Lendo arquivo das imagens de Glaucoma

df_info_imgs = pd.read_csv(CSV_GLAUCOMA_FILES)

TOTAL_NORMAL_IMGS = len(df_info_imgs[df_info_imgs.Glaucoma == 0])
TOTAL_GLAUCOMA_IMGS = len(df_info_imgs[df_info_imgs.Glaucoma == 1])

UNIQUES_NORMAL_IMGS = len(df_info_imgs.query('Glaucoma == 0 and Pixels_Rows == 1934 and Pixels_Cols == 1956'))
UNIQUES_GLAUCOMA_IMGS = len(df_info_imgs.query('Glaucoma == 1 and Pixels_Rows == 1934 and Pixels_Cols == 1956'))

MOSAICO_NORMAL_IMGS = len(df_info_imgs.query('Glaucoma == 0 and Pixels_Rows == 1200 and Pixels_Cols == 1600'))
MOSAICO_GLAUCOMA_IMGS = len(df_info_imgs.query('Glaucoma == 1 and Pixels_Rows == 1200 and Pixels_Cols == 1600'))


#%% Abrindo arquivo de log e colocando o cabecalho

log_file = open('log_create_dataset_sadalla.csv', 'w', encoding='UTF8')

csv_writer = csv.writer(log_file)

header = ['Imagem_Fonte', 'Imagem_Destino', 'Dimensoes', 'Glaucoma']

csv_writer.writerow(header)


#%% Copiando as imagens para as pastas de destino conforme o tipo e classe da mesma

if NUM_MOVED_IMAGES == 0:
    NUM_MOVED_IMAGES = len(df_info_imgs)

for i in range(NUM_MOVED_IMAGES):
    filename = df_info_imgs.loc[i, 'File'].replace('.txt', '.jpg')
    source_file = SOURCE_FOLDER_ROOT + '/' + df_info_imgs.loc[i, 'Directory'].split('2019/')[1] + '/' + filename
    
    if df_info_imgs.loc[i, 'Pixels_Rows'] == 1934 and df_info_imgs.loc[i, 'Pixels_Cols'] == 1956:
        if df_info_imgs.loc[i, 'Glaucoma'] == 0:
            dest_file = DESTINATION_FOLDER_ROOT + '/' + TYPE_OF_IMAGES[0] + '/' + LABELS[0] + '/' + filename
            row = [source_file, dest_file, '(1934 x 1956)', 0]
        if df_info_imgs.loc[i, 'Glaucoma'] == 1:
            dest_file = DESTINATION_FOLDER_ROOT + '/' + TYPE_OF_IMAGES[0] + '/' + LABELS[1] + '/' + filename
            row = [source_file, dest_file, '(1934 x 1956)', 1]
            
    elif df_info_imgs.loc[i, 'Pixels_Rows'] == 1200 and df_info_imgs.loc[i, 'Pixels_Cols'] == 1600:
        if df_info_imgs.loc[i, 'Glaucoma'] == 0:
            dest_file = DESTINATION_FOLDER_ROOT + '/' + TYPE_OF_IMAGES[1] + '/' + LABELS[0] + '/' + filename
            row = [source_file, dest_file, '(1200 x 1600)', 0]
        if df_info_imgs.loc[i, 'Glaucoma'] == 1:
            dest_file = DESTINATION_FOLDER_ROOT + '/' + TYPE_OF_IMAGES[1] + '/' + LABELS[1] + '/' + filename
            row = [source_file, dest_file, '(1200 x 1600)', 1]
    
    try:
        shutil.copyfile(source_file, dest_file)
        csv_writer.writerow(row)
    except:
        print("Erro ao copiar o arquivo {}".format(source_file))

log_file.close()
    














