"""
Created on Wed Aug 26 11:44:26 2020

@author: mateus-bit
"""

#%% Importando as libs

import shutil
import os
import pandas as pd
from random import randint


#%% Criando os diretorios do dataset para 'Glaucoma' ou 'Saudavel'

path_glaucoma = "dataset/Glaucoma/"

path_normal = "dataset/Normal/"

# Criando diretorio para as imagens com Glaucoma
try:
    os.makedirs(os.path.dirname(path_glaucoma))
except FileExistsError:
    print("O diretorio '{}' ja foi criado.\n".format(path_glaucoma))

# Criando diretorio para as imagens Normais
try:
    os.makedirs(os.path.dirname(path_normal))
except FileExistsError:
    print("O diretorio '{}' ja foi criado.\n".format(path_normal))


#%% Renomeando a extensao do deiretorio e arquivo de TXT para JPGE ou PNG

df_dados = pd.read_csv('csv/dados_info_uteis.csv')

colunas = df_dados.columns

JPEG = True

for i in range(len(df_dados)):
    diretorio = df_dados.loc[i, "Directory"]
    arquivo = df_dados.loc[i, "File"]

    if JPEG:
        diretorio = diretorio.replace("TXT", "JPG")
        arquivo = arquivo.replace(".txt", ".jpg")
    else:
        diretorio = diretorio.replace("TXT", "PNG")
        arquivo = arquivo.replace(".txt", ".png")

    df_dados.loc[i, "Directory"] = diretorio
    df_dados.loc[i, "File"] = arquivo


#%% Copiando as imagens para cada respectivo diretorio, de acordo se tem ou nao Glaucoma

cont_glaucoma = 0
cont_normal = 0

random_range = len(df_dados)
sorteados = []

while cont_glaucoma < 100 or cont_normal < 100:
    i = randint(0, random_range-1)
    
    if i in sorteados:
        continue

    tem_glaucoma = df_dados.loc[i, 'Glaucoma']

    if tem_glaucoma == 0:
        if cont_normal == 100:
            continue
        imagem_fonte = df_dados.loc[i, "Directory"]
        imagem_fonte += '/'
        imagem_fonte += df_dados.loc[i, "File"]

        imagem_destino = path_normal
        imagem_destino += df_dados.loc[i, "File"]

        shutil.copyfile(imagem_fonte, imagem_destino)
        cont_normal += 1

    elif tem_glaucoma == 1:
        if cont_glaucoma == 100:
            continue
        imagem_fonte = df_dados.loc[i, "Directory"]
        imagem_fonte += '/'
        imagem_fonte += df_dados.loc[i, "File"]

        imagem_destino = path_glaucoma
        imagem_destino += df_dados.loc[i, "File"]

        shutil.copyfile(imagem_fonte, imagem_destino)
        cont_glaucoma += 1
    
    sorteados.append(i)
    

#%% Escolher algumas fotos aleatorias para saber se estao classificadas corretamente
img = {}

img['G'] = ['1.2.392.200106.1651.4.2.228164113206078245.1547561743.89.jpg',
            '1.2.392.200106.1651.4.2.228164113206078245.1549991199.78.jpg',
            '1.2.392.200106.1651.4.2.228164113206078245.1553086442.19.jpg']

img['N'] = ['1.2.392.200106.1651.4.2.228164113206078245.1547729231.31.jpg',
            '1.2.392.200106.1651.4.2.228164113206078245.1549571885.113.jpg',
            '1.2.392.200106.1651.4.2.228164113206078245.1553269011.40.jpg']

cont = 0

for i in range(len(df_dados)):
    if cont == 6: 
        break
    
    df_img = df_dados.loc[i, "File"]
    df_class = df_dados.loc[i, "Glaucoma"]
    
    if df_img in img['G'] and df_class == 1:
        print(df_img, ":  OK (Glaugoma)")
        cont += 1
    elif df_img in img['N'] and df_class == 0:
        print(df_img, ":  OK (Normal)")
        cont += 1
        




