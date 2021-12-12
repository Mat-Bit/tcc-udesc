"""
Created on Wed Aug 26 11:44:26 2020

@author: mateus-bit
"""

#%% Importando as libs

import pandas as pd


#%% Lendo o CSV do log de geracao do dataset

df_dados = pd.read_csv('csv/log_create_dataset_sadalla.csv')

colunas = df_dados.columns

num_linhas = len(df_dados)


#%% Fazendo split dos dataset de FO_Unicas e Mosaico

df_fo_unicas = df_dados.loc[df_dados['Dimensoes'] == '(1934 x 1956)', ['Imagem_Destino', 'Glaucoma']]

df_mosaico = df_dados.loc[df_dados['Dimensoes'] == '(1200 x 1600)', ['Imagem_Destino', 'Glaucoma']]


#%% Renomeando as colunas e os valores da coluna classe (Label)

df_fo_unicas = df_fo_unicas.replace({0: 'Normal', 1: 'Glaucoma'})
df_mosaico = df_mosaico.replace({0: 'Normal', 1: 'Glaucoma'})

df_fo_unicas = df_fo_unicas.rename(columns = {'Glaucoma': 'Label', 'Imagem_Destino': 'Imagem'})
df_mosaico = df_mosaico.rename(columns = {'Glaucoma': 'Label', 'Imagem_Destino': 'Imagem'})



#%% Salvando os novos Dataframes em CSV

df_fo_unicas.to_csv('fo_unicas.csv', index=False)

df_mosaico.to_csv('mosaico.csv', index=False)



        
        
        
        
        
