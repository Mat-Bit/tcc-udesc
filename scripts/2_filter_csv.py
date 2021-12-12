"""
Created on Mon Apr 13 19:22:53 2020

@author: matbit
"""
#%% Importando as bibliotecas

import pandas as pd
import os


#%% Lendo o arquivo CSV "dados.csv" em um DataFrame

df_glaucoma = pd.read_csv("Glaucoma/Glaucoma/dados.csv")

# Removing the duplicates rows
df_glaucoma = pd.DataFrame(df_glaucoma['concat'].unique(), columns=["concat"])

n_img_glaucoma = len(df_glaucoma)

# Replacing 'DICOM' for 'TXT'
for i in range(len(df_glaucoma)):
    new_row = df_glaucoma.loc[i, 'concat']
    new_row = new_row.replace('DICOM', 'Glaucoma/Glaucoma/TXT')
    new_row = new_row.replace('.dcm', '.txt')
    df_glaucoma.loc[i, 'concat'] = new_row


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


#%% Definicao do diretorio base com os arquivos txt

text_path = "Glaucoma/Glaucoma/TXT-STORAGE/2019/"

# Aplicacao da funcao de percorrer os arquivos salvando no dicionario
d_paths, count_files = dir_walker(text_path)


#%% Criando uma lista de dicionarios com informacoes uteis
# do txt das informacoes das imagens
    
# Creating a dict for the file in text_path
l_master = []
count_glauc = 0

for n, d in enumerate(d_paths.keys()):
    for f in d_paths[d]:
        # Dict with the colmns that goint to into CSV file
        d_master = {}
        
        txt_file = os.path.join(d, f)
        with open(txt_file, "r") as arq_in:
            lines = arq_in.readlines()
            shift = 0
            if lines[0][0:17] == "Dataset.file_meta":
                shift = 10
            
            d_master["Pat_id"] = int(lines[34+shift].split("\'")[1])
            
            if lines[35+shift].split(",")[1][:5] != ' 0021':
                shift -= 1
            
            d_master["Pat_birth_year"] = int(lines[36+shift].split("\'")[2])
            d_master["Pat_years"] = 2019 - int(lines[36+shift].split("\'")[2])
            d_master["Pat_sex"] = lines[37+shift].split("\'")[2]
            img_date = lines[4+shift].split("\'")[1][:4] + '-' + lines[4+shift].split("\'")[1][4:6] + '-' + lines[4+shift].split("\'")[1][6:8]
            d_master["Img_Date"] = img_date
            d_master["Directory"] = d
            d_master["File"] = f
            if int(lines[51+shift].split(":")[1][:5]) > 100:
                d_master["Pixels_Rows"] = int(lines[50+shift].split(":")[1][:5])
                d_master["Pixels_Cols"] = int(lines[51+shift].split(":")[1][:5])
            else:
                d_master["Pixels_Rows"] = int(lines[49+shift].split(":")[1][:5])
                d_master["Pixels_Cols"] = int(lines[50+shift].split(":")[1][:5])
            
            if txt_file in df_glaucoma["concat"].values:
                d_master["Glaucoma"] = 1
                count_glauc += 1
                # print("Paciente {} com Glaucoma!\n".format(d_master["Pat_id"]))
            else:
                d_master["Glaucoma"] = 0
            
            l_master.append(d_master)
                

#%% Transformando o 'l_master' em um dataframe e extrair os arquivos uteis
            
print(count_glauc, "Imagens com glaucoma.")
print()
df_info = pd.DataFrame(l_master)

df_util = df_info[df_info.Pixels_Rows == 1934]

total_util = len(df_util)

util_glaucoma = len(df_util[df_util.Glaucoma == 1])

util_normal = total_util - util_glaucoma

print(util_glaucoma, "Imagens uteis com glaucoma.")
print(util_normal, "Imagens uteis normais.")


#%% Salvando um csv com informacoes de todas imagens e somente as uteis

df_info.to_csv("dados_info.csv", index=False)

df_util.to_csv("dados_info_uteis.csv", index=False)


