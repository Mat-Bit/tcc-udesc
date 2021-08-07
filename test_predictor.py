#%% Importando as libs

import numpy as np
import math
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


#%% Definindo o camino do result e as variaveis com nomes dos arquivos

RESULT_PATH = '../resultados/Raw-03_Final_0_2222_val_split_raw_16B_0_2_lr_05_dp/'

result_path_name = os.path.expanduser(RESULT_PATH)
model_file = os.path.join(result_path_name, 'model_trained.h5')
dataset_desc_file = os.path.join(result_path_name, 'dataset_desc.csv')
training_log_file = os.path.join(result_path_name, 'training_log.csv')
params_file = os.path.join(result_path_name, 'parameters.txt')
    

#%% Criando os dataframes de log e desc_dataset

df_dataset = pd.read_csv(dataset_desc_file)

# Get the row of 'df_dataset' where the 'accuracy' or 'val_accuracy' is biggest value
df_log_train = pd.read_csv(training_log_file)


#%% Pegar a linha que contem os valores maximos de 'accuracy'

index_best_val_acc = df_log_train['accuracy'].argmax()

row_best_val_acc = df_log_train.loc[index_best_val_acc, :]

print("Best metrics in Training Set")
for met in row_best_val_acc.index:
    if met[0:3] != 'val':
        print("{}: {:.3f};".format(met, row_best_val_acc[met]))

    
#%% Pegar a linha que contem os valores maximos de 'val_accuracy'

print()
index_best_val_acc = df_log_train['val_accuracy'].argmax()

row_best_val_acc = df_log_train.loc[index_best_val_acc, :]

print("Best metrics in Validation Set")
for met in row_best_val_acc.index:
    if met[0:3] == 'val':
        print("{}: {:.3f};".format(met, row_best_val_acc[met]))


#%% Plotando o balanceamento das bases

df_train = df_dataset[df_dataset.Subset == 'Train']
df_train = df_train.reset_index()

df_val = df_dataset[df_dataset.Subset == 'Validation']
df_val = df_val.reset_index()

df_test = df_dataset[df_dataset.Subset == 'Test']
df_test = df_test.reset_index()

count_train_normal = 0
count_train_glaucoma = 0

count_val_normal = 0
count_val_glaucoma = 0

count_test_normal = 0
count_test_glaucoma = 0

# Contanto as imagens 'Normal' e 'Glaucoma' no Subset 'Train'
for i in range(df_train.shape[0]):
    label = df_train.loc[i, 'Image'].split('/')[3]
    if label == 'Normal':
        count_train_normal += 1
    elif label == 'Glaucoma':
        count_train_glaucoma += 1

# Contando as imagens 'Normal' e 'Glaucoma' no Subset 'Validation'
for i in range(df_val.shape[0]):
    label = df_val.loc[i, 'Image'].split('/')[3]
    if label == 'Normal':
        count_val_normal += 1
    elif label == 'Glaucoma':
        count_val_glaucoma += 1

# Contando as imagens 'Normal' e 'Glaucoma' no Subset 'Test'
for i in range(df_test.shape[0]):
    label = df_test.loc[i, 'Image'].split('/')[3]
    if label == 'Normal':
        count_test_normal += 1
    elif label == 'Glaucoma':
        count_test_glaucoma += 1

# Plotando a quantidade de imagens de cada Subset em um gráfico de barras usando plt
labels_balance = ['Treinamento', 'Validação', 'Teste']

count_train_normal = np.round(count_train_normal / df_train.shape[0], 2)
count_train_glaucoma = np.round(count_train_glaucoma / df_train.shape[0], 2)
count_val_glaucoma = np.round(count_val_glaucoma / df_val.shape[0], 2)
count_val_normal = np.round(count_val_normal / df_val.shape[0], 2)
count_test_glaucoma = np.round(count_test_glaucoma / df_test.shape[0], 2)
count_test_normal = np.round(count_test_normal / df_test.shape[0], 2)

count_normal = [count_train_normal, count_val_normal, count_test_normal]
count_glaucoma = [count_train_glaucoma, count_val_glaucoma, count_test_glaucoma]

x = np.arange(len(labels_balance))  # the label locations
width = 0.30  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, count_normal, width, label='Normal')
rects2 = ax.bar(x + width/2, count_glaucoma, width, label='Glaucoma')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Porcetagem de Imagens')
ax.set_title('Porcentagem de Imagens de Cada Subconjunto')
ax.set_xticks(x)
ax.set_xticklabels(labels_balance)
plt.ylim(0, 0.7)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()
plt.savefig(os.path.join(result_path_name, 'balance_plot.png'))
plt.show()
plt.clf()


#%% Criando o 'input_paths' e as 'labels' do subconjunto de 'Test'

input_paths = df_test.loc[:, 'Image'].tolist()

labels = df_test.loc[:, 'Label'].tolist()
np_labels = []

for label in labels:
    if len(df_dataset.loc[0, 'Label']) > 8:
        l1 = int(label[1])
        l2 = int(label[4])
        l3 = int(label[7])
        label = [l1, l2, l3]
        np_labels.append(np.array(label))
    else:
        l1 = int(label[1])
        l2 = int(label[4])
        label = [l1, l2]
        np_labels.append(np.array(label))

test_labels = np.array(np_labels)

test_input_paths = np.array(input_paths)


#%% Gerando o Dataset de Teste

def generate_from_paths(
        input_paths, batch_size=32, input_size=(299, 299)):
    num_samples = len(input_paths)
    while 1:
        for i in range(0, num_samples, batch_size):
            inputs = list(map(
                lambda x: image.load_img(x, target_size=input_size),
                input_paths[i:i+batch_size]
            ))
            inputs = np.array(list(map(
                lambda x: image.img_to_array(x),
                inputs
            )))
            inputs = preprocess_input(inputs)
            yield (inputs)


test_dataset = generate_from_paths(
    input_paths=test_input_paths,
    batch_size=16
)


#%% Carregando o modelo e fazendo a predicao no conjunto de teste

model = load_model(model_file)

test_predictions = model.predict(test_dataset, 
                                 steps=math.ceil(len(test_input_paths) / 16),
                                 verbose=1)


#%% Gerando a curva ROC

fp, tp, _ = roc_curve(test_labels.ravel(), test_predictions.ravel())

auroc_value = auc(fp, tp)

lw = 2
plt.plot(fp, tp, color='darkorange', 
         lw=lw, label='AUROC = %0.3f' % auroc_value)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1 - Especificidade')
plt.ylabel('Sensibilidade')
plt.legend(loc="lower right")
plt.title('Curva ROC - Dataset com pré-processamento')

name_auroc_file = os.path.join(result_path_name, 'auroc_plot.png')
plt.savefig(name_auroc_file)
plt.show()








