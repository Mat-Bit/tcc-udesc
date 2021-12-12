# Trabalho de Conclusão de Curso - Classificador para a doença de Glaucoma

## Descrição

Este projeto tem como objetivo implementar uma Rede Neural Convolucional para a classificação da doença de Glaucoma.

Arquitetura de RNC utilizada será o [Xception](https://arxiv.org/pdf/1610.02357.pdf).

Parte do código utilizado foi baseado em [neste repositório](https://github.com/otenim/Xception-with-Your-Own-Dataset). 


## Funcionamento

1. A Arquitetura Xception terá os pesos iniciais da Rede Neural com o valor do pré-treino do dataset [ImageNet](https://image-net.org/);
2. Na primeira fase do treinamento, apenas última camada do classificador será treinada;
3. Na segunda fase do treinamento, toda a arquitetura será treinada;
4. Por fim, o modelo treinado será salvo de forma serializada dentro do diretório `<result_root>`.


## Treinando o Modelo

Para treinar o Modelo com a arquitetura Xception, digite o seguinte comanto no terminal:

```bash
$ python xception.py <train_name> <dataset_root_path> <result_root> [--batch_size N] [--dropout N] [--epochs_pre N] [--epochs_fine N] [--lr_pre N] [--lr_fine N] [--test_split N] [--val_split N]
```

* `<train_name>`: Nome do diretório que será salvo os arquivos de saída do treinamento;
* `<dataset_root_path>`: Nome do diretório onde estão as imagens que serão utilizadas para o treinamento / teste, onde o nome dos diretórios dentro do *<dataset_root_path>*, serão as imagens de cada classe (Normal / Glaucoma);
* `<result_root>`: Diretório onde estarão as imagens que serão utilizadas como entrada;
* `[--batch_size N]`: Tamanho (int) N do *batch* (padrão=16);
* `[--dropout N]`: Taxa de *dropout* entre 0 e 1 (padrão = 0.5);
* `[--epochs_pre N]`: Número inteiro de épocas para o treino somente do topo da arquitetura (padrão = 10);
* `[--epochs_fine N]`: Número inteiro de épocas para o treino de toda a arquitetura - *Fine Tunning* (padrão = 50);
* `[--lr_pre N]`: Taxa (float) de aprendizado durante o treinamento somente do topo da arquitetura (padrão = 2^-4);
* `[--lr_fine N]`: Taxa (float) de aprendizado durante o treinamentode toda a arquitetura (padrão = 2^-4);
* `[--test_split N]`: Taxa (float) da base que será feito a divisão para o subconjunto de **Teste** (Padrão = 0.1);
* `[--val_split N]`: Taxa (float) da base que será feito a divisão para o subconjunto de **Validação** (Padrão = 0.22);


Obs: **[]** indica um argumento **opcional**. **<>** indica um argumento **obrigatório**.


## Realizando a avaliação do modelo

```bash
$ python test_predictor.py
```

Obs: O Caminho do diretório do modelo treinado deve ser informado no variável `RESULT_PATH`