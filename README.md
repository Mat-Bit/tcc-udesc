# Trabalho de Conclusão de Curso - Classificador para a doença de Glaucoma

## Descrição

Este projeto tem como objetivo implementar uma Rede Neural Convolucional para a classificação da doença de Glaucoma.

Arquitetura de RNC utilizada será o [Xception](https://arxiv.org/pdf/1610.02357.pdf).

Parte do código utilizado foi baseado em [neste repositório](https://github.com/otenim/Xception-with-Your-Own-Dataset). 


## Funcionamento

1. A Arquitetura Xception terá os pesos iniciais da Rede Neural com o valor do pré-treino do dataset *ImageNet*;
2. Na primeira fase do treinamento, apenas última camada do classificador será treinada;
3. Na segunda fase do treinamento, toda a arquitetura será treinada;
4. Por fim, o modelo treinado será salvo de forma serializada dentro do diretório `<result_root>`.


## Treinando o Modelo

Para treinar o Modelo com a arquitetura Xception, digite o seguinte comanto no terminal:

```bash
$ python xception.py diretorio_com_as_imagens/ classes.txt <result_root> [epochs_pre] [epochs_fine] [batch_size_pre] [batch_size_fine] [lr_pre] [lr_fine]
```

* `<result_root>`: Diretório onde estarão as imagens que serão utilizadas como entrada;
* `[epochs_pre]`: O número de Épocas durante o pré-treino;
* `[epochs_fine]`: O número de Épocas durante o *fine tunning*;
* `[batch_size_pre]`: *Batch size* durante o pré-treino;
* `[batch_size_fine]`: *Batch size* durante o *fine tunning*;
* `[lr_pre]`: *Learning rate* durante o pré-treino;
* `[lr_fine]`: *Learning rate* durante o *fine tunning*.

Obs: **[]** indica um argumento **opcional**. **<>** indica um argumento **obrigatório**.


## Realizando a predição de uma imagem

```bash
$ python classifier.py <model> <classes> <image> [top_n]
```

* `<model>`: Caminho até o modelo treinado serializado;
* `<classes>`: Caminho com o arquivo que informa as Classes que poderãp ser ŕeditas;
* `<image>`: Caminho da imagem que será predita;
* `[top_n]`: Mostrar os *n* melhores resultados.

Obs: **[]** indica um argumento **opcional**. **<>** indica um argumento **obrigatório**.