import tensorflow as tf
import math
import os
import argparse
import matplotlib
import imghdr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


matplotlib.use('Agg')
current_directory = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('name')
parser.add_argument('dataset_root')
parser.add_argument('result_root')
parser.add_argument('--epochs_pre', type=int, default=40)
parser.add_argument('--batch_size_pre', type=int, default=16)
parser.add_argument('--lr_pre', type=float, default=5e-4)
parser.add_argument('--test_split', type=float, default=0.1)
parser.add_argument('--val_split', type=float, default=0.25)
parser.add_argument('--dropout', type=float, default=0.5)


METRICS = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc')
]


def generate_from_paths_and_labels(
        input_paths, labels, batch_size, input_size=(299, 299)):
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
            yield (inputs, labels[i:i+batch_size])


def plot_metrics(history, path):
    metrics = ['accuracy', 'loss', 'precision', 'recall']
    plt.figure(figsize=(8,8))
    if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
        plt.title("Métricas de Treinamento e Validação Usando GPU")
    else:
        plt.title("Métricas de Treinamento e Validação Usando CPU")

    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,2,n+1)
        plt.plot(history['epoch'], history[metric], label='Train')
        plt.plot(history['epoch'], history['val_'+ metric], linestyle="--", label='Val')
        if n >= len(metrics) // 2:
            plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([-0.05, 1])
        else:
            plt.ylim([0.45,1.05])

        plt.legend()
    
    filename = os.path.join(path, 'fit_metrics.png')
    plt.savefig(filename)
    plt.clf()
    plt.close()


def main(args):

    # ====================================================
    # Preparation
    # ====================================================
    # parameters
    epochs = args.epochs_pre + args.epochs_fine
    args.dataset_root = os.path.expanduser(args.dataset_root)
    args.result_root = os.path.expanduser(args.result_root)

    # create a root directory of results 
    if os.path.exists(args.result_root) is False:
        os.makedirs(args.result_root)
    
    # create a directory where results will be stored
    result_path_name = os.path.join(args.result_root, args.name)
    if os.path.exists(result_path_name) is False:
        os.makedirs(result_path_name)


    classes_dir = [class_path for class_path in os.listdir(args.dataset_root) if os.path.isdir(os.path.join(args.dataset_root, class_path))]
    num_classes = len(classes_dir)

    # make input_paths and labels
    input_paths, labels = [], []
    for class_name in classes_dir:
        class_root = os.path.join(args.dataset_root, class_name)
        class_id = classes_dir.index(class_name)
        for path in os.listdir(class_root):
            path = os.path.join(class_root, path)
            if imghdr.what(path) is None:
                # this is not an image file
                continue
            input_paths.append(path)
            labels.append(class_id)

    with open(f'{result_path_name}/parameters.txt', 'w') as file:
        file.write(f"\tTeste: {args.name}\n\n")

        file.write(f"Classes: {classes_dir};\n")
        file.write(f"Diretorio Dataset: {args.dataset_root};\n")
        file.write(f"Diretorio dos Resultados: {result_path_name};\n")
        file.write(f"Tamanho do Dataset: {len(input_paths)} imagens;\n\n")

        file.write(f"Dropout Regularization: {args.dropout};\n\n")

        file.write(f"\tPre-treino (apenas ultima camada):\n")
        file.write(f"Epocas: {args.epochs_pre};\n")
        file.write(f"Batch Size: {args.batch_size_pre};\n")
        file.write(f"Learning Rate: {args.lr_pre};\n\n")

    # convert to one-hot-vector format
    labels = to_categorical(labels, num_classes=num_classes)

    # convert to numpy array
    input_paths = np.array(input_paths)

    # split dataset for training and test
    train_input_paths, test_input_paths, train_labels, test_labels = train_test_split(input_paths, labels, test_size=args.test_split, shuffle=True, random_state=42)

    # split the training dataset for training and validation
    train_input_paths, val_input_paths, train_labels, val_labels = train_test_split(train_input_paths, train_labels, test_size=args.val_split, shuffle=True, random_state=24)

    print("Training on %d images and labels" % (len(train_input_paths)))
    print("Validation on %d images and labels" % (len(val_input_paths)))
    print("Test on %d images and labels" % (len(test_input_paths)))

    df = pd.DataFrame(columns=['Image', 'Subset', 'Label'])

    for i in range(len(train_input_paths)):
        df.loc[i] = [train_input_paths[i], 'Train', train_labels[i]]

    n = len(df)
    for i in range(len(val_input_paths)):
        df.loc[n+i] = [val_input_paths[i], 'Validation', val_labels[i]]

    n = len(df)
    for i in range(len(test_input_paths)):
        df.loc[n+i] = [test_input_paths[i], 'Test', test_labels[i]]

    dataset_desc = os.path.join(result_path_name, 'dataset_desc.csv')
    df.to_csv(dataset_desc, index=False)

    # Create a CSV logger file
    csv_logger_file = os.path.join(result_path_name, 'training_log.csv')
    csv_logger = CSVLogger(csv_logger_file, append=True)

    # Create a callback to save the model weights
    checkpoint_file = os.path.join(result_path_name, 'model_trained.h5')
    checkpoint = ModelCheckpoint(checkpoint_file, 
                                 monitor='val_accuracy', 
                                 save_weights_only=False, 
                                 verbose=1, 
                                 save_best_only=True, 
                                 mode='max')  

    custom_callbacks = [csv_logger, checkpoint]

    # ====================================================
    # Build a custom Xception
    # ====================================================
    # instantiate pre-trained Xception model
    # the default input shape is (299, 299, 3)
    # NOTE: the top classifier is not included
    base_model = Xception(
        include_top=False,
        weights='imagenet',
        input_shape=(299, 299, 3))

    # create a custom top classifier
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(args.dropout, input_shape=(1024,))(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.inputs, outputs=predictions)

    # freeze the body layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile model
    model.compile(
        loss=binary_crossentropy,
        optimizer=Adam(learning_rate=args.lr_pre, epsilon=0.1),
        metrics=METRICS
    )

    train_dataset = generate_from_paths_and_labels(
        input_paths=train_input_paths,
        labels=train_labels,
        batch_size=args.batch_size_pre
    )
    validation_dataset = generate_from_paths_and_labels(
        input_paths=val_input_paths,
        labels=val_labels,
        batch_size=args.batch_size_pre
    )

    # start count time
    start_time = time.time()

    # train
    hist = model.fit(
        train_dataset,
        steps_per_epoch = math.ceil(
            len(train_input_paths) / args.batch_size_pre),
        epochs = args.epochs_pre,
        validation_data = validation_dataset,
        validation_steps=math.ceil(
            len(val_input_paths) / args.batch_size_pre),
        verbose=1,
        callbacks=custom_callbacks
    )

    # End time
    end_time = time.time()

    # Write the training time into the log file 'parameters.txt'
    with open(os.path.join(result_path_name, 'parameters.txt'), 'a') as f:
        f.write('\nTempo de treinamento: {:.2f}s.\n'.format(end_time - start_time))

    # ====================================================
    # Create & save result graphs
    # ====================================================

    # concatinate plot data
    epochs = hist.history['epoch']
    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    precision = hist.history['precision']
    val_precision = hist.history['val_precision']
    recall = hist.history['recall']
    val_recall = hist.history['val_recall']
    auc = hist.history['auc']
    val_auc = hist.history['val_auc']

    # group all training history
    history = {
        'epoch': epochs,
        'accuracy': acc,
        'val_accuracy': val_acc,
        'loss': loss,
        'val_loss': val_loss,
        'precision': precision,
        'val_precision': val_precision,
        'recall': recall,
        'val_recall': val_recall,
        'auc': auc,
        'val_auc': val_auc
    }

    # Plot the training metrics
    plot_metrics(history, result_path_name)

    test_dataset = generate_from_paths_and_labels(
        input_paths=test_input_paths,
        labels=test_labels,
        batch_size=args.batch_size_pre
    )

    # Evaluate the model on the test dataset
    score = model.evaluate(
        test_dataset,
        steps=math.ceil(len(test_input_paths) / args.batch_size_pre),
        verbose=1
    )

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Write the test accuracy and loss into the log file 'parameters.txt'
    with open(os.path.join(result_path_name, 'parameters.txt'), 'a') as f:
        f.write('\nTest loss: {:.4f}.\n'.format(score[0]))
        f.write('Test accuracy: {:.4f}.\n'.format(score[1]))
        f.write('Test precision: {:.4f}.\n'.format(score[2]))
        f.write('Test recall: {:.4f}.\n'.format(score[3]))
        f.write('Test auc: {:.4f}.\n'.format(score[4]))


if __name__ == '__main__':
    args = parser.parse_args()

    print("GPUs: ", len(tf.config.experimental.list_physical_devices('GPU')))

    main(args)