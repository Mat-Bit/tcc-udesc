import math
import os
import argparse
import matplotlib
import imghdr
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve


matplotlib.use('Agg')
current_directory = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('name')
parser.add_argument('dataset_root')
parser.add_argument('result_root')
parser.add_argument('--epochs_pre', type=int, default=10)
parser.add_argument('--epochs_fine', type=int, default=20)
parser.add_argument('--batch_size_pre', type=int, default=32)
parser.add_argument('--batch_size_fine', type=int, default=32)
parser.add_argument('--lr_pre', type=float, default=1e-3)
parser.add_argument('--lr_fine', type=float, default=5e-4)
parser.add_argument('--test_split', type=float, default=0.1)
parser.add_argument('--val_split', type=float, default=0.2)
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


def generate_from_paths(
        input_paths, batch_size, input_size=(299, 299)):
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
            yield inputs


def plot_metrics(history, name_path, title):
    metrics = ['accuracy', 'loss', 'precision', 'recall']
    plt.figure(figsize=(8,8))
    plt.title(title)

    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,2,n+1)
        plt.plot(history.epoch, history.history[metric], label='Train')
        plt.plot(history.epoch, history.history['val_'+ metric], linestyle="--", label='Val')
        if n >= len(metrics) // 2:
            plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, 4])
        else:
            plt.ylim([0.45,1.1])

        plt.legend()
    plt.savefig(name_path + 'fit_metrics.png')
    plt.clf()


def plot_roc(name, labels, predictions, **kwargs):
    fp, tp, _ = roc_curve(labels, predictions)

    plt.plot(fp, tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([-0.05,0.40])
    plt.ylim([0.6,1.05])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')


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

    with open(f'{result_path_name}/parameters.txt', 'w') as file:
        file.write(f"\tTeste: {args.name}\n\n")

        file.write(f"Diretorio Dataset: {args.dataset_root};\n")
        file.write(f"Diretorio dos Resultados: {result_path_name};\n\n")

        file.write(f"Dropout Regularization: {args.dropout};\n\n")

        file.write(f"\tPre-treino (apenas ultima camada):\n")
        file.write(f"Epocas: {args.epochs_pre};\n")
        file.write(f"Batch Size: {args.batch_size_pre};\n")
        file.write(f"Learning Rate: {args.lr_pre};\n\n")

        file.write(f"\tFine Tunning de toda Arquitetura:\n")
        file.write(f"Epocas: {args.epochs_fine};\n")
        file.write(f"Batch Size: {args.batch_size_fine};\n")
        file.write(f"Learning Rate: {args.lr_fine};\n")

    classes = os.listdir(args.dataset_root)
    num_classes = len(classes)

    # make input_paths and labels
    input_paths, labels = [], []
    classes_dir = [class_path for class_path in os.listdir(args.dataset_root) if os.path.isdir(os.path.join(args.dataset_root, class_path))]
    for class_name in classes_dir:
        class_root = os.path.join(args.dataset_root, class_name)
        class_id = classes.index(class_name)
        for path in os.listdir(class_root):
            path = os.path.join(class_root, path)
            if imghdr.what(path) is None:
                # this is not an image file
                continue
            input_paths.append(path)
            labels.append(class_id)

    # convert to one-hot-vector format
    labels = to_categorical(labels, num_classes=num_classes)

    # convert to numpy array
    input_paths = np.array(input_paths)

    # split dataset for training and test
    train_input_paths, test_input_paths, train_labels, test_labels = train_test_split(input_paths, labels, test_size=args.test_split, shuffle=True, random_state=47)

    # split the training dataset for training and validation
    train_input_paths, val_input_paths, train_labels, val_labels = train_test_split(train_input_paths, train_labels, test_size=args.val_split, shuffle=True, random_state=26)

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

    # ====================================================
    # Train only the top classifier
    # ====================================================
    # freeze the body layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile model
    model.compile(
        loss=binary_crossentropy(from_logits=True),
        optimizer=Adam(learning_rate=args.lr_pre, episilon=0.1),
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


    # train
    hist_pre = model.fit(
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

    # ====================================================
    # Train the whole model
    # ====================================================
    # set all the layers to be trainable
    for layer in model.layers:
        layer.trainable = True

    # recompile
    model.compile(
        loss=binary_crossentropy(from_logits=True),
        optimizer=Adam(learning_rate=args.lr_fine, epsilon=0.1),
        metrics=METRICS
    )

    # train
    hist_fine = model.fit(
        train_dataset,
        steps_per_epoch = math.ceil(
            len(train_input_paths) / args.batch_size_fine),
        epochs = args.epochs_fine,
        validation_data = validation_dataset,
        validation_steps=math.ceil(
            len(val_input_paths) / args.batch_size_fine),
        verbose=1,
        callbacks=custom_callbacks
    )

    # ====================================================
    # Create & save result graphs
    # ====================================================

    # Plot the training metrics
    plot_path = os.path.join(result_path_name, 'pre_training_')
    plot_metrics(hist_pre, plot_path, 'Pré Treino')

    plot_path = os.path.join(result_path_name, 'fine_tunning_')
    plot_metrics(hist_fine, plot_path, 'Fine Tunning')


    test_dataset = generate_from_paths_and_labels(
        input_paths=test_input_paths,
        labels=test_labels,
        batch_size=args.batch_size_pre
    )

    # evaluate 
    test_predictions_baseline = model.evaluate(test_dataset, batch_size=args.batch_size_pre)

    print("Test accuracy on test set: %0.4f" % (test_predictions_baseline[1]))

    # Plot the ROC curve
    plot_roc("Test", test_labels, test_predictions_baseline)
    plt.savefig(os.path.join(args.result_root, 'auroc.png'))
    plt.clf()


if __name__ == '__main__':
    args = parser.parse_args()

    print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")

    main(args)