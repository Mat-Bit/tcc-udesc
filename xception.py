import math
import os
import argparse
import matplotlib
import imghdr
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import CSVLogger
# from tensorflow.keras.metrics import AUC
from sklearn.model_selection import train_test_split


matplotlib.use('Agg')
current_directory = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('name')
parser.add_argument('dataset_root')
parser.add_argument('classes')
parser.add_argument('result_root')
parser.add_argument('--epochs_pre', type=int, default=10)
parser.add_argument('--epochs_fine', type=int, default=50)
parser.add_argument('--batch_size_pre', type=int, default=16)
parser.add_argument('--batch_size_fine', type=int, default=32)
parser.add_argument('--lr_pre', type=float, default=1e-3)
parser.add_argument('--lr_fine', type=float, default=2e-4)
parser.add_argument('--split', type=float, default=0.8)


def generate_from_paths_and_labels(
        input_paths, labels, batch_size, input_size=(299, 299)):
    num_samples = len(input_paths)
    while 1:
        perm = np.random.permutation(num_samples)
        input_paths = input_paths[perm]
        labels = labels[perm]
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


def main(args):

    # ====================================================
    # Preparation
    # ====================================================
    # parameters
    epochs = args.epochs_pre + args.epochs_fine
    args.dataset_root = os.path.expanduser(args.dataset_root)
    args.result_root = os.path.expanduser(args.result_root)
    args.classes = os.path.expanduser(args.classes)

    # load class names
    with open(args.classes, 'r') as f:
        classes = f.readlines()
        classes = list(map(lambda x: x.strip(), classes))
    num_classes = len(classes)

    # make input_paths and labels
    input_paths, labels = [], []
    for class_name in os.listdir(args.dataset_root):
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

    # shuffle dataset
    perm = np.random.permutation(len(input_paths))
    labels = labels[perm]
    input_paths = input_paths[perm]

    print("Input_paths:", len(input_paths))

    # split dataset for training and validation
    train_input_paths, val_input_paths, train_labels, val_labels = train_test_split(input_paths, labels, train_size=args.split)
    
    print("Training on %d images and labels" % (len(train_input_paths)))
    print("Validation on %d images and labels" % (len(val_input_paths)))

    # create a directory where results will be saved (if necessary)
    if os.path.exists(args.result_root) is False:
        os.makedirs(args.result_root)
    
    # Create a CSV logger file
    csv_logger = CSVLogger(f'{args.name}_log.csv', append=True)

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
        loss=binary_crossentropy,
        optimizer=Adam(lr=args.lr_pre),
        metrics=['accuracy']
    )

    # Generate a train and validation sets for pre training
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
        callbacks=[csv_logger]
    )
    model.save(os.path.join(args.result_root, f'model_{args.name}_pre_training.h5'))

    # ====================================================
    # Train the whole model
    # ====================================================
    # set all the layers to be trainable
    for layer in model.layers:
        layer.trainable = True

    # recompile
    model.compile(
        loss=binary_crossentropy,
        optimizer=Adam(lr=args.lr_fine),
        metrics=['accuracy'])

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
        callbacks=[csv_logger],
    )
    model.save(os.path.join(args.result_root, f'model_{args.name}_fine_tunning.h5'))

    # ====================================================
    # Create & save result graphs
    # ====================================================
    # concatinate plot data
    acc = hist_pre.history['accuracy']
    val_acc = hist_pre.history['val_accuracy']
    loss = hist_pre.history['loss']
    val_loss = hist_pre.history['val_loss']
    acc.extend(hist_fine.history['accuracy'])
    val_acc.extend(hist_fine.history['val_accuracy'])
    loss.extend(hist_fine.history['loss'])
    val_loss.extend(hist_fine.history['val_loss'])

    # save graph image
    plt.plot(range(epochs), acc, marker='.', label='accuracy')
    plt.plot(range(epochs), val_acc, marker='.', label='val_accuracy')
    plt.legend(loc='best')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig(os.path.join(args.result_root, 'accuracy.png'))
    plt.clf()

    plt.plot(range(epochs), loss, marker='.', label='loss')
    plt.plot(range(epochs), val_loss, marker='.', label='val_loss')
    plt.legend(loc='best')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(os.path.join(args.result_root, 'loss.png'))
    plt.clf()

    # save plot data as pickle file
    plot = {
        'accuracy': acc,
        'val_accuracy': val_acc,
        'loss': loss,
        'val_loss': val_loss,
    }
    with open(os.path.join(args.result_root, 'plot.dump'), 'wb') as f:
        pkl.dump(plot, f)


if __name__ == '__main__':
    args = parser.parse_args()

    print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")

    with open(f'{args.result_root}/{args.name}', 'w') as file:
        file.write(f"\tTeste: {args.name}\n\n")

        file.write(f"Diretorio Dataset: {args.dataset_root};\n")
        file.write(f"Arquivo com as Classes: {args.classes};\n\n")

        file.write(f"\tPre-treino (apenas ultima camada):\n")
        file.write(f"Epocas: {args.epoch_pre};\n")
        file.write(f"Batch Size: {args.batch_size_pre};\n")
        file.write(f"Learning Rate: {args.lr_pre};\n")

        file.write(f"\tFine Tunning de toda Arquitetura:\n")
        file.write(f"Epocas: {args.epochs_fine};\n")
        file.write(f"Batch Size: {args.batch_size_fine};\n")
        file.write(f"Learning Rate: {args.lr_fine};\n")

    main(args)