import argparse
import numpy as np
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('classes')
parser.add_argument('image')
parser.add_argument('--top_n', type=int, default=2)


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


# Define the 'plot_auroc' function
# def plot_auroc(name, labels, predictions, **kwargs):
#     fp, tp, _ = roc_curve(labels, predictions)
#     auROC = 1 - roc_auc_score(labels, predictions)
#     plt.plot(fp, tp, label=name + ' (AUC = %0.3f)' % auROC, linewidth=2, **kwargs)
#     plt.xlabel('False positives [%]')
#     plt.ylabel('True positives [%]')
#     plt.xlim([-0.05,1.05])
#     plt.ylim([-0.05,1.05])
#     plt.legend(loc='lower right')

#     plt.savefig(os.path.join(args.result_root, 'auroc.png'))
#     plt.clf()



# def plot_roc(name, labels, predictions, **kwargs):
#     fp, tp, _ = roc_curve(labels, predictions)

#     plt.plot(fp, tp, label=name, linewidth=2, **kwargs)
#     plt.xlabel('False positives [%]')
#     plt.ylabel('True positives [%]')
#     plt.xlim([-0.05,0.40])
#     plt.ylim([0.6,1.05])
#     plt.grid(True)
#     ax = plt.gca()
#     ax.set_aspect('equal')


def main(args):

    # create model
    model = load_model(args.model)

    # load class names
    classes = []
    with open(args.classes, 'r') as f:
        classes = list(map(lambda x: x.strip(), f.readlines()))

    # load an input image
    img = image.load_img(args.image, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # predict
    pred = model.predict(x)[0]
    result = [(classes[i], float(pred[i]) * 100.0) for i in range(len(pred))]
    result.sort(reverse=True, key=lambda x: x[1])
    for i in range(args.top_n):
        (class_name, prob) = result[i]
        print("Top %d ====================" % (i + 1))
        print("Class name: %s" % (class_name))
        print("Probability: %.2f%%" % (prob))

    # Plot the auroc graph on the test dataset
    # plot_path = os.path.join(result_path_name, 'test_auroc_')
    # plot_auroc(model, test_dataset, plot_path, 'Test')


    # plot_path = os.path.join(result_path_name, 'auroc_')
    # plot_auroc(hist_fine, plot_path, 'AUROC')


    # # Plot the ROC curve
    # plot_roc("Test", test_labels, test_predictions_baseline)
    # plt.savefig(os.path.join(args.result_root, 'auroc.png'))
    # plt.clf()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)