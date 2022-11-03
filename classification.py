import numpy as np
import matplotlib.pyplot as plt
from spektral.data import SingleLoader
from matplotlib.colors import ListedColormap

import utils
import train


def main():
    trained_model, aq_data = train.train_classification()
    inputs, ground_truth = SingleLoader(aq_data, epochs=1).__next__()
    predictions = trained_model(inputs, training=False).numpy()
    predicted_labels = np.argmax(predictions, axis=1).reshape(-1, 50)
    gt_labels = predicted_labels.flatten().astype(np.float64) * np.nan
    gt_labels[aq_data.mask_tr] = ground_truth.argmax(axis=1)[aq_data.mask_tr]

    mappable = ListedColormap([[.2, .9, .2], [1., 1., 0.], [1., .8, .0], [1., 0., 0.]])
    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
    ax[0].matshow(predicted_labels, extent=(0, predicted_labels.shape[1], predicted_labels.shape[0] / 24, 0),
                  cmap=mappable)
    ax[0].set_title("Predictions")
    ax[0].set_ylabel("Days")
    ax[0].set_xlabel("Ground Stations")
    ax[1].matshow(gt_labels.reshape(-1, 50), extent=(0, predicted_labels.shape[1], predicted_labels.shape[0] / 24, 0),
                  cmap=mappable)
    ax[1].set_title("Ground Truth\n(training only)")
    ax[1].set_ylabel("Days")
    ax[1].set_xlabel("Ground Stations")
    gt_labels[aq_data.mask_va] = ground_truth.argmax(axis=1)[aq_data.mask_va]
    gt_labels[aq_data.mask_te] = ground_truth.argmax(axis=1)[aq_data.mask_te]
    ax[2].matshow(gt_labels.reshape(-1, 50), extent=(0, predicted_labels.shape[1], predicted_labels.shape[0] / 24, 0),
                  cmap=mappable)
    ax[2].set_title("Ground Truth\n(all)")
    ax[2].set_ylabel("Days")
    ax[2].set_xlabel("Ground Stations")
    plt.show()


if __name__ == '__main__':
    main()
