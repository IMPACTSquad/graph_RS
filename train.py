"""
This example implements the experiments on citation networks from the paper:

Semi-Supervised Classification with Graph Convolutional Networks (https://arxiv.org/abs/1609.02907)
Thomas N. Kipf, Max Welling
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam

from spektral.data.loaders import SingleLoader
from spektral.layers import GCNConv
from spektral.models.gcn import GCN
from spektral.transforms import LayerPreprocess

import utils
from Graph import AirQuality


def train():
    learning_rate = 1e-2
    seed = 1
    epochs = 1000
    patience = 50

    tf.random.set_seed(seed=seed)  # make weight initialization reproducible

    # Load data
    dataset = AirQuality(transforms=[LayerPreprocess(GCNConv)])
    mask_tr, mask_va, mask_te = [utils.mask_to_weights(mask) for mask in utils.get_masks(dataset[0].y, seed, train_ratio=.3, val_ratio=.3)]

    model = GCN(n_labels=1, channels=32, output_activation="sigmoid")
    model.compile(
        optimizer=Adam(learning_rate),
        loss=MeanSquaredError(),
    )

    # Train model
    loader_tr = SingleLoader(dataset, sample_weights=mask_tr)
    loader_va = SingleLoader(dataset, sample_weights=mask_va)
    model.fit(
        loader_tr.load(),
        steps_per_epoch=loader_tr.steps_per_epoch,
        validation_data=loader_va.load(),
        validation_steps=loader_va.steps_per_epoch,
        epochs=epochs,
        callbacks=[EarlyStopping(patience=patience, restore_best_weights=True, monitor="val_loss")],
    )

    # Evaluate model
    print("Evaluating model.")
    inputs, gt = SingleLoader(dataset, epochs=1).__next__()
    predictions = model(inputs, training=False).numpy().flatten()
    test_rmse = tf.reduce_sum(((predictions - gt.flatten()) ** 2) * mask_te / mask_te.sum())
    print("Done.\n" "Test Mean Squared Error: {}".format(test_rmse))
    return model, dataset
