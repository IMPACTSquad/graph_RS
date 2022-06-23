import numpy as np
import matplotlib.pyplot as plt
from spektral.data import SingleLoader

import utils
import train


def main():
    trained_model, aq_data = train.train()
    inputs, ground_truth = SingleLoader(aq_data, epochs=1).__next__()
    predictions = trained_model(inputs, training=False).numpy().flatten()

    # undo flattening
    ip = utils.open_pm25()
    gt = utils.open_dem()
    mask = ~np.all(np.isnan(ip), axis=-1) * ~np.isnan(gt)
    predictions_unflattened = np.zeros(ip.shape[:2]) * np.nan
    predictions_unflattened[mask] = predictions.flatten()

    # reverse normalization
    gt_unnormalized = gt[mask]
    op_min, op_max = np.nanmin(gt_unnormalized), np.nanmax(gt_unnormalized)
    predictions_unflattened = predictions_unflattened * (op_max - op_min) + op_min

    vmin = min(np.nanmin(gt), predictions_unflattened.min())
    vmax = max(np.nanmax(gt), predictions_unflattened.max())

    fig, ax = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(12, 5))
    cbar = ax[0].matshow(gt, vmin=vmin, vmax=vmax)
    utils.add_colorbar(fig, cbar, ax[0], x_shift=.06)
    ax[0].set_title('Ground truth DEM')
    cbar = ax[1].matshow(predictions_unflattened, vmin=vmin, vmax=vmax)
    utils.add_colorbar(fig, cbar, ax[1], x_shift=.06)
    ax[1].set_title('Predicted DEM using PM2.5 time series')
    plt.show()


if __name__ == '__main__':
    main()
