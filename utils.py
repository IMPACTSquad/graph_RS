import rasterio
import glob
import numpy as np


def open_dem(no_data_val=-3.40e+38):
    dem_tif = rasterio.open('data/DEM.tif').read(1)
    dem_tif[dem_tif < no_data_val] = np.nan
    return dem_tif


def open_pm25(no_data_val=-999):
    files = glob.glob('data/PM2_5/*.tif')
    pm25 = []
    for f in files:
        with rasterio.open(f) as src:
            pm25.append(src.read(1))
    pm25 = np.stack(pm25, axis=-1)
    pm25[pm25 == no_data_val] = np.nan
    return pm25


def get_masks(ops, seed, train_ratio=.3, val_ratio=.3):
    np.random.seed(seed)
    N = len(ops.flatten())

    # don't put any no data pixels in any set (i.e. should contribute to any loss values)
    idxs = np.argwhere(ops.flatten() != -999).flatten()

    # shuffle the indices
    idxs = np.random.choice(idxs, size=len(idxs), replace=False)

    # take first train_ratio * len(idxs) indices for training, and so on
    masks = idxs[:int(len(idxs) * train_ratio)], idxs[int(len(idxs) * train_ratio):int(len(idxs) * (train_ratio + val_ratio))],\
           idxs[int(len(idxs) * (train_ratio + val_ratio)):]

    mask_tr = np.zeros(N).astype(bool)
    mask_va = np.zeros(N).astype(bool)
    mask_te = np.zeros(N).astype(bool)

    mask_tr[masks[0]] = True
    mask_va[masks[1]] = True
    mask_te[masks[2]] = True
    return mask_tr, mask_va, mask_te


def mask_to_weights(mask):
    return mask.astype(np.float32) * len(mask) / np.count_nonzero(mask)


def add_colorbar(fig, img, one_ax, x_shift=0.2, height_scale=0.95):
    bounds = one_ax.get_position().bounds
    bounds = (bounds[0] + x_shift, (3 - height_scale) * bounds[1] / 2, bounds[2], bounds[3] * height_scale,)
    cbar = fig.add_axes(bounds)
    cbar.axis("off")
    fig.colorbar(img, ax=cbar)
