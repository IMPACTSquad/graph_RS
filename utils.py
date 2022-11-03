import rasterio
from glob import glob
import numpy as np
import pandas as pd
from sklearn.neighbors import kneighbors_graph
import os
import geopandas as gpd
from config import DATA_DIR


def open_dem(no_data_val=-3.40e+38):
    dem_tif = rasterio.open('data/DEM.tif').read(1)
    dem_tif[dem_tif < no_data_val] = np.nan
    return dem_tif


def open_pm25(no_data_val=-999):
    files = glob('data/PM2_5/*.tif')
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


def open_ground_stations_shp(region, datatype):
    return gpd.read_file(get_ground_station_path(region, datatype))


def get_ground_stations(region, datatype):
    return open_ground_stations_shp(region, datatype).geometry.unique()
    
   
def load_xy(with_coords=False):
    """
    returns x (numb_samples, numb_features) and y (numb_samples,) with np.nan values if no data
    """
    files = glob(os.path.join(DATA_DIR, "Italy", "CAMS", "PM2_5", "CAMS*.tif"))
    days = sorted(np.unique([int(f.split("day")[-1].split('_')[0]) for f in files]))
    station_measurements = gpd.read_file(os.path.join(DATA_DIR, "Italy", "ground_air_quality", "PM25",
                                                      "PM25_italy_ground.shp"))
    stations = get_ground_stations("Italy", "PM25")

    # cross-reference stations with measurements
    data_dict = {}
    for i, station in enumerate(stations):
        sub_df = station_measurements[station_measurements.geometry == station]
        data_dict[i] = pd.Series(sub_df.AirQuality.values, index=sub_df.Date.values)

    df = pd.DataFrame(data_dict)
    missing_days = [day for day in days if day not in df.index.values]
    df = pd.concat([df, pd.DataFrame(np.nan, index=missing_days, columns=df.columns)])
    df.sort_index(inplace=True)
    station_measurements = df.to_numpy(dtype=np.float64, na_value=np.nan)

    x = []
    y = []

    camsNO2_path = None
    sen5pNO2_path = None

    for i, day in enumerate(days):
        for hour in range(24):
            path = CAMS_PM25_fpath(day, hour)
            if file_exists(path):
                camsPM25_path = path

                if file_exists(CAMS_NO2_fpath(day, hour)):
                    camsNO2_path = CAMS_NO2_fpath(day, hour)
                if file_exists(SEN5P_NO2_fpath(day, hour)):
                    sen5pNO2_path = SEN5P_NO2_fpath(day, hour)

                # outputs
                for j in range(len(stations)):
                    y.append([station_measurements[i, j]])

                # inputs
                with rasterio.open(camsPM25_path) as camsPM25_tif:
                    if camsNO2_path is not None:
                        with rasterio.open(camsNO2_path) as camsNO2_tif:
                            if sen5pNO2_path is not None:
                                with rasterio.open(sen5pNO2_path) as sen5pNO2_tif:
                                    for s in stations:
                                        x.append([])
                                        x[-1].append(sample_tif(camsPM25_tif, s))
                                        x[-1].append(sample_tif(camsNO2_tif, s))
                                        x[-1].append(sample_tif(sen5pNO2_tif, s))
                                        if with_coords:
                                            x[-1].append(s.x)
                                            x[-1].append(s.y)
                            else:
                                for s in stations:
                                    x.append([])
                                    x[-1].append(sample_tif(camsPM25_tif, s))
                                    x[-1].append(sample_tif(camsNO2_tif, s))
                                    x[-1].append(np.nan)
                                    if with_coords:
                                        x[-1].append(s.x)
                                        x[-1].append(s.y)
                    else:
                        for s in stations:
                            x.append([])
                            x[-1].append(sample_tif(camsPM25_tif, s))
                            x[-1].append(np.nan)
                            x[-1].append(np.nan)
                            if with_coords:
                                x[-1].append(s.x)
                                x[-1].append(s.y)

    return np.array(x), np.array(y)


def open_ground_stations_shp(region, datatype):
    return gpd.read_file(get_ground_station_path(region, datatype))


def get_ground_stations(region, datatype):
    return open_ground_stations_shp(region, datatype).geometry.unique()


def CAMS_NO2_fpath(day, hour, region="Italy"):
    fname = "CAMS_NO2_day{}_h{}.tif".format(day, "0" * (2 - len(str(hour))) + str(hour))
    return os.path.join(DATA_DIR, region, "CAMS", "NO2_surface", fname)


def CAMS_PM25_fpath(day, hour, region="Italy"):
    fname = "CAMS_PM2_5_day{}_h{}.tif".format(day, "0" * (2 - len(str(hour))) + str(hour))
    return os.path.join(DATA_DIR, region, "CAMS", "PM2_5", fname)


def SEN5P_NO2_fpath(day, hour, region="Italy"):
    fname = "S5P_NO2_OFFL_L2_day{}_T{}.tif".format(day, "0" * (2 - len(str(hour))) + str(hour))
    return os.path.join(DATA_DIR, region, "sentinel5P", "NO2", fname)


def file_exists(fpath):
    return os.path.isfile(fpath)


def sample_tif(tif, point, low=-900, high=900):
    if tif is None:
        return np.nan
    else:
        val = list(rasterio.sample.sample_gen(tif, point.coords[:]))[0][0]
        if low < val < high:
            return val
        else:
            return np.nan


def get_ground_station_path(region, datatype):
    if region not in ["Italy", "California", "South Africa"]:
        raise ValueError("Region '{}' data does not exist".format(region))
    if datatype not in ["NO2", "PM25"]:
        raise ValueError("Datatype '{}' data does not exist".format(datatype))

    folder = os.path.join(DATA_DIR, '_'.join(region.split(' ')), "ground_air_quality", datatype)
    path_list = glob(os.path.join(folder, "*.shp"))
    if len(path_list) == 0:
        raise FileNotFoundError("No shapefile found for region '{}' and datatype '{}'".format(region, datatype))
    elif len(path_list) > 1:
        raise ValueError("Multiple shapefiles found for region '{}' and datatype '{}'".format(region, datatype))
    else:
        return path_list[0]


def add_colorbar(fig, img, one_ax, x_shift=0.2, height_scale=0.95):
    bounds = one_ax.get_position().bounds
    bounds = (bounds[0] + x_shift, (3 - height_scale) * bounds[1] / 2, bounds[2], bounds[3] * height_scale,)
    cbar = fig.add_axes(bounds)
    cbar.axis("off")
    fig.colorbar(img, ax=cbar)


def one_hot_classify(y_cont, numb_classes, splits):
    """
    Takes the continuously-valued air quality measurements and bins them into classes

    The class boundaries are computed to give a roughly equal number of samples in each class
    These boundaries are computed using the air quality measurements in the training set only

    Returns: one-hot encoded matrix (numb_samples, numb_classes) with any all-zero rows corresponding to missing data
    """
    percentiles = np.linspace(0, 100, numb_classes + 1)
    # compute class boundaries based on observed (training) data only
    boundaries = np.percentile(y_cont[splits == "train"], q=percentiles)

    y_one_hot = np.stack([((y_cont > low) * (y_cont < high)).flatten() for low, high in zip(boundaries[:-1],
                                                                                            boundaries[1:])],
                         axis=1)
    return y_one_hot.astype(int)


def make_splits(y_cont, seed, train_ratio, val_ratio):
    np.random.seed(seed)
    # since ground stations measure only once per day and get repeated 24 times (to give per hour labels), sample every
    # 24th value when reshaped to give time in axis 0 (this allows us to split entire days into train/test rather than
    # by hour which would allow specific ground station measurements to exist across dataset splits)
    y_daily = y_cont.reshape(-1, 50)[::24].flatten()
    numb = np.sum((~np.isnan(y_daily)))  # number of ground stations with measurements (not nan)
    idxs = np.random.choice(np.argwhere(~np.isnan(y_daily)).flatten(), size=numb, replace=False)
    train = idxs[:int(numb * train_ratio)]
    val = idxs[int(numb * train_ratio):int(numb * (train_ratio + val_ratio))]
    test = idxs[int(numb * (train_ratio + val_ratio)):]
    splits = np.array(["empty"] * y_daily.shape[0])
    splits[train] = 'train'
    splits[val] = 'val'
    splits[test] = 'test'

    # reverse the thinning by 24 to get back to value for every hour
    splits = np.repeat(splits.reshape(-1, 50), 24, axis=0).flatten()
    return splits


def normalize_xy(x, y):
    """
    Linearly scale x and y to between 0 and 1 using min/max values (ignoring nan values)
    """
    x = (x - np.nanmin(x, axis=0, keepdims=True)) / (
            np.nanmax(x, axis=0, keepdims=True) - np.nanmin(x, axis=0, keepdims=True))
    y = (y - np.nanmin(y, axis=0, keepdims=True)) / (
            np.nanmax(y, axis=0, keepdims=True) - np.nanmin(y, axis=0, keepdims=True))
    return x, y


def get_adj(x, k):
    print("Computing k neighbors graph...")
    a = kneighbors_graph(x, k, include_self=False)
    a = a + a.T  # to make graph symmetric (using k neighbours in "either" rather than "mutual" mode)
    a[a > 1] = 1  # get rid of any edges we just made double
    print("Graph computed.")
    return a
