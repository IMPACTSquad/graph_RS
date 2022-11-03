import os
import numpy as np
from spektral.data import Dataset, Graph
from sklearn.neighbors import kneighbors_graph
from spektral.datasets.utils import DATASET_FOLDER

import utils


class AirQualityRegression(Dataset):
    """
    Our graph for the air quality data

    Subclassing from https://graphneural.network/creating-dataset/
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def download(self):
        inputs = utils.open_pm25()
        inputs = inputs.reshape(-1, inputs.shape[-1])  # flatten in space, keep time (i.e. last dimension)
        outputs = utils.open_dem().flatten()  # the DEM

        # don't use pixels in graph that don't have inputs or where the output is nan (i.e. over sea)
        mask = ~np.all(np.isnan(inputs), axis=1) * ~np.isnan(outputs)
        outputs = outputs[mask]
        inputs = inputs[mask]

        # normalize inputs
        ip_min, ip_max = np.nanmin(inputs), np.nanmax(inputs)
        inputs = (inputs - ip_min) / (ip_max - ip_min)
        # normalize outputs
        op_min, op_max = np.nanmin(outputs), np.nanmax(outputs)
        outputs = (outputs - op_min) / (op_max - op_min)

        # graph where each pixel has 10 neighbors (based on their proximity in input space)
        adj = kneighbors_graph(inputs, n_neighbors=10, mode='distance', metric='euclidean')

        # save the graph
        filename = os.path.join(self.path, 'graph.npz')
        os.mkdir(self.path)
        np.savez(filename, x=inputs, y=outputs, adj=adj)

    def read(self):
        data = np.load(os.path.join(self.path, 'graph.npz'), allow_pickle=True)
        return [Graph(x=data['x'], a=data['adj'].tolist(), y=data['y'])]


class AirQualityClassification(Dataset):
    """
    Our graph for the air quality data

    Subclassing from https://graphneural.network/creating-dataset/
    """
    def __init__(self, region, datatype, numb_op_classes, seed, train_ratio, val_ratio, **kwargs):
        if region not in ["Italy"]:
            raise NotImplementedError("Parameter 'region' must be one of {}".format(["Italy"]))
        if datatype not in ["PM25"]:
            raise NotImplementedError("Parameter 'datatype' must be one of {}".format(["PM25"]))

        self.mask_tr, self.mask_va, self.mask_te = None, None, None
        self.datatype = datatype
        self.numb_op_classes = numb_op_classes
        self.seed = seed
        self.region = region
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        super().__init__(**kwargs)


    @property
    def path(self):
        return os.path.join(DATASET_FOLDER,
                            self.__class__.__name__
                            ) + "_{}_{}_classes{}_seed{}_train{}_val{}".format(self.region,
                                                                               self.datatype,
                                                                               self.numb_op_classes,
                                                                               self.seed,
                                                                               self.train_ratio,
                                                                               self.val_ratio
                                                                               )

    def download(self):
        x, y = utils.load_xy(with_coords=True)
        x, y = utils.normalize_xy(x, y)
        # replace missing values with means
        x = np.where(np.isnan(x), np.ma.array(x, mask=np.isnan(x)).mean(axis=0), x)

        # get graph adjacency
        a = utils.get_adj(x, k=5)

        splits = utils.make_splits(y, self.seed, train_ratio=self.train_ratio, val_ratio=self.val_ratio)

        # Create the directory
        os.mkdir(self.path)

        filename = os.path.join(self.path, 'graph')
        split_fname = os.path.join(self.path, 'graph_splits.npy'.format(self.numb_op_classes, self.seed))
        np.savez(filename, x=x, a=a, y=y)
        np.save(split_fname, splits)

    def read(self):
        data = np.load(os.path.join(self.path, 'graph.npz'.format(self.numb_op_classes, self.seed)),
                       allow_pickle=True)
        splits = np.load(os.path.join(self.path, 'graph_splits.npy'.format(self.numb_op_classes, self.seed)),
                         allow_pickle=True)
        y = utils.one_hot_classify(data['y'], self.numb_op_classes, splits)
        x, a = data['x'].astype(np.float32), data['a'].tolist()

        self.mask_tr = (splits == "train").flatten()
        self.mask_va = (splits == "val").flatten()
        self.mask_te = (splits == "test").flatten()

        return [Graph(x=x, a=a, y=y)]


