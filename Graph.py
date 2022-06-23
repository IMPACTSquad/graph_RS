import os

import numpy as np
from spektral.data import Dataset, Graph
from sklearn.neighbors import kneighbors_graph

import utils


class AirQuality(Dataset):
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
