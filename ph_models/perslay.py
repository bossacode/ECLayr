import tensorflow as tf
import gudhi.tensorflow.perslay as prsl
from tensorflow.keras.layers import Dense
import gudhi as gd
import numpy as np


class CubicalPerslay(tf.keras.layers.Layer):
    def __init__(self, interval=[0., 1.], steps=32, sublevel=True, k=2, rho=tf.identity,
                 *args, **kwargs):
        super().__init__()
        self.sublevel = sublevel
        interval = interval if sublevel else [-i for i in reversed(interval)]
        self.t_min, self.t_max = interval
        self.k = k

        # set grid
        grid = np.random.uniform(1, 1, size=(10, 10)).astype(np.float32)
        grid_bnds = (interval, interval)
        weight = prsl.GridPerslayWeight(grid=grid, grid_bnds=grid_bnds)
        # point transformation
        phi = prsl.FlatPerslayPhi(np.linspace(self.t_min, self.t_max, steps).astype(np.float32), theta=50.)
        # permutation invariant op
        perm_op = f"top{k}"
        self.perslay = prsl.Perslay(weight=weight, phi=phi, perm_op=perm_op, rho=rho)
    
    def call(self, x):
        batch_size, _, _, num_channels = x.shape
        diag_list = []
        x = x if self.sublevel else -x
        for b in range(batch_size):
            for c in range(num_channels):
                cub_cpx = gd.CubicalComplex(vertices=x[b, :, :, c])
                diag = cub_cpx.persistence()
                for dim in range(2):
                    dim_diag = np.array([pair[1] for pair in diag if pair[0] == dim and pair[1][0] > self.t_min and pair[1][1] < self.t_max], dtype=np.float32)
                    num_hom = len(dim_diag)         # number of homology features in dimension "dim"
                    if num_hom < self.k:            # concatenate zero arrays if there are less than k homology features
                        if num_hom == 0:
                            dim_diag = np.zeros((self.k, 2), dtype=np.float32) + self.t_min
                        dim_diag = tf.concat((dim_diag, np.zeros((self.k - num_hom, 2), dtype=np.float32) + self.t_min), axis=0)
                    dim_diag = tf.RaggedTensor.from_tensor(dim_diag[None, :])
                    diag_list.append(dim_diag)
        diags = tf.concat(diag_list, axis=0)
        pers = self.perslay(diags)                  # shape: (B*C*2, steps*k)
        pers = tf.reshape(pers, (batch_size, -1))   # size: (B, C*k*2*steps)
        return pers