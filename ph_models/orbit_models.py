import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, ReLU, Flatten
from tensorflow.keras import Sequential
import numpy as np
from perslay import CubicalPerslay
from pllay import PersistenceLandscapeLayer, DTMWeightLayer


# Cnn
class Cnn(tf.keras.Model):
    def __init__(self, num_classes=5):
        super().__init__()
        self.conv = Sequential([
            Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu'),
            Conv2D(filters=1, kernel_size=3, strides=1, padding='same')
        ])
        self.fc = Sequential([
            Dense(64, activation='relu'),
            Dense(num_classes)
        ])
        self.relu = ReLU()
        self.flatten = Flatten()

    def call(self, x):
        x, x_dtm = x
        x = self.relu(self.conv(x))
        x = self.fc(self.flatten(x))
        return x


# Cnn + Perslay
class PersCnn(Cnn):
    def __init__(self, num_classes=5, *args, **kwargs):
        super().__init__(num_classes)
        self.perslay = CubicalPerslay(rho=Dense(kwargs["topo_out"], activation='relu'), *args, **kwargs)

    def call(self, x):
        x, x_dtm = x
        pers = self.perslay(x_dtm)  # Perslay
        x = self.relu(self.conv(x)) # CNN
        x = tf.concat((self.flatten(x), pers), axis=-1)
        x = self.fc(x)
        return x


# Cnn + PLLay
class PLCnn_i(Cnn):
    def __init__(self, num_classes=5, *args, **kwargs):
        super().__init__(num_classes)
        self.sublevel, interval = kwargs["sublevel"], kwargs["interval"]
        interval = interval if self.sublevel else [-i for i in reversed(interval)]
        tseq = np.linspace(*interval, kwargs["steps"])
        self.pllay = PersistenceLandscapeLayer(tseq=tseq ,*args, **kwargs)
        self.gtheta = Dense(kwargs["topo_out"], activation='relu') # postprocessing layer

    def call(self, x):
        x, x_dtm = x
        x_dtm = x_dtm if self.sublevel else -x_dtm  # PLLay
        pl = self.pllay(self.flatten(x_dtm))
        pl = self.gtheta(self.flatten(pl))
        x = self.relu(self.conv(x))                 # CNN
        x = tf.concat((self.flatten(x), pl), axis=-1)
        x = self.fc(x)
        return x


# Cnn + PLLay + PLLay after conv
class PLCnn(Cnn):
    def __init__(self, num_classes=5, *args, **kwargs):
        super().__init__(num_classes)
        self.sublevel1, interval1 = kwargs["sublevel_1"], kwargs["interval_1"]
        interval1 = interval1 if self.sublevel1 else [-i for i in reversed(interval1)]
        tseq1 = np.linspace(*interval1, kwargs["steps"])

        self.sublevel2, interval2 = kwargs["sublevel_2"], kwargs["interval_2"]
        interval2 = interval2 if self.sublevel2 else [-i for i in reversed(interval2)]
        tseq2 = np.linspace(*interval2, kwargs["steps"])
        
        self.pllay1 = PersistenceLandscapeLayer(tseq=tseq1, *args, **kwargs)
        self.gtheta1 = Dense(kwargs["topo_out"], activation='relu')
        self.pllay2 = PersistenceLandscapeLayer(tseq=tseq2, *args, **kwargs)
        self.gtheta2 = Dense(kwargs["topo_out"], activation='relu')
        self.dtm = DTMWeightLayer(m0=0.02, lims=[[-0.5, 0.5], [-0.5, 0.5]], by=1/27, r=2)

    def call(self, x):
        x, x_dtm = x
        x_dtm = x_dtm if self.sublevel1 else -x_dtm  # first PLLay
        pl1 = self.pllay1(self.flatten(x_dtm))
        pl1 = self.gtheta1(self.flatten(pl1))
        x = self.conv(x)                            # CNN

        # before ReLU with DTM
        x_max = tf.reduce_max(tf.stop_gradient(x), axis=(1, 2), keepdims=True)  # shape: (B, 1, 1, C)
        x_min = tf.reduce_min(tf.stop_gradient(x), axis=(1, 2), keepdims=True)  # shape: (B, 1, 1, C)
        grids = tf.broadcast_to(self.dtm.grid, shape=(x.shape[0], np.prod(x.shape[1:]), self.dtm.grid.shape[-1]))
        x_dtm2 = tf.reshape(self.dtm(inputs=grids, weight=self.flatten((x-x_min) / (x_max-x_min))), x.shape)    # min-max normalization between 0 and 1 for each data and channel
        pl2 = self.pllay2(self.flatten(x_dtm2 if self.sublevel2 else - x_dtm2))
        pl2 = self.gtheta2(self.flatten(pl2))

        x = tf.concat((self.flatten(self.relu(x)), pl1, pl2), axis=-1)
        x = self.fc(x)
        return x