import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras import Sequential
import numpy as np
from perslay import CubicalPerslay
from pllay import PersistenceLandscapeLayer


# Cnn
class Cnn(tf.keras.Model):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = Sequential([
            Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu'),
            Conv2D(filters=1, kernel_size=3, strides=1, padding='same', activation='relu')
        ])
        self.fc = Sequential([
            Dense(64, activation='relu'),
            Dense(num_classes)
        ])
        self.flatten = Flatten()

    def call(self, x):
        x, x_dtm = x
        x = self.conv(x)
        x = self.fc(self.flatten(x))
        return 


# Cnn + Perslay
class PersCnn(Cnn):
    def __init__(self, num_classes=10, *args, **kwargs):
        super().__init__()
        self.perslay = CubicalPerslay(rho=Dense(kwargs["steps"], activation='relu'), *args, **kwargs)
        self.fc = Sequential([
            Dense(64, activation='relu'),
            Dense(num_classes)
        ])

    def call(self, x):
        x, x_dtm = x
        pers = self.perslay(x_dtm)  # Perslay
        x = self.conv(x)            # CNN
        x = tf.concat((self.flatten(x), pers), axis=-1)
        x = self.fc(x)
        return x


# Cnn + PLLay
class PLCnn_i(Cnn):
    def __init__(self, num_classes=10, *args, **kwargs):
        super().__init__()
        self.sublevel, interval = kwargs["sublevel"], kwargs["interval"]
        interval = interval if self.sublevel else [-i for i in reversed(interval)]
        tseq = np.linspace(*interval, kwargs["steps"])
        self.pllay = PersistenceLandscapeLayer(tseq=tseq ,*args, **kwargs)
        self.gtheta = Dense(kwargs["steps"], activation='relu') # postprocessing layer
        self.fc = Sequential([
            Dense(64, activation='relu'),
            Dense(num_classes)
        ])

    def call(self, x):
        x, x_dtm = x
        pl = self.pllay(self.flatten(x_dtm if self.sublevel else -x_dtm))   # PLLay
        pl = self.gtheta(self.flatten(pl))
        x = self.conv(x)                                                    # CNN
        x = tf.concat((self.flatten(x), pl), axis=-1)
        x = self.fc(x)
        return x


class PLCnn(Cnn):
    def __init__(self, num_classes=10, *args, **kwargs):
        super().__init__()
        self.sublevel_1, interval_1 = kwargs["sublevel_1"], kwargs["interval_1"]
        interval_1 = interval_1 if self.sublevel_1 else [-i for i in reversed(interval_1)]
        tseq_1 = np.linspace(*interval_1, kwargs["steps"])

        self.sublevel_2, interval_2 = kwargs["sublevel_2"], kwargs["interval_2"]
        interval_2 = interval_2 if self.sublevel_2 else [-i for i in reversed(interval_2)]
        tseq_2 = np.linspace(*interval_2, kwargs["steps"])
        
        self.pllay_1 = PersistenceLandscapeLayer(tseq=tseq_1, *args, **kwargs)
        self.gtheta_1 = Dense(kwargs["steps"], activation='relu')
        self.pllay_2 = PersistenceLandscapeLayer(tseq=tseq_2, *args, **kwargs)
        self.gtheta_2 = Dense(kwargs["steps"], activation='relu')
        self.fc = Sequential([
            Dense(64, activation='relu'),
            Dense(num_classes)
        ])

    def call(self, x):
        x, x_dtm = x
        pl_1 = self.pllay_1(self.flatten(x_dtm if self.sublevel_1 else -x_dtm)) # first PLLay
        pl_1 = self.gtheta_1(self.flatten(pl_1))
        x = self.conv(x)                                                        # CNN

        # second PLLay after conv layer
        max_vals = tf.reduce_max(x, axis=(1, 2), keepdims=True) # shape: (B, 1, 1, C)
        if tf.reduce_all(max_vals != 0):
            x_2  = x / max_vals     # normalize between 0 and 1 for each data and channel
            pl_2 = self.pllay_2(self.flatten(x_2 if self.sublevel_2 else - x_2))
            pl_2 = self.gtheta_2(self.flatten(pl_2))
        else:
            pl_2 = self.pllay_2(self.flatten(x if self.sublevel_2 else - x))
            pl_2 = self.gtheta_2(self.flatten(pl_2))

        x = tf.concat((self.flatten(x), pl_1, pl_2), axis=-1)
        x = self.fc(x)
        return x