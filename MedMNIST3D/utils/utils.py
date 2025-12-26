import torch.nn as nn
import numpy as np
from .batchnorm import SynchronizedBatchNorm3d, SynchronizedBatchNorm2d

class Transform3D:

    def __init__(self, mul=None):
        self.mul = mul

    def __call__(self, voxel):
   
        if self.mul == '0.5':
            voxel = voxel * 0.5
        elif self.mul == 'random':
            voxel = voxel * np.random.uniform()
        
        return voxel.astype(np.float32)


def model_to_syncbn(model):
    preserve_state_dict = model.state_dict()
    _convert_module_from_bn_to_syncbn(model)
    model.load_state_dict(preserve_state_dict)
    return model


def _convert_module_from_bn_to_syncbn(module):
    for child_name, child in module.named_children(): 
        if hasattr(nn, child.__class__.__name__) and \
            'batchnorm' in child.__class__.__name__.lower():
            TargetClass = globals()['Synchronized'+child.__class__.__name__]
            arguments = TargetClass.__init__.__code__.co_varnames[1:]
            kwargs = {k: getattr(child, k) for k in arguments}
            setattr(module, child_name, TargetClass(**kwargs))
        else:
            _convert_module_from_bn_to_syncbn(child)


class EarlyStopping:
    def __init__(self, patience, threshold, val_metric="loss"):
        """_summary_

        Args:
            patience (_type_): _description_
            threshold (_type_): _description_
            val_metric (str, optional): _description_. Defaults to "loss".
        """
        self.patience = patience
        self.threshold = threshold
        self.count = 0
        self.best_loss, self.best_acc, self.best_epoch = float("inf"), 0, None
        self.val_metic = val_metric

    def stop_training(self, val_loss, val_acc, epoch):
        stop, improvement = True, True
        diff = (self.best_loss - val_loss) if self.val_metic == "loss" else (val_acc - self.best_acc)
        if diff > self.threshold:   # improvement needs to be above threshold 
            self.count = 0
            self.best_loss, self.best_acc, self.best_epoch = val_loss, val_acc, epoch
            return not stop, improvement
        else:
            self.count += 1
            if self.count > self.patience:  # stop training if no improvement for patience + 1 epochs
                print("-"*30)
                print(f"Best Epoch: {self.best_epoch}")
                print(f"Best Validation Accuracy: {(self.best_acc):>0.1f}%")
                print(f"Best Validation Loss: {self.best_loss:>8f}")
                print("-"*30)
                return stop, not improvement
            return not stop, not improvement