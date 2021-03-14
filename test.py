import numpy as np
from foundations import model_base


class test1(model_base.ModelBase):
    def __init__(self,
                 hyperparameters,
                 input_placeholder,
                 label_placeholder,
                 presets=None,
                 masks=None):
        super(test1, self).__init__(presets=presets, masks=masks)
        self.conv2D()