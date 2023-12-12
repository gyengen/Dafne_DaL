from datalv2 import DaL
import numpy as np

image_paths = user_path

dataset = DaL(path=image_paths,
              image_size=(432, 432),
              crop_per=0.8,
              channels=(3, 1),
              augment=True,
              compose=False,
              seed=47,
              verbose = True)

data = dataset.data_read()
