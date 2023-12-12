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

























# https://qib-sheffield.github.io/dbdicom/
# https://github.com/dafne-imaging/dafne-models  !! experimental 
# https://github.com/dafne-imaging/dafne-models/tree/experimental_model_template

# dafne model/bin/create/create model
#generate model

# base_folder
# - patient1.npz
# - patient2.npz

# *.npz
# - data 3d array
# - resolution [x,y,z]  
# - mask_LK  3d array
# - mask_RK


#Steven P Sourbron11:08
#https://qib-sheffield.github.io/dbdicom/
#Francesco Santini11:23
#https://github.com/dafne-imaging/dafne-models
#https://github.com/dafne-imaging/dafne-models/tree/experimental_model_template