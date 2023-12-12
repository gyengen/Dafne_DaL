# TensorFlow Data API-based Loader for Medical Image Segmentation

DaL (Data Loader) class, is a versatile data loading tool tailored for semantic segmentation challenges, particularly in the context of medical image analysis. It facilitates efficient and customizable preprocessing by offering functions for various image augmentations, including contrast, brightness, saturation adjustments, cropping, and flipping. The class ensures the integrity of the dataset through a pre-loading sanity check, verifying the expected file structure. Employing TensorFlow, DaL parses image and mask data, performs one-hot encoding when required, and creates a TensorFlow dataset for training. Adjustable parameters, such as image size, channels, and augmentation flags, allowing users to customise the loading process.

## Class Initialization

### Parameters

**path** (str): The root path to the dataset containing subdirectories for each patient.

**image_size** (Tuple[int]): Tuple of two integers representing the final height and width of the loaded images.

**channels** (Tuple[int]) [Optional]: Tuple of two integers representing the number of channels in images and masks (default is (3, 3)).

**crop_per** (float) [Optional]: Percentage of image to randomly crop (default is None, no cropping).

**seed** (int) [Optional]: Integer to set the random seed for the data pipeline (default is None, random seed is generated).

**augment** (bool) [Optional]: Boolean indicating whether data augmentation should be applied during training (default is True).

**compose** (bool) [Optional]: Boolean indicating whether to compose multiple augmentations during training (default is False).

**one_hot_encoding** (bool) [Optional]: Boolean indicating whether to perform one-hot encoding on the mask images (default is False).

**palette** [Optional]: A list of RGB pixel values in the mask for one-hot encoding (default is None).

**verbose** (bool) [Optional]: Boolean indicating whether to display verbose information during data loading (default is True).

## Methods

**1. _contrast(image, mask)**
Randomly applies a random contrast change to the image.

**3. _saturation(image, mask)**
Randomly applies a random saturation change to the image.

**5. _brightness(image, mask)**
Randomly applies a random brightness change to the image.

**7. _crop(image, mask)**
Randomly crops the image and mask in accord.

**9. _resize_data(image, mask)**
Resizes images and masks to the specified size.

**11. _flip_left_right(image, mask)**
Randomly flips the image and mask left or right in accord.

**13. _parse_data(image_paths, mask_paths)**
Reads image and mask files depending on the specified extension.

**15. _one_hot_encode(image, mask)**
Converts mask to a one-hot encoding specified by the semantic map.

**17. _map_function(images_path, masks_path)**
TensorFlow function to apply data augmentation and preprocessing.

**19. _file_structure_docs(msg)**
Generates file structure documentation based on an error message.

**21. _pre_sanity_check()**
Performs a pre-loading sanity check on the dataset.

**23. _mapping_img()**
Generates a list of image paths by traversing the directory structure.

**25. _data_batch(data, batch_size, shuffle=False)**
Reads data, normalizes it, shuffles it (if specified), then batches it.

**27. data_read()**
Performs data loading and preprocessing, returning a TensorFlow dataset object.
Usage

# Import the DaL class

```
from datalv3 import DaL
```

# Set the path to the dataset

```
image_paths = {path}
```

# Initialize the DaL object with specified parameters

```
dataset = DaL(path=image_paths,
 image_size=(432, 432),
 crop_per=0.8,
 channels=(3, 1),
 augment=True,
 compose=False,
 seed=47,
 verbose=True)

# Read and preprocess the data, returning a TensorFlow dataset object
data = dataset.data_read()

```
Now `data` can be used for training or evaluation 


