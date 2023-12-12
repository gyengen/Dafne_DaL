from typing import List, Tuple

import tensorflow as tf
import dicom2jpg
import logging
import random
import numpy as np
import imghdr
#import dicom
import os


AUTOTUNE = tf.data.experimental.AUTOTUNE  # Setting AUTOTUNE constant for TensorFlow data pipeline optimization
logging.basicConfig(level=logging.INFO)  # Configuring logging level to INFO


class DaL(object):

    """
    TensorFlow Data API based loader for semantic segmentation problems.
    If arguments are not defined augmentation functions will be
    chosen randomly.
    """

    def __init__(self, path: str, image_size: Tuple[int],
                 channels: Tuple[int] = (3, 3), crop_per: float = None,
                 seed: int = None,augment: bool = True, compose: bool = False,
                 one_hot_encoding: bool = False, palette=None,
                 verbose: bool = True):

        """
        Constructor for the DaL class. 
        It initializes various data loading parameters.

        Arguments:
            - path: The root path to the dataset containing subdirectories
                for each patient.
            - image_size: Tuple of two integers representing the final height
                and width of the loaded images.
            - channels: Tuple of two integers representing the number of
                channels in images and masks.
            - crop_per: Float (0-1) or percentage (0-100) defining the
                percentage of image to randomly crop.
            - seed: Integer to set the random seed for the data pipeline.
            - augment: Boolean indicating whether data augmentation should be
                applied during training.
            - compose: Boolean indicating whether to compose multiple
                augmentations during training.
            - one_hot_encoding: Boolean indicating whether to perform one-hot
                encoding on the mask images.
            - palette: A list of RGB pixel values in the mask for one-hot
                encoding.
            - verbose: Boolean indicating whether to display verbose
                information during data loading.
        """

        # Flag for verbose information during data loading
        self.verbose = verbose  

        # Root path to the dataset
        self.path = path

        # List to store paths of image files
        self.image_paths = []  

        # List to store paths of mask files
        self.mask_paths = []  

        # Palette for one-hot encoding
        self.palette = palette  

        # Tuple representing the final height and width of the loaded images
        self.image_size = image_size  

        # Flag indicating whether data augmentation should be applied during training
        self.augment = augment  

        # Flag indicating whether to compose multiple augmentations during training
        self.compose = compose  

        # Flag indicating whether to perform one-hot encoding on the mask images
        self.one_hot_encoding = one_hot_encoding  

        if crop_per is not None:
            if 0.0 < crop_per <= 1.0:
                self.crop_per = tf.constant(crop_per, tf.float32)
            elif 0 < crop_per <= 100:
                self.crop_per = tf.constant(crop_per / 100., tf.float32)
            else:
                raise ValueError("Invalid value entered for crop size. \
                                  Please use an integer between 0 and 100, \
                                  or a float between 0 and 1.0")
        else:
            self.crop_per = None

        # Tuple representing the number of channels in images and masks
        self.channels = channels

        # Setting random seed for data pipeline
        if seed is None:
            self.seed = random.randint(0, 1000)

        else:
            self.seed = seed


    def _contrast(self, image, mask):

        """
        Function to randomly apply a random contrast change to the image.

        Arguments:
            - image: The image tensor.
            - mask: The mask tensor.

        Returns:
            - image: The augmented image tensor.
            - mask: The mask tensor (unchanged in this case).
        """

        # Generate a random boolean condition for applying contrast
        cond_con = tf.cast(tf.random.uniform([],maxval=2, dtype=tf.int32),
                           tf.bool)

        # Apply random contrast if the condition is True, else keep the image unchanged
        image = tf.cond(cond_con,
                        lambda: tf.image.random_contrast(image, 0.1, 0.8),
                        lambda: tf.identity(image))

        return image, mask


    def _saturation(self, image, mask):

        """
        Function to randomly apply a random saturation to the image.

        Arguments:
            - image: The image tensor.
            - mask: The mask tensor.
        Returns:
            - image: The augmented image tensor.
            - mask: The mask tensor (unchanged in this case).
        """

        # Generate a random boolean condition for applying saturation
        cond_saturation = tf.cast(tf.random.uniform([],
                                                    maxval=2,
                                                    dtype=tf.int32),
                                                    tf.bool)

        # Apply random saturation if the condition is True, else keep the image unchanged
        image = tf.cond(cond_saturation,
                        lambda: tf.image.random_saturation(image, 0.1, 0.8),
                        lambda: tf.identity(image))

        return image, mask


    def _brightness(self, image, mask):

        """
        Function to randomly apply a random brightness change to the image.

        Arguments:
            - image: The image tensor.
            - mask: The mask tensor.

        Returns:
            - image: The augmented image tensor.
            - mask: The mask tensor (unchanged in this case).
        """

        # Generate a random boolean condition for applying brightness
        br = tf.cast(tf.random.uniform([], maxval=2, dtype=tf.int32), tf.bool)

         # Apply random brightness if the condition is True, else keep the image unchanged
        image = tf.cond(br,
                        lambda: tf.image.random_brightness(image, 0.1),
                        lambda: tf.identity(image))

        return image, mask


    def _crop(self, image, mask):

        """
        Function to randomly crop the image and mask in accord.

         Arguments:
            - image: The image tensor.
            - mask: The mask tensor.

        Returns:
            - image: The augmented image tensor.
            - mask: The mask tensor (unchanged in this case).
        """

        # Generate a random boolean condition for applying crop
        cond_crop = tf.cast(tf.random.uniform([], maxval=2, dtype=tf.int32,
                            seed=self.seed), tf.bool)

        # Calculate crop dimensions based on the percentage specified
        shape = tf.cast(tf.shape(image), tf.float32)
        h = tf.cast(shape[0] * self.crop_per, tf.int32)
        w = tf.cast(shape[1] * self.crop_per, tf.int32)

        # Concatenate image and mask along channels axis for joint cropping
        comb_tensor = tf.concat([image, mask], axis=2)

        # Apply random crop if the condition is True, else keep the tensor unchanged
        ir = tf.image.random_crop(comb_tensor,[h,
                                               w,
                                               self.channels[0] + self.channels[1]],
                                               seed=self.seed)
        
        comb_tensor = tf.cond(cond_crop,
                              lambda: ir,
                              lambda: tf.identity(comb_tensor))

        # Split the combined tensor back into image and mask
        image, mask = tf.split(comb_tensor,
                               [self.channels[0], self.channels[1]], axis=2)

        return image, mask


    def _resize_data(self, image, mask):

        """
        Function to resize images and masks to the specified size.

        Arguments:
            - image: The image tensor.
            - mask: The mask tensor.

        Returns:
            - image: The resized image tensor.
            - mask: The resized mask tensor.
        """

        image = tf.image.resize(image, self.image_size)
        mask = tf.image.resize(mask, self.image_size, method="nearest")
        
        return image, mask


    def _flip_left_right(self, image, mask):

        """
        Randomly flip the image and mask left or right in accord.

         Arguments:
            - image: The image tensor.
            - mask: The mask tensor.

        Returns:
            - image: The augmented image tensor.
            - mask: The mask tensor (unchanged in this case).
        """


        # Concatenate image and mask along channels axis for joint flipping
        comb_tensor = tf.concat([image, mask], axis=2)

        # Apply random left-right flip
        comb_tensor = tf.image.random_flip_left_right(comb_tensor,
                                                      seed=self.seed)

        # Split the combined tensor back into image and mask
        image, mask = tf.split(comb_tensor,
                               [self.channels[0], self.channels[1]], axis=2)

        return image, mask


    def _parse_data(self, image_paths, mask_paths):

        """
        Read image and mask files depending on the specified extension.

        Arguments:
            - image_paths: The path to the image file.
            - mask_paths: The path to the mask file.
        
        Returns:
            - images: The decoded image tensor.
            - masks: The decoded mask tensor.
         """

        #image_paths = self.image_paths
        #mask_paths = self.mask_paths

        # Read the content of the image and mask files
        image_content = tf.io.read_file(image_paths)
        mask_content = tf.io.read_file(mask_paths)

        # Decode the content of the image and mask files as JPEG images
        images = tf.image.decode_jpeg(image_content, channels=self.channels[0])
        masks = tf.image.decode_jpeg(mask_content, channels=self.channels[1])

        return images, masks


    def _one_hot_encode(self, image, mask):

        """
        Convert mask to a one-hot encoding specified by the semantic map.
        
        Arguments:
            - image: The image tensor.
            - mask: The mask tensor.
        
        Returns:
            - image: The image tensor (unchanged in this case).
            - one_hot_map: The one-hot encoded mask tensor.
        """

        # Initialize an empty list to store one-hot encoded class maps
        one_hot_map = []

        # Iterate through the specified colors in the palette
        for colour in self.palette:

            # Create a boolean tensor indicating where mask equals the specified color
            class_map = tf.reduce_all(tf.equal(mask, colour), axis=-1)
            one_hot_map.append(class_map)

        # Stack the boolean tensors along the last axis to create the one-hot encoding
        one_hot_map = tf.stack(one_hot_map, axis=-1)
        one_hot_map = tf.cast(one_hot_map, tf.float32)
        
        return image, one_hot_map

    @tf.function
    def _map_function(self, images_path, masks_path):

        """
        TensorFlow function to apply data augmentation and preprocessing.

        Arguments:
            - images_path: The path to the image file.
            - masks_path: The path to the mask file.

        Returns:
            - image: The augmented and preprocessed image tensor.
            - mask: The augmented and preprocessed mask tensor.
        """

        # Parse the data from image and mask paths
        image, mask = self._parse_data(images_path, masks_path)

        def _augmentation_func(image_f, mask_f):

            # Apply augmentation functions based on configuration
            if self.augment:
                if self.compose:
                    image_f, mask_f = self._brightness(image_f, mask_f)
                    image_f, mask_f = self._contrast(image_f, mask_f)
                    image_f, mask_f = self._saturation(image_f, mask_f)
                    image_f, mask_f = self._crop(image_f, mask_f)
                    image_f, mask_f = self._flip_left_right(image_f, mask_f)

                else:
                    # Randomly select and apply one augmentation function
                    options = [self._brightness,
                               self._contrast,
                               self._saturation,
                               self._crop,
                               self._flip_left_right]
                    augment_func = random.choice(options)
                    image_f, mask_f = augment_func(image_f, mask_f)

            # Apply one-hot encoding if specified
            if self.one_hot_encoding:
                if self.palette is None:
                    raise ValueError('No Palette for one-hot encoding \
                                      specified in the data loader! \
                                      please specify one when initializing \
                                      the loader.')

                image_f, mask_f = self._one_hot_encode(image_f, mask_f)

            # Resize the augmented and preprocessed image and mask
            image_f, mask_f = self._resize_data(image_f, mask_f)

            return image_f, mask_f

        # Convert the augmentation function to a TensorFlow function
        return tf.py_function(_augmentation_func,
                              [image, mask],
                              [tf.float32, tf.uint8])


    def _file_structure_docs(self, msg):
        '''
        Function to generate file structure documentation.
        
        Arguments:
            - msg: The error message to be displayed.

        Returns:
            - doc: A formatted string containing documentation for the
                expected file structure.
        '''

        # Generate a formatted string with documentation for expected file structure
        doc = ['\nPatient #'+ x + '\n\t data (dicom) \n\t label (png)'
               for x in ['1','2','3','4', '...', 'n']]

        return msg + ''.join(doc)

    def _pre_sanity_check(self):

        '''
        Function to perform a pre-loading sanity check on the dataset.
        It checks for the correct file structure and format.

        Returns:
            - If sanity check fails: warnings/error
            - If sanity check does not fail: self.image_paths and
                self.mask_paths variables updated in objects DaL
        '''

        for i, f_structure in enumerate(os.walk(self.path)):
            root = f_structure[0]
            dirs = f_structure[1]
            files = f_structure[2]

            # List of patient directories 
            if i == 0:
                if self.verbose:

                    logging.info(', '.join(dirs))
                    logging.info(('Number of patients: '+ str(len(dirs)) + '\n'))

            # Sub directories in each patient folder
            else:

                # Check the the subdirectories for each patient folder
                if dirs:

                    # Check the number of subfolders, should be label and data
                    if len(dirs) != 2:
                        msg = (str('_' * 87) + '\nOnly two directories are \
                              expected in each patients folder, found: '
                              + str(len(dirs)) + '\n' + str('_' * 100))

                        raise ValueError(self._file_structure_docs(msg))

                    # Check the name of the directories
                    if ('label' not in dirs or 'data' not in dirs):
                        msg = 'Only directories, named data and label are \
                              allowed in each patients folder'

                        raise ValueError(self._file_structure_docs(msg))

                # Individual image files in patent subfolders
                else:

                    excluded_extensions = {'png', 'dcm', ''}
                    img_ext = not any(f.endswith(ext)
                                      for f in files
                                      for ext in excluded_extensions)

                    # Unsupported data format error
                    if img_ext:
                        logging.warning('Unsupported format (only png and dcm).')
                        raise ValueError(self._file_structure_docs())

                    logging.info((' ').join(root.split('/')[-2:]) + ': ok')

                self.image_paths.extend([os.path.join(root, f)
                                        for f in files if f.endswith('png')])

                self.mask_paths.extend([os.path.join(root, f)
                                        for f in files if f.endswith('png')])


    def _mapping_img(self):
        """
        Function to generate a list of image paths by traversing the directory structure.

        Returns:
            - img_paths: A list of image file paths.
        """

        # Use list comprehension to generate a list of image paths
        return [os.path.join(root, name)
                for root, dirs, files in os.walk(self.path)
                for name in files
                if not name.startswith('.')]

    def _data_batch(self, data, batch_size, shuffle=False):
        """
        Reads data, normalizes it, shuffles it, then batches it, returns a
        the next element in dataset op and the dataset initializer op.
        Inputs:
            batch_size: Number of images/masks in each batch returned.
            augment: Boolean, whether to augment data or not.
            shuffle: Boolean, whether to shuffle data in buffer or not.
            one_hot_encode: Boolean, whether to one hot encode the mask image or not.
                            Encoding will done according to the palette specified when
                            initializing the object.
        Returns:
            data: A tf dataset object.
        """

        if shuffle:
            # Prefetch, shuffle then batch
            data = data.prefetch(AUTOTUNE).shuffle(random.randint(0, len(self.image_paths))).batch(batch_size)
        else:
            # Batch and prefetch
            data = data.batch(batch_size).prefetch(AUTOTUNE)

        return data

    def data_read(self):
        """
        Function to perform data loading and preprocessing.

        Returns:
            - data: A TensorFlow dataset object.
        """

        # Perform pre-loading sanity check on the dataset
        self._pre_sanity_check()

        # Create dataset out of the 2 files:
        data = tf.data.Dataset.from_tensor_slices((self.image_paths, self.mask_paths))

        # Parse images and labels
        return self._data_batch(data.map(self._map_function), 100000)
