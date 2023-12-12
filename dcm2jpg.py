#!/usr/bin/env python
# coding: utf-8

"""
Created on Thu Jun 29 2023

@author: ngyenge
"""

import dicom2jpg

dicom_dir = user_path

# convert all DICOM files in dicom_dir folder to png format
dicom2jpg.dicom2bmp(dicom_dir, multiprocessing=False)
