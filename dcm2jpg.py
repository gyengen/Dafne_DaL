import dicom2jpg

dicom_dir = user_path

# convert all DICOM files in dicom_dir folder to png format
dicom2jpg.dicom2bmp(dicom_dir, multiprocessing=False)