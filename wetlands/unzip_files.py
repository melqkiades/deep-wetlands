import zipfile
import glob
import os
import shutil
# import py7zr
from unidecode import unidecode
from pathlib import Path


def list_files_walk(start_path='.'):
    for root, dirs, files in os.walk(start_path):
        for directory in dirs:
            # if 'wetlands_new' in directory:
            print(directory)

def list_files_pathlib(path=Path('.')):
    for entry in path.iterdir():
        if entry.is_file():
            # print(entry)
            subpath, filename = os.path.split(entry)
            if unidecode(filename) != filename:
                os.rename(entry, os.path.join(subpath, unidecode(filename)))
        elif entry.is_dir():
            list_files_pathlib(entry)
            dirname = entry.parts[-1]
            if unidecode(dirname) != dirname:
                os.rename(entry, os.path.join(entry.parent, unidecode(dirname)))

def main():
    # list_files_walk('/mimer/NOBACKUP/groups/deep-wetlands-data-2025')
    list_files_walk('/cephyr/users/ioannisi')
    # for file in glob.glob(
    #         '/mimer/NOBACKUP/groups/deep-wetlands-data-2025/data/sar_tiles/Orebro lan_2020-05-30_64x64/*.png'):
    #     os.remove(file)
    # for file in glob.glob(
    #         '/mimer/NOBACKUP/groups/deep-wetlands-data-2025/data/ndwi_masks_tiles/Orebro lan_2018-07-04_64x64/*.png'):
    #     os.remove(file)
    # # REMOVE ACCENTS FROM ALL DATA IN DATA FOLDER
    # list_files_pathlib(Path('/mimer/NOBACKUP/groups/deep-wetlands-data-2025/data/ndwi_masks_tiles/Örebro län_2018-07-04_224x224/'))
    # shutil.move('/mimer/NOBACKUP/groups/deep-wetlands-data-2025/old_data/sar/bulk_export_svartadalen_sar/',
    #             '/mimer/NOBACKUP/groups/deep-wetlands-data-2025/data/sar/bulk_export_svartadalen_sar/')
    # LOOK INTO BIG ZIP
    # with zipfile.ZipFile('/mimer/NOBACKUP/groups/deep-wetlands-data-2025/data.zip', 'r') as zip:
    #     for file in zip.namelist():
    #         if file.startswith('data/sar/'):
    #             zip.extract(file, '/mimer/NOBACKUP/groups/deep-wetlands-data-2025/')
        # zip.extract('data/models/', '/mimer/NOBACKUP/groups/deep-wetlands-data-2025/SU_data/models/')
        # zip.printdir()
    # REMOVE ACCENTS FROM FILENAMES
    # directory_path = '/mimer/NOBACKUP/groups/deep-wetlands-data-2025/old_data/sar_tiles'
    # for dir in glob.glob(directory_path + '/*'):
    #     os.rename(dir, unidecode(dir))
    # list_files_walk(directory_path)
    # directory_path = '/mimer/NOBACKUP/groups/deep-wetlands-data-2025/old_data/ndwi_masks_tiles'
    # for dir in glob.glob(directory_path + '/*'):
    #     os.rename(dir, unidecode(dir))
    # list_files_walk(directory_path)
    # for file in glob.glob(
    #     '/mimer/NOBACKUP/groups/deep-wetlands-data-2025/old_data/sar_tiles/Örebro län_2018-07-11_64x64/*.png'):
    #     os.remove(file)
    # for file in glob.glob(
    #     '/mimer/NOBACKUP/groups/deep-wetlands-data-2025/old_data/sar_tiles/Örebro län_2020-06-18_64x64/*.png'):
    #     os.remove(file)
    # for file in glob.glob(
    #     '/mimer/NOBACKUP/groups/deep-wetlands-data-2025/old_data/sar_tiles/Örebro län_2020-06-30_64x64/*.png'):
    #     os.remove(file)
    # for file in glob.glob(
    #     '/mimer/NOBACKUP/groups/deep-wetlands-data-2025/old_data/sar_tiles/Örebro län_2018-06-29_64x64/*.png'):
    #     os.remove(file)
    # with py7zr.SevenZipFile('/mimer/NOBACKUP/groups/deep-wetlands-data-2025/su_data/manual_annotations.7z', 'r') as zip_ref:
    #     zip_ref.extractall('/mimer/NOBACKUP/groups/deep-wetlands-data-2025/su_data')
    # with py7zr.SevenZipFile('C:\\Users\\anubi\\PycharmProjects\\deep-wetlands-cephyr\\manual_annotations.7z',
    #                         'r') as zip_ref:
    #     zip_ref.extractall('C:\\Users\\anubi\\PycharmProjects\\deep-wetlands-cephyr')
    # shutil.rmtree('/mimer/NOBACKUP/groups/deep-wetlands-data-2025/old_data/ndwi_masks_tiles/Örebro län_2020-06-23_64x64_wrong/')
    # print('deleting')
    # with zipfile.ZipFile('/mimer/NOBACKUP/groups/deep-wetlands-data-2025/old_data/sar_tiles/Örebro län_2020-06-23_64x64.zip', 'r') as zip_ref:
    #     zip_ref.extractall('/mimer/NOBACKUP/groups/deep-wetlands-data-2025/old_data/sar_tiles/')
    # for file in glob.glob('/mimer/NOBACKUP/groups/deep-wetlands-data-2025/old_data/sar_tiles/Örebro län_2020-06-23_64x64/*.png'):
    #     os.remove(file)
    # print('deleting2')
    # with zipfile.ZipFile('/mimer/NOBACKUP/groups/deep-wetlands-data-2025/old_data/ndwi_masks_tiles/Örebro län_2020-06-23_64x64.zip', 'r') as zip_ref:
    #     zip_ref.extractall('/mimer/NOBACKUP/groups/deep-wetlands-data-2025/old_data/ndwi_masks_tiles/')
    # for file in glob.glob('/mimer/NOBACKUP/groups/deep-wetlands-data-2025/old_data/ndwi_masks_tiles/Örebro län_2020-06-23_64x64/*.png'):
    #     os.remove(file)
    # print('deleting3')
    # with zipfile.ZipFile('/mimer/NOBACKUP/groups/deep-wetlands-data-2025/old_data/manual_annotations/deepaqua_test_dataset_no_nov.zip', 'r') as zip_ref:
    #     zip_ref.extractall('/mimer/NOBACKUP/groups/deep-wetlands-data-2025/old_data/manual_annotations/')
    # print('end')

# main()