import os
from cnnClassifier.entity import *
from cnnClassifier.utils import *
from cnnClassifier import logger
from urllib import request
from zipfile import ZipFile
from tqdm import tqdm
from pathlib import Path

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    
    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url = self.config.source_URL,
                filename = self.config.local_data_file)
            logger.info(f'{filename} downloaded with following info: \n{headers}')
        else:
            logger.info(f'File already exists with size : {get_size(Path(self.config.local_data_file))}')
    
    def _get_updated_list_of_files(self,list_of_files):
        return [f for f in list_of_files if f.endswith('.jpg') and ('Cat' in f or 'Dog' in f)]
    
    def _preprocess(self, zf: ZipFile, f: str, working_dir: str):
        target_file_path = os.path.join(working_dir, f)
        if not os.path.exists(target_file_path):
            zf.extract(f, working_dir)
        
        if os.path.getsize(target_file_path) == 0:
            os.remove(target_file_path)
    
    def extract_zipfile(self):
        '''
        zip_file_path : str
        Extracts the zip file into data directory path
        Function returns None
        '''
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with ZipFile(self.config.local_data_file, 'r') as zip_ref:
            list_of_files = zip_ref.namelist()
            updated_list_of_files = self._get_updated_list_of_files(list_of_files)
            for f in tqdm(updated_list_of_files):
                self._preprocess(zip_ref, f, unzip_path)