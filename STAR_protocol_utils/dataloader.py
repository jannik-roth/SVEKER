import os
import numpy as np
import pandas as pd
from rdkit.Chem import rdFingerprintGenerator, MolFromSmiles
from abc import ABC, abstractmethod

class BaseDataLoader(ABC):
    def __init__(self, path) -> None:
        super().__init__()
        self.path = path
        self._df = pd.read_csv(path)

    @abstractmethod
    def get_data(self):
        '''
        Obtain data for the Machine Learning algorithm

        return x, y, (additional_data, )

        where:
            x: np.array, shape n_samples x n_dim, binary representation of the features
            y: np.array, shape n_samples, labels (can be float if regression, or 0/1 for binary classification)
            additional_data: additional data (as tuple)

        Needs to be implemented for the specific application!
        '''
        return # x, y, (additional_info, )

class MorganFPDataLoader(BaseDataLoader):
    def __init__(self, 
                 chembl_tid: str, 
                 folder: str = os.path.join('STAR_protocol_data/'),
                 radius: int = 2,
                 size : int = 2048,
                 smiles_column_name: str = 'nonstereo_aromatic_smiles',
                 activity_column_name: str = 'label',
                 active_phrase: str = 'active') -> None:
        
        super().__init__(path=os.path.join(folder, f'{chembl_tid}.csv'))

        self._fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=size)
        
        self.smiles_column_name = smiles_column_name
        self.activity_column_name = activity_column_name
        self.active_phrase = active_phrase

    def get_data(self):
        # get fingerprints
        fps_list = [self._fpgen.GetFingerprintAsNumPy(MolFromSmiles(smiles)) for smiles in self._df[self.smiles_column_name]]
        labels = [1 if label == self.active_phrase else 0 for label in self._df[self.activity_column_name]]
        return np.stack(fps_list, dtype=np.int8), np.array(labels), (self._df[self.smiles_column_name].to_numpy(),)