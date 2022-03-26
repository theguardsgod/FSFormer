'''The following module declares the Dataset objects required by torch to iterate over the data.'''
from enum import Enum
import glob
import pathlib

import numpy as np
import nibabel as nib
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from skimage import transform

class Task(Enum):
    '''
        Enum class for the two classification tasks
    '''
    NC_v_AD = 1
    sMCI_v_pMCI = 2

def get_ptid(path):
    '''Gets the image id from the file path string'''
    fname = path.stem
    ptid_str = ""
    #the I that comes before the id needs to be removed hence [1:]
    ptid_str = fname
    return ptid_str

def get_ptid1(path):
    '''Gets the image id from the file path string'''
    fname = path.stem
    ptid_str = ""
    #the I that comes before the id needs to be removed hence [1:]
    ptid_str = fname[5:15]
    return ptid_str


# def get_acq_year(im_data_id, im_df):
#     '''Gets the acquisition year from a pandas dataframe by searching the image id'''
#     acq_date = im_df[im_df['Image Data ID'] == im_data_id]["Acq Date"].iloc[0]
#     acq_year_str = ""

#     slash_count = 0
#     for char in acq_date:
#         if char == "/":
#             slash_count += 1

#         if slash_count == 2:
#             acq_year_str += char

#     return acq_year_str[1:]

def get_label(path, labels):
    '''Gets label from the path'''
    label_str = path.parent.stem
    label = None

    if label_str == labels[0]:
        label = np.array([0], dtype=np.double)
    elif label_str == labels[1]:
        label = np.array([1], dtype=np.double)
    return label

def get_mri(path):
    '''Gets a numpy array representing the mri object from a file path'''
    mri = nib.load(str(path)).get_fdata()
    #mri = transform.resize(mri,(148, 148, 148))
    mri = np.expand_dims(mri, axis=0)
    #print(mri.shape)
    mri = np.asarray(mri)

    return mri

def get_clinical(sub_id, clin_df):
    '''Gets clinical features vector by searching dataframe for image id'''
    clinical = np.zeros(18)

    row = clin_df.loc[clin_df["PTID"] == sub_id].iloc[0]
    
    # GENDER
    if row["PTGENDER"] == "Male":
        clinical[0] = 1
    else:
        clinical[0] = 0
    
    # AGE
    clinical[1] = row["AGE"]
    # Education
    clinical[2] = row["PTEDUCAT"]
    # APOE4
    clinical[3] = row["APOE4"]


    
    clinical[4] = row["ABETA_bl"]
    # ABETA_bl
   # if row["ABETA_bl"].empty:
    #    clinical[4] = 0
   # else:
    
    #if row["TAU_bl"].empty:
    #    clinical[5] = 0
    #else:
    clinical[5] = row["TAU_bl"]
    #if row["PTAU_bl"].empty:
    #    clinical[6] = 0
    #else:
    clinical[6] = row["PTAU_bl"]
    #if row["FDG_bl"].empty:
    #    clinical[7] = 0
    #else:
    clinical[7] = row["FDG_bl"]
    clinical[8] = row["Ventricles_bl"]
    clinical[9] = row["Hippocampus_bl"]
    clinical[10] = row["WholeBrain_bl"]
    clinical[11] = row["Entorhinal_bl"]
    clinical[12] = row["Fusiform_bl"]
    clinical[13] = row["MidTemp_bl"]
    clinical[14] = row["ICV_bl"]
    clinical[15] = row["MMSE_bl"]
    clinical[16] = row["RAVLT_immediate_bl"]
    clinical[17] = row["CDRSB_bl"]

    return clinical


class MRIDataset(Dataset):
    '''Provides an object for the MRI data that can be iterated.'''
    def __init__(self, root_dir, labels, transform=None):

        self.root_dir = root_dir  # root_dir="../data/"
        self.transform = transform
        self.directories = []
        self.len = 0
        self.labels = labels
        self.clin_data = pd.read_csv(r"./newstandADNIMERGE.csv")
    
        train_dirs = []

        for label in labels:
            train_dirs.append(root_dir + label)

        for train_dir in train_dirs:
            for path in glob.glob(train_dir + "/*"):
                self.directories.append(pathlib.Path(path))

        self.len = len(self.directories)

    def __len__(self):

        return self.len

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        repeat = True

        while repeat:
            try:
                path = self.directories[idx]
                im_id = get_ptid1(path)
                mri = get_mri(path)
                clinical = get_clinical(im_id, self.clin_data)
                

                label = get_label(path, self.labels)

                sample = {'mri': mri, 'clinical':clinical, 'label':label}
                #sample = {'mri': mri, 'label':label}
                if self.transform:
                    sample = self.transform(sample)

                return sample

            except IndexError as index_e:
                print(index_e)
                if idx < self.len:
                    idx += 1
                else:
                    idx = 0

        return sample


def minmaxscaler(data):
    min = np.amin(data)
    max = np.amax(data)    
    return (data - min)/(max-min)

class ToTensor():
    '''Convert ndarrays in sample to Tensors.'''
    def __call__(self, sample):
        image, clinical, label = sample['mri'],sample['clinical'], sample['label']
        #image, label = sample['mri'], sample['label']
        #mri_t = torch.from_numpy(minmaxscaler(image))
        mri_t = torch.from_numpy(image)
        clin_t = torch.from_numpy(clinical)
        label = torch.from_numpy(label).double()
        return {'mri': mri_t,
                'clin_t': clin_t,
                'label': label}

        # image, label = sample['mri'], sample['label']
        # #mri_t = torch.from_numpy(minmaxscaler(image))
        # mri_t = torch.from_numpy(image)
        
        # label = torch.from_numpy(label).double()
        # return {'mri': mri_t,
        #         'label': label}
