import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import random
import pandas as pd
import torchvision.transforms.v2.functional as F_v2
from torchvision.io import decode_image
from torch.utils.data import Sampler
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

class MegaWeightedRandomSampler(Sampler):
    """
    Bypasses PyTorch's 2^24 multinomial limit by using 64-bit Numpy for CDF generation.
    """
    def __init__(self, weights, num_samples, replacement=True):
        # Convert to 64-bit float and normalize to sum to 1.0 (required by numpy)
        self.weights = np.array(weights, dtype=np.float64)
        self.weights /= self.weights.sum()
        self.num_samples = num_samples
        self.replacement = replacement

    def __iter__(self):
        # Generate all indices simultaneously in highly-optimized C
        indices = np.random.choice(
            len(self.weights), 
            size=self.num_samples, 
            replace=self.replacement, 
            p=self.weights
        )
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples
    
def get_balanced_loader(datasets_dict: dict[str, torch.utils.data.Dataset], batch_size: int, num_workers: int, epoch_size: int = None):
    """

    This function returns a dataloader such that each task will have an equal probability of being chosen. 

    Args:
        datasets (dict): A dictionary of datasets for each task. 
                              Each dataset must have a get_sample_weight method defined.
        batch_size (int): How many samples per batch to load.
        num_workers (int): How many subprocesses to use for data loading.
        epoch_size (int): How many images to sample per epoch. 
    
    Returns:
        dataloader (torch.utils.data.DataLoader): A dataloader with balanced loading of the datasets.                          
    """
    # 1. get the sample weights per task and concatenate the datasets of each task.
    task_weight = 1 / len(datasets_dict) # assign equal weight for each task.
    task_sample_weights = [] # accumulate sample weights per task here.
    task_datasets = [] # accumulate concatenated datasets per task here.
    for task_name in datasets_dict.keys():
        if len(datasets_dict[task_name]) == 1: # for tasks with one dataset
            task_datasets.append(datasets_dict[task_name][0])
            task_sample_weights.append(datasets_dict[task_name][0].get_sample_weights() * task_weight)
        else: # for tasks with more than one dataset
            task_datasets.append(ConcatDataset(datasets_dict[task_name])) # concatenate the task's datasets and add them
            
            # concatenate the sample weights of each dataset within this task, then multiply by the task_weight / the number of datasets in this task 
            task_sample_weights.append(np.concatenate([dataset.get_sample_weights() for dataset in datasets_dict[task_name]]) \
                                             * (task_weight / len(datasets_dict[task_name]))) # distribute the task weight equally among the datasets of this task.


    # 2. concatenate the datasets of each task and the sample weights
    unified_dataset = ConcatDataset(task_datasets)
    sample_weights = np.concatenate(list(task_sample_weights))
    
    # 3. Define Epoch Size
    # If we don't set this, an 'epoch' is just the sum of all raw lengths
    # But we likely want to define a 'Virtual Epoch' to check validation often.
    if epoch_size is None:
        epoch_size = len(unified_dataset)
        
    # 4. Create the Sampler
    sampler = MegaWeightedRandomSampler(
        weights=sample_weights,
        num_samples=epoch_size,
        replacement=True # Crucial: Allows picking samples multiple times
    )
    
    # 5. Create Loader
    # Note: shuffle must be False when using a sampler! It already shuffles everything nicely.
    loader = DataLoader(
        unified_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler,
        pin_memory=True,
        persistent_workers=True
    )
    
    return loader



class BaseDatasetClass(torch.utils.data.Dataset):
    """
        The base dataset class for all datasets
    """

    def __init__(self, dataset_dir, transform = None, return_name = False):
        super().__init__()

        self.images_dir = os.path.join(dataset_dir, 'Images')
        self.labels_df = pd.read_csv(os.path.join(dataset_dir, 'labels.csv'))
        self.transform = transform
        self.return_name = return_name

    def __getitem__(self, idx):
        raise NotImplementedError
    
    def get_sample_weights(self):
        raise NotImplementedError
    
    def __len__(self):
        return len(self.labels_df)

    # used for getting the default labels for each task and, optionally, the name of the database
    def get_default_labels(self):
        label = { # the default label is -1 because putting someting like None won't work in the torch.utils.data.DataLoader, and using something like float('nan') won't work with integer labels.
            'face_recognition' : -1,
            'emotion' : -1,
            'smooth_emotion' : torch.ones(7, dtype = torch.float32) * -1,
            'smooth_emotion_is_logit' : False,
            'age' : -1,
            'gender' : -1,
            'race' : -1,
            'attributes' : torch.ones(40, dtype = torch.int32) * -1, # place -1 for all the 40 attributes
            'pose' : torch.ones(3, dtype = torch.float32) * -999, # place -999 for all the 3 angles. We can't place -1 because it could be a valid angle.
            'landmarks_98' : torch.ones(98*2, dtype = torch.float32) * -1, # place -1 for all visibility landmarks
            'valid_mask_98' : torch.ones(98, dtype = torch.float32) * -1, # place -1 for all 98 mask labels
            'landmarks_68' : torch.ones(68*2, dtype = torch.float32) * -1, # place -1 for all landmarks
            'valid_mask_68' : torch.ones(68, dtype = torch.float32) * -1, # place -1 for all 68 mask label
            'landmarks_29' : torch.ones(29*2, dtype = torch.float32) * -1, # place -1 for all visibility landmarks
            'visibility' : torch.ones(29, dtype = torch.float32) * -1 # place -1 for all visibility labels
        }
        if self.return_name:
            label['dataset_name'] = self.__class__.__name__
        return label



###################################

#       Face Recognition

###################################

class FaceRecognitionClass(BaseDatasetClass):
    """
        The base class for all face recognition datasets.
    """

    def __init__(self, dataset_dir, transform = None, **kwargs):
        super().__init__(dataset_dir, transform, **kwargs)
    
    def number_of_classes(self):
        return len(self.labels_df['label'].unique())
    
    def __getitem__(self, idx):
        filename = self.labels_df['filename'][idx]
        image_path = os.path.join(self.images_dir, filename)
        image = decode_image(image_path, mode = torchvision.io.image.ImageReadMode.RGB)
        
        if self.transform:
            image = self.transform(image)

        label = self.get_default_labels()
        label['face_recognition'] = self.labels_df['label'][idx]

        return image, label

    def get_sample_weights(self):
        """
            Face recognition datasets will not be balanced.
            This method returns an equal weight for each sample to be combined in the get_balanced_loader method.
        """
        sample_weights = np.ones(len(self.labels_df))
        return sample_weights / sample_weights.sum()

class Glint360k(FaceRecognitionClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'face recognition', 'glint360k'), transform = None, **kwargs):
        super().__init__(dataset_dir, transform, **kwargs)

class MS1MV2(FaceRecognitionClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'face recognition', 'ms1mv2'), transform = None, **kwargs):
        super().__init__(dataset_dir, transform, **kwargs)

class VGGFace(FaceRecognitionClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'face recognition', 'VGG-Face'), transform = None, **kwargs):
        super().__init__(dataset_dir, transform, **kwargs)

class CasiaWebFace(FaceRecognitionClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'face recognition', 'Casia webface'), transform = None, **kwargs):
        super().__init__(dataset_dir, transform, **kwargs)



###################################

#       Face Verification

###################################

class FaceVerificationClass(BaseDatasetClass):
    """
        The base class for all face verification datasets.
    """
    def __init__(self, dataset_dir, transform = None):
        super().__init__(dataset_dir, transform)
    
    def __getitem__(self, idx):
        image1 = self.labels_df['image1'][idx]
        image2 = self.labels_df['image2'][idx]
        same = self.labels_df['same'][idx]

        image1 = decode_image(os.path.join(self.images_dir, image1), mode = torchvision.io.image.ImageReadMode.RGB)
        image2 = decode_image(os.path.join(self.images_dir, image2), mode = torchvision.io.image.ImageReadMode.RGB)

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return (image1, image2), same

class CFPFP(FaceVerificationClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'face recognition', 'cfp_fp'), transform = None):
        super().__init__(dataset_dir, transform)

class CFPFF(FaceVerificationClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'face recognition', 'cfp_ff'), transform = None):
        super().__init__(dataset_dir, transform)

class CALFW(FaceVerificationClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'face recognition', 'calfw'), transform = None):
        super().__init__(dataset_dir, transform)

class CPLFW(FaceVerificationClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'face recognition', 'cplfw'), transform = None):
        super().__init__(dataset_dir, transform)

class LFW(FaceVerificationClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'face recognition', 'lfw'), transform = None):
        super().__init__(dataset_dir, transform)

class AgeDB30(FaceVerificationClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'face recognition', 'agedb30'), transform = None):
        super().__init__(dataset_dir, transform)

class VGG2FP(FaceVerificationClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'face recognition', 'vgg2_fp'), transform = None):
        super().__init__(dataset_dir, transform)



###################################

#       Emotion Recognition

###################################

class AffectNet(BaseDatasetClass):

    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'emotion recognition', 'AffectNet'), subset = 'combined', transform = None, **kwargs):
        super().__init__(dataset_dir, transform, **kwargs)

        if subset != None: # Either train, test, or combined (train + test).
            if subset != 'combined':
                self.labels_df = self.labels_df[self.labels_df['split'] == subset]

            self.labels_df.reset_index(drop = True, inplace = True)

    
    def __getitem__(self, idx):
        filename = self.labels_df['filename'][idx]
        image_path = os.path.join(self.images_dir, filename)
        image = decode_image(image_path, mode = torchvision.io.image.ImageReadMode.RGB)

        if self.transform:
            image = self.transform(image)

        label = self.get_default_labels()
        label['emotion'] = self.labels_df['label'][idx]
        
        smooth_emotion = torch.tensor(np.array(self.labels_df.iloc[idx][['0', '1', '2', '3', '4', '5', '6']], dtype = np.float32))
        label['smooth_emotion'] = smooth_emotion
        label['smooth_emotion_is_logit'] = True
        return image, label

    def get_sample_weights(self):
        """
            Returns the weight of each sample in this dataset such that it balances the classes.
            Classes with low samples get a higher weight.
            The final weight must sum to 1.
        """

        label_counts = self.labels_df['label'].value_counts().sort_index()
        class_weights = len(self.labels_df) / label_counts
        sample_weights = self.labels_df['label'].map(class_weights).to_numpy()
        return sample_weights / sample_weights.sum()


class RAFDB(BaseDatasetClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'emotion recognition', 'RAF_DB'), subset = 'train', transform = None, **kwargs):
        super().__init__(dataset_dir, transform, **kwargs)
        if subset != None: 
            self.labels_df = self.labels_df[self.labels_df['split'] == subset]
            self.labels_df.reset_index(drop = True, inplace = True)
    
    
    def __getitem__(self, idx):
        filename = self.labels_df['filename'][idx]
        image_path = os.path.join(self.images_dir, filename)
        image = decode_image(image_path, mode = torchvision.io.image.ImageReadMode.RGB)

        if self.transform:
            image = self.transform(image)

        label = self.get_default_labels()
        label['emotion'] = self.labels_df['label'][idx]
        
        smooth_emotion = torch.tensor(np.array(self.labels_df.iloc[idx][['0', '1', '2', '3', '4', '5', '6']], dtype = np.float32))
        label['smooth_emotion'] = smooth_emotion
        label['smooth_emotion_is_logit'] = True
        return image, label

    def get_sample_weights(self):
        """
            Returns the weight of each sample in this dataset such that it balances the classes.
            Classes with low samples get a higher weight.
            The final weight must sum to 1.
        """

        label_counts = self.labels_df['label'].value_counts().sort_index()
        class_weights = len(self.labels_df) / label_counts
        sample_weights = self.labels_df['label'].map(class_weights).to_numpy()
        return sample_weights / sample_weights.sum()


class FERPlus(BaseDatasetClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'emotion recognition', 'FERPlus'), subset = 'combined', transform = None, **kwargs):
        super().__init__(dataset_dir, transform, **kwargs)

        if subset != None: # Either train, test, or combined (train + test).
            if subset != 'combined':
                self.labels_df = self.labels_df[self.labels_df['split'] == subset]

            self.labels_df.reset_index(drop = True, inplace = True)
    
    
    def __getitem__(self, idx):
        filename = self.labels_df['filename'][idx]
        image_path = os.path.join(self.images_dir, filename)
        image = decode_image(image_path, mode = torchvision.io.image.ImageReadMode.RGB)

        if self.transform:
            image = self.transform(image)

        label = self.get_default_labels()
        smooth_emotion = torch.tensor(np.array(self.labels_df.iloc[idx][['0', '1', '2', '3', '4', '5', '6']], dtype = np.float32))
        label['smooth_emotion'] = smooth_emotion / sum(smooth_emotion)
        label['smooth_emotion_is_logit'] = False
        label['emotion'] = torch.argmax(smooth_emotion).item()

        return image, label

    def get_sample_weights(self):
        """
            Returns the weight of each sample in this dataset such that it balances the classes.
            Classes with low samples get a higher weight.
            The final weight must sum to 1.
        """

        label_counts = self.labels_df['label'].value_counts().sort_index()
        class_weights = len(self.labels_df) / label_counts
        sample_weights = self.labels_df['label'].map(class_weights).to_numpy()
        return sample_weights / sample_weights.sum()


###################################

#       Age, Gender, and Race Recognition

###################################

class AgeDB(BaseDatasetClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'age gender and race estimation', 'AgeDB'), transform = None, **kwargs):
        super().__init__(dataset_dir, transform, **kwargs)
    
    def __getitem__(self, idx):
        filename = self.labels_df['filename'][idx]
        image = decode_image(os.path.join(self.images_dir, filename), mode = torchvision.io.image.ImageReadMode.RGB)

        if self.transform:
            image = self.transform(image)
        
        age = self.labels_df['age'][idx]
        gender = self.labels_df['gender'][idx]

        label = self.get_default_labels()
        label['age'] = age
        label['gender'] = gender

        return image, label
    
    def get_sample_weights(self):
        """
            Returns the weight of each sample such that it balances the males and females. Age is not balanced
            The final weight must sum to 1.
        """

        label_counts = self.labels_df['gender'].value_counts().sort_index()
        class_weights = len(self.labels_df) / label_counts
        sample_weights = self.labels_df['gender'].map(class_weights).to_numpy()
        return sample_weights / sample_weights.sum()



class MORPH(BaseDatasetClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'age gender and race estimation', 'MORPH'), transform = None, subset = 'combined', **kwargs):
        super().__init__(dataset_dir, transform, **kwargs)
        if subset != None: # Can be 'train', 'test', or 'validation'
            if subset == 'combined':
                self.labels_df = pd.concat([self.labels_df[self.labels_df['split'] == 'train'], self.labels_df[self.labels_df['split'] == 'validation']])
            else:
                self.labels_df = self.labels_df[self.labels_df['split'] == subset]
            self.labels_df.reset_index(drop = True, inplace = True)

    def __getitem__(self, index):
        filename = self.labels_df['filename'][index]
        image = decode_image(os.path.join(self.images_dir, filename), mode = torchvision.io.image.ImageReadMode.RGB)

        if self.transform:
            image = self.transform(image)

        age = self.labels_df['age'][index]
        gender = self.labels_df['gender'][index]
        label = self.get_default_labels()
        label['age'] = age
        label['gender'] = gender

        return image, label

    def get_sample_weights(self):
        """
            Returns the weight of each sample such that it balances the males and females.
            The final weight must sum to 1.
        """

        label_counts = self.labels_df['gender'].value_counts().sort_index()
        class_weights = len(self.labels_df) / label_counts
        sample_weights = self.labels_df['gender'].map(class_weights).to_numpy()
        return sample_weights / sample_weights.sum()


class UTKFace(BaseDatasetClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'age gender and race estimation', 'UTKFace'), transform = None, subset = None, **kwargs):
        super().__init__(dataset_dir, transform, **kwargs)
        if subset != None: # Can be either 'train' or 'test'
            self.labels_df = self.labels_df[self.labels_df['split'] == subset]
            self.labels_df.reset_index(drop = True, inplace = True)
    
    def __getitem__(self, idx):
        filename = self.labels_df['filename'][idx]
        image = decode_image(os.path.join(self.images_dir, filename), mode = torchvision.io.image.ImageReadMode.RGB)
        
        if self.transform:
            image = self.transform(image)

        age = self.labels_df['age'][idx]
        gender = self.labels_df['gender'][idx]
        race = self.labels_df['race'][idx]
        label = self.get_default_labels()
        label['age'] = age
        label['gender'] = gender
        label['race'] = race

        return image, label
    
    def get_sample_weights(self):
        """
            Returns the weight of each sample such that it balanced both the race and the gender.
            The final weight must sum to 1.
        """

        gender_label_counts = self.labels_df['gender'].value_counts().sort_index()
        gender_class_weights = len(self.labels_df) / gender_label_counts
        gender_sample_weights = self.labels_df['gender'].map(gender_class_weights).to_numpy()

        race_label_counts = self.labels_df['race'].value_counts().sort_index()
        race_class_weights = len(self.labels_df) / race_label_counts
        race_sample_weights = self.labels_df['race'].map(race_class_weights).to_numpy()

        sample_weights = gender_sample_weights * race_sample_weights
        return sample_weights / sample_weights.sum()




class FairFace(BaseDatasetClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'age gender and race estimation', 'FairFace'), transform = None, subset = None, **kwargs):
        super().__init__(dataset_dir, transform, **kwargs)
        if subset != None: # Can be either 'train' or 'test'
            self.labels_df = self.labels_df[self.labels_df['split'] == subset]
            self.labels_df.reset_index(drop = True, inplace = True)

        # Age mapping
        self.age_map = {
            '0-2': 0,
            '3-9': 1,
            '9-Mar': 1,   # Catches the Excel bug for 3-9
            '10-19': 2,
            '19-Oct': 2,  # Catches the Excel bug for 10-19
            '20-29': 3,
            '30-39': 4,
            '40-49': 5,
            '50-59': 6,
            '60-69': 7,
            '70+': 8,
            'more than 70': 8 # Sometimes appears in variants of the dataset
        } 

    def __getitem__(self, idx):
        filename = self.labels_df['filename'][idx]
        image = decode_image(os.path.join(self.images_dir, filename), mode = torchvision.io.image.ImageReadMode.RGB)

        if self.transform:
            image = self.transform(image)
        gender = self.labels_df['gender'][idx]
        race = self.labels_df['race'][idx]
        label = self.get_default_labels()
        label['gender'] = gender
        label['race'] = race
        # label['age_group'] = self.age_map[self.labels_df['age range'][idx]]

        return image, label 

    def get_sample_weights(self):
        """
            Returns the weight of each sample such that it balances both the race and the gender.
            The final weight must sum to 1.
        """

        gender_label_counts = self.labels_df['gender'].value_counts().sort_index()
        gender_class_weights = len(self.labels_df) / gender_label_counts
        gender_sample_weights = self.labels_df['gender'].map(gender_class_weights).to_numpy()

        race_label_counts = self.labels_df['race'].value_counts().sort_index()
        race_class_weights = len(self.labels_df) / race_label_counts
        race_sample_weights = self.labels_df['race'].map(race_class_weights).to_numpy()

        sample_weights = gender_sample_weights * race_sample_weights
        return sample_weights / sample_weights.sum()


class IMDB_WIKI(BaseDatasetClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'age gender and race estimation', 'IMDB_WIKI'), transform = None, **kwargs):
        super().__init__(dataset_dir, transform, **kwargs)
    
    def __getitem__(self, idx):
        filename = self.labels_df['filename'][idx]
        image = decode_image(os.path.join(self.images_dir, filename), mode = torchvision.io.image.ImageReadMode.RGB)

        if self.transform:
            image = self.transform(image)

        gender = self.labels_df['gender'][idx]
        age = self.labels_df['age'][idx]

        label = self.get_default_labels()
        label['gender'] = gender
        label['age'] = age

        return image, label

    def get_sample_weights(self):
        """
            Returns the weight of each sample such that it balances the gender.
            The final weight must sum to 1.
        """
        label_counts = self.labels_df['gender'].value_counts().sort_index()
        class_weights = len(self.labels_df) / label_counts
        sample_weights = self.labels_df['gender'].map(class_weights).to_numpy()
        return sample_weights / sample_weights.sum()



###################################

#       Attribute recognition

###################################

class CelebA(BaseDatasetClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'attribute recognition', 'CelebA'), transform = None, subset = 'combined', **kwargs):
        super().__init__(dataset_dir, transform, **kwargs)
        if subset != None: # Can be 'train', 'test', or 'validation'
            if subset == 'combined':
                self.labels_df = pd.concat([self.labels_df[self.labels_df['split'] == 'train'], self.labels_df[self.labels_df['split'] == 'validation']])
            else:
                self.labels_df = self.labels_df[self.labels_df['split'] == subset]
            self.labels_df.reset_index(drop = True, inplace = True)

        self.attribute_groups = {
            'Whole' : ['Attractive', 'Blurry', 'Chubby', 'Heavy_Makeup', 'Oval_Face', 'Pale_Skin', 'Male', 'Young'],
            'Hair' : ['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair', 'Receding_Hairline', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Hat'],
            'Eyes' : ['Arched_Eyebrows', 'Bags_Under_Eyes', 'Bushy_Eyebrows', 'Eyeglasses', 'Narrow_Eyes'],
            'Nose' : ['Big_Nose', 'Pointy_Nose'],
            'Cheek' : ['High_Cheekbones', 'Rosy_Cheeks', 'Wearing_Earrings', 'Sideburns'],
            'Mouth' : ['5_o_Clock_Shadow', 'Big_Lips', 'Mouth_Slightly_Open', 'Mustache', 'Wearing_Lipstick', 'No_Beard', 'Smiling', 'Goatee'],
            'Neck' : ['Wearing_Necklace', 'Wearing_Necktie', 'Double_Chin']
        }
    
    def get_cut_indices(self):
        cut_indices = [0]
        for key, value in self.attribute_groups.items():
            cut_indices.append(len(value) + cut_indices[-1])
        
        return cut_indices

    def __getitem__(self, idx):
        filename = self.labels_df['filename'][idx]
        image = decode_image(os.path.join(self.images_dir, filename), mode = torchvision.io.image.ImageReadMode.RGB)

        if self.transform:
            image = self.transform(image)

        label = self.get_default_labels()

        # Convert -1 to 0
        raw_attrs = self.labels_df.iloc[idx, 1:-1].values
        label['attributes'] = torch.tensor((raw_attrs > 0).astype(np.uint8), dtype=torch.uint8)

        return image, label
    
    def get_sample_weights(self):
        """
            Balancing this dataset is handled using the loss function.
            This method returns an equal weight for each sample to be combined in the get_balanced_loader method.
        """
        sample_weights = np.ones(len(self.labels_df))
        return sample_weights / sample_weights.sum()
    

    def get_attribute_weights(self):
            """
            Returns the pos_weight for BCEWithLogitsLoss.
            Formula: Negative_Count / Positive_Count
            """
            # Count positives. Replace -1 with 0 for the negative.
            labels = self.labels_df.iloc[:, 1:-1].replace(-1, 0)
            
            pos_counts = labels.sum()
            total_counts = len(self.labels_df)
            neg_counts = total_counts - pos_counts
            
            # Calculate Ratio (Neg / Pos)
            # Add epsilon to avoid division by zero
            pos_weights = neg_counts / (pos_counts + 1e-6)
            
            # Return as tensor
            return torch.tensor(pos_weights.values, dtype=torch.float32)



###################################

#       Head Pose Estimation

###################################

class W300LP(BaseDatasetClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'head pose estimation', '300W_LP'), transform = None, **kwargs):
        super().__init__(dataset_dir, transform, **kwargs)
    
    def __getitem__(self, idx):
        filename = self.labels_df['filename'][idx]
        image = decode_image(os.path.join(self.images_dir, filename), mode = torchvision.io.image.ImageReadMode.RGB)
        label = self.get_default_labels()
        label['pose'] = torch.tensor(np.array(self.labels_df.iloc[idx, 1:].values, dtype = np.float32), dtype = torch.float32)

        if self.transform:
            if random.random() > 0.5: # 50% chance to rotate
                angle = random.uniform(-45.0, 45.0)
                
                # Rotate the image tensor
                image = F_v2.rotate(image, angle)
                
                label['pose'][0] -= torch.tensor(angle, dtype = torch.float32)

            image = self.transform(image)

        return image, label
    
    def get_sample_weights(self):
        """
            This dataset will not be balanced since it represents a continuous range of angles for the head pose.
            This method returns an equal weight for each sample to be combined in the get_balanced_loader method.
        """
        sample_weights = np.ones(len(self.labels_df))
        return sample_weights / sample_weights.sum()


class BIWI(BaseDatasetClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'head pose estimation', 'BIWI'), transform = None, **kwargs):
        super().__init__(dataset_dir, transform, **kwargs)
    
    def __getitem__(self, idx):
        filename = self.labels_df['filename'][idx]
        image = decode_image(os.path.join(self.images_dir, filename), mode=torchvision.io.image.ImageReadMode.RGB)

        if self.transform:
            image = self.transform(image)

        label = self.get_default_labels()
        label['pose'] = torch.tensor(np.array(self.labels_df.iloc[idx, 1:].values, dtype = np.float32), dtype = torch.float32)

        return image, label
    
    def get_sample_weights(self):
        """
            This dataset will not be balanced since it represents a continuous range of angles for the head pose.
            This method returns an equal weight for each sample to be combined in the get_balanced_loader method.
        """
        sample_weights = np.ones(len(self.labels_df))
        return sample_weights / sample_weights.sum()


#####################

# Landmark detection

#####################

class W300(BaseDatasetClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'landmark detection', '300W'), transform = None, subset = 'train', **kwargs):
        """
            This is a dataset class for the 300W dataset.
            The subset can be either 'train', 'full' (testing), 'common' (testing), or 'challenge' (testing).
        """
        super().__init__(dataset_dir, transform, **kwargs)

        if subset == 'train':
            self.labels_df = self.labels_df[self.labels_df['split'] == subset]
            self.labels_df.reset_index(drop = True, inplace = True)
            self.subset = 'train'

        elif subset == 'full':
            self.labels_df = self.labels_df[np.logical_or(self.labels_df['test_type'] == 'common',  self.labels_df['test_type'] == 'challenge')]
            self.labels_df.reset_index(drop = True, inplace = True)
            self.subset = 'test'

        elif subset == 'common' or subset == 'challenge':
            self.labels_df = self.labels_df[self.labels_df['test_type'] == subset]
            self.labels_df.reset_index(drop = True, inplace = True)
            self.subset = 'test'


    def __getitem__(self, idx):
        filename = self.labels_df['filename'][idx]
        image = decode_image(os.path.join(self.images_dir, filename), mode = torchvision.io.image.ImageReadMode.RGB)

        landmarks = np.array(self.labels_df.iloc[idx, 1:-2].values, dtype = np.float32)
        landmarks = landmarks.reshape(-1, 2)

        if self.transform and self.subset == 'train':
            image, valid_mask_68 = self.apply_dynamic_erasing(image, landmarks)
            image = image.permute(1, 2, 0).numpy()
            augmented = self.transform(image = image, keypoints = landmarks)
            image = augmented['image']
            landmarks = np.array(augmented['keypoints'], dtype = np.float32)

        elif self.transform and self.subset == 'test':
            image = self.transform(image)


        label = self.get_default_labels()
        label['landmarks_68'] = torch.clamp(torch.tensor(landmarks.reshape(-1), dtype = torch.float32) / 224, min = 0.0, max = 1.0) #  clamp the landmarks to be between 0 and 1
        
        if self.subset == 'train':
            label['valid_mask_68'] = valid_mask_68

        return image, label

    def apply_dynamic_erasing(self, image, landmarks_68, p=0.15, scale=(0.02, 0.10)):
        """
        image: Tensor of shape (C, H, W)
        landmarks_68: Tensor of shape (68, 2) containing (x, y) coordinates
        """
        _, H, W = image.shape
        
        # Create a mask of 1s (all landmarks valid by default)
        valid_mask = torch.ones(68, device=image.device)
        
        if random.random() > p:
            return image, valid_mask # Skip erasing this time
            
        # Calculate random box dimensions
        area = H * W
        target_area = random.uniform(scale[0], scale[1]) * area
        aspect_ratio = random.uniform(0.3, 3.3)
        
        h = int(round((target_area * aspect_ratio) ** 0.5))
        w = int(round((target_area / aspect_ratio) ** 0.5))
        
        if w < W and h < H:
            y1 = random.randint(0, H - h)
            x1 = random.randint(0, W - w)
            y2 = y1 + h
            x2 = x1 + w
            
            # Apply the erasing (fill with random noise)
            image[:, y1:y2, x1:x2] = torch.rand(image.shape[0], h, w, device=image.device)
            
            # landmarks_68[:, 0] is X, landmarks_68[:, 1] is Y
            erased_idx = (landmarks_68[:, 0] >= x1) & (landmarks_68[:, 0] <= x2) & \
                        (landmarks_68[:, 1] >= y1) & (landmarks_68[:, 1] <= y2)
                        
            # Set the mask to 0 for erased landmarks
            valid_mask[erased_idx] = 0.0
            
        return image, valid_mask

    def get_sample_weights(self):
        """
            This dataset will not be balanced since it represents a continuous range of points for the landmarks.
            This method returns an equal weight for each sample to be combined in the get_balanced_loader method.
        """
        sample_weights = np.ones(len(self.labels_df))
        return sample_weights / sample_weights.sum()


class WFLW(BaseDatasetClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'landmark detection', 'WFLW'), transform = None, subset = 'train', **kwargs):
        super().__init__(dataset_dir, transform, **kwargs)
        self.labels_df = self.labels_df[self.labels_df['split'] == subset]
        self.labels_df.reset_index(drop = True, inplace = True)
        self.subset = subset
    
    def __getitem__(self, idx):
        filename = self.labels_df['filename'][idx]
        image = decode_image(os.path.join(self.images_dir, filename), mode = torchvision.io.image.ImageReadMode.RGB)

        landmarks = np.array(self.labels_df.iloc[idx, 1:-1].values, dtype = np.float32)
        landmarks = landmarks.reshape(-1, 2)

        if self.transform and self.subset == 'train':
            image, valid_mask_98 = self.apply_dynamic_erasing(image, landmarks)
            image = image.permute(1, 2, 0).numpy()
            augmented = self.transform(image = image, keypoints = landmarks)
            image = augmented['image']
            landmarks = np.array(augmented['keypoints'], dtype = np.float32)
        elif self.transform and self.subset == 'test':
            image = self.transform(image)


        label = self.get_default_labels()
        label['landmarks_98'] = torch.clamp(torch.tensor(landmarks.reshape(-1), dtype = torch.float32) / 224, min = 0.0, max = 1.0) # clamp the landmarks to be between 0 and 1

        if self.subset == 'train':
            label['valid_mask_98'] = valid_mask_98

        return image, label

    def apply_dynamic_erasing(self, image, landmarks_98, p=0.15, scale=(0.02, 0.10)):
        """
        image: Tensor of shape (C, H, W)
        landmarks_98: Tensor of shape (98, 2) containing (x, y) coordinates
        """
        _, H, W = image.shape

        # Create a mask of 1s (all landmarks valid by default)
        valid_mask = torch.ones(98, device=image.device)

        if random.random() > p:
            return image, valid_mask # Skip erasing this time
            
        # Calculate random box dimensions
        area = H * W
        target_area = random.uniform(scale[0], scale[1]) * area
        aspect_ratio = random.uniform(0.3, 3.3)

        h = int(round((target_area * aspect_ratio) ** 0.5))
        w = int(round((target_area / aspect_ratio) ** 0.5))

        if w < W and h < H:
            y1 = random.randint(0, H - h)
            x1 = random.randint(0, W - w)
            y2 = y1 + h
            x2 = x1 + w
            
            # Apply the erasing (fill with random noise)
            image[:, y1:y2, x1:x2] = torch.rand(image.shape[0], h, w, device=image.device)
            
            # landmarks_98[:, 0] is X, landmarks_98[:, 1] is Y
            erased_idx = (landmarks_98[:, 0] >= x1) & (landmarks_98[:, 0] <= x2) & \
                            (landmarks_98[:, 1] >= y1) & (landmarks_98[:, 1] <= y2)
                            
            # Set the mask to 0 for erased landmarks
            valid_mask[erased_idx] = 0.0
            
        return image, valid_mask

    def get_sample_weights(self):
        """
            This dataset will not be balanced since it represents a continuous range of points for the landmarks.
            This method returns an equal weight for each sample to be combined in the get_balanced_loader method.
        """
        sample_weights = np.ones(len(self.labels_df))
        return sample_weights / sample_weights.sum()


#####################

# Visibility Detection

#####################

class COFW(BaseDatasetClass):
    def __init__(self, dataset_dir = os.path.join('data', 'datasets', 'visibility detection', 'COFW'), transform = None, subset = 'train', **kwargs):
        super().__init__(dataset_dir, transform, **kwargs)
        self.labels_df = self.labels_df[self.labels_df['split'] == subset]
        self.labels_df.reset_index(drop = True, inplace = True)
        self.subset = subset
    
    def __getitem__(self, idx):
            filename = self.labels_df['filename'][idx]
            image = decode_image(os.path.join(self.images_dir, filename), mode = torchvision.io.image.ImageReadMode.RGB)
            
            # Reshape to (29, 2) for Albumentations
            landmarks = np.array(self.labels_df.iloc[idx, 1:1 + 29*2].values, dtype = np.float32).reshape(-1, 2)
            visibility = np.array(self.labels_df.iloc[idx, 1 + 29*2 : 1 + 29*3].values, dtype = np.float32)
            
            if self.transform and self.subset == 'train':
                image = image.permute(1, 2, 0).numpy()

                augmented = self.transform(image = image, keypoints = landmarks)
                image = augmented['image']
                landmarks = np.array(augmented['keypoints'], dtype = np.float32)

            elif self.transform and self.subset == 'test':
                image = self.transform(image)

            label = self.get_default_labels()
            label['landmarks_29'] = torch.clamp(torch.tensor(landmarks.reshape(-1), dtype = torch.float32) / 224, min = 0.0, max = 1.0)
            label['visibility'] = torch.tensor(visibility, dtype = torch.float32)

            return image, label


    def get_sample_weights(self):
        """
            This dataset will not be balanced since it represents a continuous range of points for the visibility.
            This method returns an equal weight for each sample to be combined in the get_balanced_loader method.
        """
        sample_weights = np.ones(len(self.labels_df))
        return sample_weights / sample_weights.sum()

    def get_pos_weight(self):
        """
            Returns the pos_weight for BCEWithLogitsLoss.
            Formula: Negative_Count / Positive_Count
            """
        # 1. Count positives.
        visibility_labels = self.labels_df.iloc[:, 1 + 29*2 : 1 + 29*3]
        
        visible_counts = (visibility_labels == 0).sum()
        occluded_counts = (visibility_labels == 1).sum()
        # 2. Calculate Ratio
        pos_weights = visible_counts / (occluded_counts + 1e-6)  # Add epsilon to avoid division by zero
        
        
        # Return as tensor
        return torch.tensor(pos_weights.values, dtype=torch.float32)