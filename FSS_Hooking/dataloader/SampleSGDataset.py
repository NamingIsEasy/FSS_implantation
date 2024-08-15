# data loader
import torch
from skimage import transform
import numpy as np
from torch.utils.data import Dataset
import cv2
import random


# data augment method.
class Rescale(object):
    def __init__(self, output_size):
        # output_size = [h,w]
        self.output_size = output_size

    def __call__(self, sample):
        # electrode region: [x1, y1, x2, y2]
        raw_image = sample['image']
        origin_h, origin_w = raw_image.shape[:2]
        raw_mask_label = sample['mask']
        single_electrode_region_list = sample['region']
        new_h, new_w = int(self.output_size[0]), int(self.output_size[1])
        raw_image = cv2.resize(raw_image, (new_h, new_w))
        if len(raw_mask_label) != 0:
            raw_mask_label = np.transpose(raw_mask_label, (1, 2, 0))
            raw_mask_label = cv2.resize(raw_mask_label, (new_h, new_w))
            if len(raw_mask_label.shape) == 2:
                raw_mask_label = np.expand_dims(raw_mask_label, -1)
            raw_mask_label = np.transpose(raw_mask_label, (2, 0, 1))
            for single_electrode_index, single_electrode_region in enumerate(single_electrode_region_list):
                single_electrode_region[0] = single_electrode_region[0]/origin_w*new_w
                single_electrode_region[1] = single_electrode_region[1]/origin_h*new_h
                single_electrode_region[2] = single_electrode_region[2]/origin_w*new_w
                single_electrode_region[3] = single_electrode_region[3]/origin_h*new_h
                single_electrode_region_list[single_electrode_index] = single_electrode_region
        sample['image'] = raw_image
        sample['mask'] = raw_mask_label
        sample['region'] = single_electrode_region_list
        return sample


class Rotation(object):
    def __init__(self, rotation_angle_range):
        self.rotation_angle_range = int(rotation_angle_range)

    def __call__(self, sample):
        raw_image = sample['raw_image']
        raw_mask_label = sample['raw_mask_label']
        rotation_angle = np.random.rand(1)[0]*(2*self.rotation_angle_range) - self.rotation_angle_range
        raw_image = transform.rotate(raw_image, rotation_angle)
        raw_mask_label = transform.rotate(raw_mask_label, rotation_angle)
        sample['raw_image'] = raw_image
        sample['raw_mask_label'] = raw_mask_label
        return sample


class RandomFlip(object):
    def __init__(self, random_flip):
        self.random_flip = random_flip

    def __call__(self, sample):
        raw_image = sample['raw_image']
        raw_mask_label = sample['raw_mask_label']
        random_value = np.random.rand(1)[0]
        if self.random_flip <= random_value:
            raw_image = raw_image[:, ::-1, :]
            raw_mask_label = raw_mask_label[:, ::-1]
        sample['raw_image'] = raw_image
        sample['raw_mask_label'] = raw_mask_label
        return sample


# dataset definition (get focus reference)
class SampleSGDataset(Dataset):
    def __init__(self, images_list, transform_function, dataset_sir, train_val_test_mode, main_args=None):
        # --init dir name.
        self.dataset_dir = dataset_sir
        self.raw_images_dir = "raw_images/"
        self.mask_labels_dir = "mask_labels/"
        self.images_name_list = images_list
        self.train_val_test_mode = train_val_test_mode
        # --get dir length.
        self.data_length = len(images_list)
        self.transform = transform_function
        self.main_args = main_args

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        batch_size = self.main_args.Batchsize
        # --load image and label
        assert idx < self.data_length
        raw_image_filename = (self.images_name_list[idx]).split('\\')[-1]
        # load images of the image class and instance class. The number of image is (batchsize +1)
        raw_images = cv2.imread(self.dataset_dir + self.raw_images_dir + raw_image_filename)
        raw_images_size = raw_images.shape
        # load masks
        mask_label_filename = self.dataset_dir + self.mask_labels_dir + raw_image_filename
        mask_label = cv2.imread(mask_label_filename)
        sample = {'images': np.transpose(raw_images, (2, 0, 1)), 'masks': np.transpose(mask_label, (2, 0, 1))}
        # if self.transform is not None:
        #     if 'resize' in self.transform:
        #         rescale_function = Rescale(self.transform['resize'])
        #         sample = rescale_function(sample)
        #     # if len(sample['single_electrode_region'].shape) == 1:
        #     #     left = sample['single_electrode_region'][0]
        #     #     top = sample['single_electrode_region'][1]
        #     #     right = sample['single_electrode_region'][2]
        #     #     bottom = sample['single_electrode_region'][3]
        #     # else:
        #     #     left = sample['single_electrode_region'][0][0]
        #     #     top = sample['single_electrode_region'][0][1]
        #     #     right = sample['single_electrode_region'][0][2]
        #     #     bottom = sample['single_electrode_region'][0][3]
        #     # raw_image = sample['image'][top:bottom, left:right]
        #     # cv2.imshow('1', raw_image)
        #     # cv2.waitKey(0)
        sample['images'] = (torch.tensor(sample['images']/255)).type(torch.FloatTensor).cuda()
        sample['masks'] = (torch.tensor(sample['masks'])).type(torch.FloatTensor).cuda()
        raw_image_filename = raw_image_filename
        sample['image_filename'] = raw_image_filename
        # sample['train_val_test_mode'] = self.train_val_test_mode
        # sample['label'] = torch.tensor(sample['label']).type(torch.int64).cuda()
        sample['raw_image_size'] = raw_images_size
        return sample
