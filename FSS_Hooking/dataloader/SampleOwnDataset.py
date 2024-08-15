# data loader
import torch
from skimage import transform
import numpy as np
from torch.utils.data import Dataset
import cv2

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


class ImageToTensor(object):
    def __call__(self, input_img):
        input_img = input_img.transpose((2, 0, 1))
        input_img = torch.from_numpy(input_img.copy())
        return input_img


class SampleOwnDataset(Dataset):
    def __init__(self, images_list, crap_method, transform_function, dataset_sir, train_val_test_mode, square_crap,
                 main_args):
        # --init dir name.
        self.dataset_dir = dataset_sir
        self.raw_images_dir = "raw_images/"
        self.mask_labels_dir = "mask_labels/"
        self.crap_method = crap_method
        self.square_crap = square_crap
        if crap_method is None:
            self.rectangular_labels_dir = None
        elif crap_method == "label":
            self.rectangular_labels_dir = "rectangular_labels/"
        elif crap_method == "yolo":
            self.rectangular_labels_dir = "yolo_detect_rectangular_labels/"
        else:
            raise ValueError("Rectangular label dir is wrong!")
        self.images_name_list = images_list
        self.train_val_test_mode = train_val_test_mode
        # --get dir length.
        self.data_length = len(images_list)
        self.transform = transform_function
        self.main_args = main_args

    def crap_image(self, rectangular_label_filename, raw_image):
        rectangular_label_file = open(rectangular_label_filename, 'r')
        electrode_needle_exist = []
        for temp_line in rectangular_label_file:
            rectangular_label = temp_line.split(" ")
            electrode_needle_exist.append(rectangular_label[0])
        rectangular_label_file.close()
        rectangular_label_file = open(rectangular_label_filename, 'r')
        if len(electrode_needle_exist) == 1:
            for temp_line in rectangular_label_file:
                rectangular_label = temp_line.split(" ")
                x_center = int(float(rectangular_label[1]) * raw_image.shape[1])
                y_center = int(float(rectangular_label[2]) * raw_image.shape[0])
                width = int(float(rectangular_label[3]) * raw_image.shape[1])
                height = int(float(rectangular_label[4][0:-1]) * raw_image.shape[0])
                if width % 32 != 0:
                    width = np.fix(width/32)*32 + 32
                if height % 32 != 0:
                    height = np.fix(height/32)*32 + 32
                if self.square_crap is True:
                    if width > height:
                        height=width
                    else:
                        width=height
                if (int(y_center - 0.5 * height)) < 0:
                    y_center = np.fix(0.5*height) + 1
                if (int(x_center - 0.5 * width)) < 0:
                    x_center = np.fix(0.5*width) + 1
                if (int(y_center + 0.5 * height)) > raw_image.shape[0]:
                    y_center = np.fix(raw_image.shape[0] - 0.5 * height) -1
                if (int(x_center + 0.5 * width)) > raw_image.shape[1]:
                    x_center = np.fix(raw_image.shape[1] - 0.5 * width) - 1
                raw_image = raw_image[int(y_center - 0.5 * height):int(y_center + 0.5 * height),
                                      int(x_center - 0.5 * width):int(x_center + 0.5 * width)]
                return raw_image, [int(y_center - 0.5 * height), int(y_center + 0.5 * height),
                                   int(x_center - 0.5 * width), int(x_center + 0.5 * width)]
        if len(electrode_needle_exist) > 1:
            rectangular_label_list = []
            for temp_line in rectangular_label_file:
                rectangular_label = temp_line.split(" ")
                x_center = int(float(rectangular_label[1]) * raw_image.shape[1])
                y_center = int(float(rectangular_label[2]) * raw_image.shape[0])
                width = int(float(rectangular_label[3]) * raw_image.shape[1])
                height = int(float(rectangular_label[4][0:-1]) * raw_image.shape[0])
                if width % 32 != 0:
                    width = np.fix(width/32)*32 + 32
                if height % 32 != 0:
                    height = np.fix(height/32)*32 + 32
                if self.square_crap is True:
                    if width > height:
                        height=width
                    else:
                        width=height
                if (int(y_center - 0.5 * height)) < 0:
                    y_center = np.fix(0.5*height) + 1
                if (int(x_center - 0.5 * width)) < 0:
                    x_center = np.fix(0.5*width) + 1
                if (int(y_center + 0.5 * height)) > raw_image.shape[0]:
                    y_center = np.fix(raw_image.shape[0] - 0.5 * height) -1
                if (int(x_center + 0.5 * width)) > raw_image.shape[1]:
                    x_center = np.fix(raw_image.shape[1] - 0.5 * width) - 1
                rectangular_label_list.append([x_center - 0.5 * width, y_center - 0.5 * height,
                                               x_center + 0.5 * width, y_center + 0.5 * height])
            rectangular_label_list = np.array(rectangular_label_list)
            raw_image = raw_image[int(rectangular_label_list.min(0)[1]):
                                  int(rectangular_label_list.max(0)[3]),
                                  int(rectangular_label_list.min(0)[0]):
                                  int(rectangular_label_list.max(0)[2]), :]
            return raw_image, [int(rectangular_label_list.min(0)[1]), int(rectangular_label_list.max(0)[3]),
                               int(rectangular_label_list.min(0)[0]), int(rectangular_label_list.max(0)[2])]

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        # --load image and label
        # the order of index is that condition1: defocus image list, focus image list, condition2: ...
        # rectangular label is [x_center y_center width height], which is the yolo format.
        assert idx < self.data_length
        raw_image_filename = (self.images_name_list[idx])
        raw_image = cv2.imread(raw_image_filename)
        raw_image_size = raw_image.shape
        rectangular_label_filename = self.dataset_dir + self.rectangular_labels_dir\
                                     + self.images_name_list[idx].split("\\")[-1].split('.png')[0] + '.txt'
        # --load mask label
        mask_label_filename = self.dataset_dir + self.mask_labels_dir + self.images_name_list[idx].split('\\')[-1]
        raw_mask_label = cv2.imread(mask_label_filename)
        # --load rectangular label and crap image
        if self.crap_method is not None:
            raw_image, roi_rectangular = self.crap_image(rectangular_label_filename=rectangular_label_filename, raw_image=raw_image)
            unresized_image_size = raw_image.shape
            raw_mask_label, roi_rectangular = self.crap_image(rectangular_label_filename=rectangular_label_filename,
                                             raw_image=raw_mask_label)
            raw_mask_label = raw_mask_label[:, :, 0]
            # cv2.imshow('1', raw_image)
            # cv2.waitKey(0)
            rectangular_label = []
            mask_label = []
            label = []
            for single_electrode_index in range(1, 9):
                single_electrode_place = np.where(raw_mask_label == single_electrode_index)
                # the label is x1, y1, x2, y2
                if len(single_electrode_place[0]) != 0:
                    # --get single electrode rectangular label
                    left = np.min(single_electrode_place[1])
                    top = np.min(single_electrode_place[0])
                    right = np.max(single_electrode_place[1])
                    bottom = np.max(single_electrode_place[0])
                    # raw_image[top:bottom, left:right] = 1
                    # cv2.imshow('1', raw_image)
                    # cv2.waitKey(0)
                    rectangular_label.append([left, top, right, bottom])
                    # --get single electrode binary label
                    temp_single_electrode_mask_image = np.zeros(
                        shape=(raw_image.shape[0], raw_image.shape[1]))
                    temp_single_electrode_mask_image[
                        single_electrode_place[0], single_electrode_place[1]] = 1
                    mask_label.append(temp_single_electrode_mask_image)
                    label.append(1)
            # the second needle is masked by 10 (synetic image)
            for needle_index in range(9, 11):
                needle_place = np.where(raw_mask_label == needle_index)
                # the label is x1, y1, x2, y2
                if len(needle_place[0]) != 0:
                    # --get single electrode rectangular label
                    left = np.min(needle_place[1])
                    top = np.min(needle_place[0])
                    right = np.max(needle_place[1])
                    bottom = np.max(needle_place[0])
                    rectangular_label.append([left, top, right, bottom])
                    # --get single electrode binary label
                    temp_needle_mask_image = np.zeros(
                        shape=(raw_image.shape[0], raw_image.shape[1]))
                    temp_needle_mask_image[
                        needle_place[0], needle_place[1]] = 1
                    mask_label.append(temp_needle_mask_image)
                    label.append(2)


        sample = {'image': raw_image, 'region': np.array(rectangular_label),
                  'mask': np.array(mask_label), 'label': np.array(label)}
        if self.transform is not None:
            if 'resize' in self.transform:
                rescale_function = Rescale(self.transform['resize'])
                sample = rescale_function(sample)
            # if len(sample['single_electrode_region'].shape) == 1:
            #     left = sample['single_electrode_region'][0]
            #     top = sample['single_electrode_region'][1]
            #     right = sample['single_electrode_region'][2]
            #     bottom = sample['single_electrode_region'][3]
            # else:
            #     left = sample['single_electrode_region'][0][0]
            #     top = sample['single_electrode_region'][0][1]
            #     right = sample['single_electrode_region'][0][2]
            #     bottom = sample['single_electrode_region'][0][3]
            # raw_image = sample['image'][top:bottom, left:right]
            # cv2.imshow('1', raw_image)
            # cv2.waitKey(0)
        toTensor = ImageToTensor()
        sample['image'] = ((toTensor(sample['image'])/255).type(torch.FloatTensor)).cuda()
        sample['region'] = (torch.from_numpy(sample['region'])).cuda()
        sample['mask'] = (torch.from_numpy(sample['mask'])).cuda()
        raw_image_filename = raw_image_filename.split('\\')[-1]
        sample['image_filename'] = raw_image_filename
        sample['train_val_test_mode'] = self.train_val_test_mode
        sample['label'] = torch.tensor(sample['label']).type(torch.int64).cuda()
        sample['roi'] = roi_rectangular
        sample['raw_image_size'] = raw_image_size
        sample['unresized_image_size'] = unresized_image_size
        return sample
