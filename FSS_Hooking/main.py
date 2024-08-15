import copy
import glob
import os
import random

import PIL.ImageShow
import numpy as np
import torch
torch.manual_seed(123)
torch.cuda.manual_seed(123)
from torch.utils.data import DataLoader
# origin torchvision library
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import torch.optim as optim
import cv2
import matplotlib.pyplot as plt
import argparse
from sklearn.cluster import KMeans
import time
# import dataset class
from dataloader.SampleOwnDataset import SampleOwnDataset


def load_images(raw_images_dir):
    raw_images_list = glob.glob(raw_images_dir + '*.png')
    images_list = []
    for image_filename_index, image_filename in enumerate(raw_images_list):
        temp_image_filename = image_filename.split('\\')[-1]
        images_list.append(temp_image_filename)
    return images_list

def draw_image(clarity_value_list, save_img_name, estimation_level):
    # --define color list.
    red_color = (1, 0, 0)
    green_color = (0, 1, 0)
    blue_color = (0, 0, 1)
    light_blue_color = (0, 1, 1)
    yellow_color = (1, 1, 1)
    purple_color = (180 / 255, 30 / 255, 145 / 255)
    orange_color = (49 / 255, 130 / 255, 245 / 255)
    light_purple_color = (230 / 255, 50 / 255, 240 / 255)
    color_list = [red_color, green_color, blue_color, light_blue_color,
                  yellow_color, purple_color, orange_color, light_purple_color]
    if estimation_level == "image":
        x_axis_value = np.zeros(shape=(len(clarity_value_list)))
        y_axis_value = np.zeros(shape=(len(clarity_value_list)))
        for image_index in range(0, len(clarity_value_list)):
            x_axis_value[image_index] = clarity_value_list[image_index][0]
            y_axis_value[image_index] = clarity_value_list[image_index][1]
        value_index = np.argsort(x_axis_value)
        x_axis_value = x_axis_value[value_index]
        y_axis_value = y_axis_value[value_index]
        plt.plot(x_axis_value, y_axis_value, c=red_color)
        plt.scatter(x_axis_value, y_axis_value, c=red_color)
        plt.savefig(save_img_name)
        plt.close()
    else:
        if len(color_list) < len(clarity_value_list):
            raise ValueError('Color number is less than condition number. Please add colors!')
        # draw curves.
        clarity_image_name_list = []
        total_max_point_number = -1 * len(clarity_value_list)
        electrode_max_point_list = []
        for single_electrode_index, single_electrode_name in enumerate(clarity_value_list):
            single_electrode_max_point_number = -1
            single_electrode_clarity_list = clarity_value_list[single_electrode_name]
            x_axis_value = np.zeros(shape=(len(single_electrode_clarity_list)))
            y_axis_value = np.zeros(shape=(len(single_electrode_clarity_list)))
            for distance_index in range(0, len(single_electrode_clarity_list)):
                temp_distance = float(single_electrode_clarity_list[distance_index][-1])
                x_axis_value[distance_index] = temp_distance
                y_axis_value[distance_index] = single_electrode_clarity_list[distance_index][-2]
            clarity_index = np.argmax(y_axis_value)
            clarity_image_name_list.append([y_axis_value[clarity_index], clarity_value_list[single_electrode_name][clarity_index][-1]])
            temp_color = np.zeros(shape=(len(single_electrode_clarity_list), 3))
            temp_color[:] = color_list[single_electrode_index]
            plt.plot(x_axis_value, y_axis_value, c=color_list[single_electrode_index])
            plt.scatter(x_axis_value, y_axis_value, c=temp_color)
            temp_y_axis_value = y_axis_value.copy()
            temp_y_axis_value = np.append(temp_y_axis_value, 0.0)
            for temp_y_value_index in range(1, y_axis_value.shape[0]):
                if (y_axis_value[temp_y_value_index] > temp_y_axis_value[temp_y_value_index-1]) and\
                        (y_axis_value[temp_y_value_index] > temp_y_axis_value[temp_y_value_index+1]):
                    single_electrode_max_point_number += 1
                    total_max_point_number += 1
            electrode_max_point_list.append(single_electrode_max_point_number)
        electrode_max_point_list.append(total_max_point_number)
        # --save the number of local max value of clarity curve.
        # We submit the number of each curve by 1, because We want to get the number of redundant local maxima,
        # ie: do not contain the global maxima.
        file = open((save_img_name.split('png')[0] + '.txt'), 'w')
        for temp_clarity_value in electrode_max_point_list:
            file.write(str(temp_clarity_value) + '\n')
        file.close()
        plt.savefig(save_img_name)
        plt.close()


def crap_image(main_args, input_image, rectangular_label_filename):
    rectangular_label_file = open(rectangular_label_filename, 'r')
    cropped_img_list = []
    for line in rectangular_label_file:
        temp_rectangular_label = line.split(' ')
        object_class = int(float(temp_rectangular_label[0]))
        x_center = int(float(temp_rectangular_label[-4]) * input_image.shape[1])
        y_center = int(float(temp_rectangular_label[-3]) * input_image.shape[0])
        width = int(float(temp_rectangular_label[-2]) * input_image.shape[1])
        height = int(float(temp_rectangular_label[-1]) * input_image.shape[0])
        if main_args.square_crap is True:
            if width > height:
                height = width
        rectangular_label = [y_center, x_center, height, width]
        input_image = input_image[int(rectangular_label[0] - rectangular_label[2] * 0.5):int(
            rectangular_label[0] + rectangular_label[2] * 0.5),
                      int(rectangular_label[1] - rectangular_label[3] * 0.5):int(
                          rectangular_label[1] + rectangular_label[3] * 0.5), :]
        cropped_img_list.append([object_class, input_image])
    rectangular_label_file.close()
    return cropped_img_list


def main(main_args):
    # --load dir name
    dataset_dir = main_args.dataset_dir
    raw_images_dir = 'raw_images/'
    mask_labels_dir = 'mask_labels/'
    if main_args.crap_image_method is None:
        rectangular_label_dir = 'rectangular_labels/'
    elif main_args.crap_image_method == "label":
        rectangular_label_dir = 'rectangular_labels/'
    elif main_args.crap_image_method == 'yolo':
        rectangular_label_dir = 'yolo_detect_rectangular_labels/'
    else:
        raise ValueError("You give an illegal rectangular label directory!")

    # --load support and query dataset
    support_condition_list = glob.glob(dataset_dir + 'raw_images/train/' + '*')
    query_condition_list = glob.glob(dataset_dir + 'raw_images/test/' + '*')
    ref_condition_list = glob.glob(dataset_dir + 'raw_images/train/' + '*')

    clear_image_list = ['0.0.png']
    support_image_list = []
    query_image_list = []
    ref_image_list = []
    # define train image dataset
    for temp_support_condition in support_condition_list:
        temp_support_condition_only = temp_support_condition.split('\\')[-1]
        for clear_image_filename in clear_image_list:
            temp_support_clear_image_filename = temp_support_condition + '\\' + temp_support_condition_only + "_" + clear_image_filename
            support_image_list.append(temp_support_clear_image_filename)
    support_image_dataset = main_args.sample_dataset(images_list=support_image_list,
                                                     crap_method=main_args.crap_image_method,
                                                     transform_function=None,
                                                     dataset_sir=dataset_dir,
                                                     train_val_test_mode="train/",
                                                     square_crap=main_args.square_crap,
                                                     main_args=main_args)
    # define val image dataset
    for temp_query_condition in query_condition_list:
        temp_query_condition_only = temp_query_condition.split('\\')[-1]
        for clear_image_filename in clear_image_list:
            temp_query_clear_image_filename = temp_query_condition + '\\' + temp_query_condition_only + "_" + clear_image_filename
            query_image_list.append(temp_query_clear_image_filename)
    query_images_dataset = main_args.sample_dataset(images_list=query_image_list,
                                                    crap_method=main_args.crap_image_method,
                                                    transform_function=None,
                                                    dataset_sir=dataset_dir,
                                                    train_val_test_mode="test/",
                                                    square_crap=main_args.square_crap,
                                                    main_args=main_args)
    # define ref image dataset
    for temp_ref_condition in ref_condition_list:
        temp_ref_condition_only = temp_ref_condition.split('\\')[-1]
        for clear_image_filename in clear_image_list:
            temp_ref_clear_image_filename = temp_ref_condition + '\\' + temp_ref_condition_only + "_" + clear_image_filename
            ref_image_list.append(temp_ref_clear_image_filename)
    ref_image_dataset = main_args.sample_dataset(images_list=ref_image_list,
                                                 crap_method=main_args.crap_image_method,
                                                 transform_function=None,
                                                 dataset_sir=dataset_dir,
                                                 train_val_test_mode="ref/",
                                                 square_crap=main_args.square_crap,
                                                 main_args=main_args)
    # --def collect function of Mask-RCNN.
    def own_collate_fn(sample_list):
        temp_image_list = []
        temp_label_list = []
        temp_apparatus_list = []
        for temp_sample in sample_list:
            if len(temp_sample['region']) != 0 or (temp_sample['train_val_test_mode'] != "train/"):
                temp_image_list.append(temp_sample['image'])
                temp_label = {'boxes': temp_sample['region'],
                              'masks': temp_sample['mask'],
                              'label': temp_sample['label'],
                              'image_filename': temp_sample['image_filename']}
                temp_label_list.append(temp_label)
                temp_apparatus = {'roi': temp_sample['roi'], 'raw_image_size': temp_sample['raw_image_size']}
                temp_apparatus_list.append(temp_apparatus)
        return temp_image_list, temp_label_list, temp_apparatus_list

    # --def dataloader.
    train_image_dataloader = DataLoader(support_image_dataset, batch_size=main_args.Batchsize, shuffle=True,
                                        collate_fn=own_collate_fn)
    val_images_dataloader = DataLoader(query_images_dataset, batch_size=1, shuffle=False,
                                       collate_fn=own_collate_fn)
    ref_electrode_sample = own_collate_fn([ref_image_dataset[1]])
    ref_needle_sample = own_collate_fn([ref_image_dataset[1]])
    support_guide = main_args.support_guide
    if main_args.finetune is True:
        # --def model.
        neural_network_model = maskrcnn_resnet50_fpn_v2(weights='DEFAULT', min_size=main_args.augment_process['resize'][0],
                                                        box_nms_thresh=0.0, box_score_thresh=0.05,
                                                        rpn_nms_thresh=0.5,
                                                        rpn_post_nms_top_n_train=200,
                                                        rpn_post_nms_top_n_test=50, support_guide=support_guide,
                                                        residual_rpn_mode=main_args.residual_rpn_mode,
                                                        anchor_free_cluster_loss=main_args.anchor_free_cluster_loss)
        if support_guide is True:
            box_predict_head_ch = 1
            mask_predict_head_ch = 1
        else:
            box_predict_head_ch = 3
            mask_predict_head_ch = 3
        neural_network_model.roi_heads.box_predictor = FastRCNNPredictor(1024, box_predict_head_ch, support_guide=support_guide)
        in_features_mask = neural_network_model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        neural_network_model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                                          hidden_layer, mask_predict_head_ch)
        # --load init parameters
        pretrained_dict = torch.load("init.pt")
        model_dict = neural_network_model.state_dict()
        # 1. filter out unnecessary keys
        if support_guide is True:
            delet_diction = ["roi_heads.box_predictor.bbox_pred.weight",
                             "roi_heads.box_predictor.bbox_pred.bias",
                             "roi_heads.mask_predictor.mask_fcn_logits.weight",
                             "roi_heads.mask_predictor.mask_fcn_logits.bias"]
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and (k not in delet_diction)}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        neural_network_model.load_state_dict(model_dict)

        neural_network_model.cuda()
        optimizer = optim.Adam(neural_network_model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08,
                               weight_decay=0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.5)
        # --start training.
        print("Start training.")
        best_ele_correct_number = 0
        best_nee_correct_number = 0
        for epoch in range(0, main_args.Epoch):
            neural_network_model.train()
            running_epoch_loss = 0.0
            for batch_index, sample in enumerate(train_image_dataloader):
                temp_image_list, temp_label_list = sample[0], sample[1]
                if len(temp_image_list) != 0:
                    # during training, the mask-rcnn only output losses
                    optimizer.zero_grad()
                    if support_guide is True:
                        loss, output = neural_network_model(temp_image_list, temp_label_list,
                                                        ref_electrode_sample, ref_needle_sample)
                    else:
                        loss, output = neural_network_model(temp_image_list, temp_label_list)
                    total_loss = torch.tensor([0.0]).cuda()
                    for loss in list(loss.values()):
                        total_loss += loss
                    running_epoch_loss += np.array(total_loss.detach().cpu())
                    # backward
                    total_loss.backward()
                    optimizer.step()
            print("Epoch: %3f/%3f, total loss: %3f." % ((epoch + 1), main_args.Epoch, running_epoch_loss))
            scheduler.step()

            # --eval
            if ((epoch+1) >= 150):
                neural_network_model.eval()
                total_val_electrode_number = 0
                electrode_totally_correct_number = 0
                needle_totally_correct_number = 0
                for val_batch_index, sample in enumerate(val_images_dataloader):
                    temp_image_list, temp_label_list, temp_apparatus_list = sample[0], sample[1], sample[2]
                    total_val_electrode_number += temp_label_list[0]['masks'].shape[0]
                    if support_guide is True:
                        loss, output = neural_network_model(temp_image_list, temp_label_list,
                                                        ref_electrode_sample, ref_needle_sample)
                    else:
                        loss, output = neural_network_model(temp_image_list, temp_label_list)

                    if len(output) != 0:
                        predicted_box = np.array(output[0]['boxes'].detach().cpu())
                        if predicted_box.shape[0] != 0:
                            predicted_scores = np.array(output[0]['scores'].detach().cpu())
                            predicted_label = np.array(output[0]['label'].detach().cpu())
                            predicted_mask = np.array(output[0]['masks'].detach().cpu())
                            predicted_mask[np.where(predicted_mask < 0.5)] = 0
                            # if we predict more electrode number than we have,
                            # we store the one that have the most probabilities.
                            predicted_electrode_num = (np.array(np.where(predicted_label == 1))).shape[1]
                            predicted_needle_num = (np.array(np.where(predicted_label == 2))).shape[1]
                            for time_index in range(0, 100):
                                if predicted_electrode_num > main_args.electrode_num:
                                    electrode_index = np.where(predicted_label == 1)[0]
                                    predicted_box = np.delete(predicted_box, electrode_index[-1], 0)
                                    predicted_scores = np.delete(predicted_scores, electrode_index[-1], 0)
                                    predicted_label = np.delete(predicted_label, electrode_index[-1], 0)
                                    predicted_mask = np.delete(predicted_mask, electrode_index[-1], 0)
                                    predicted_electrode_num = (np.array(np.where(predicted_label == 1))).shape[1]
                                elif predicted_needle_num > 1:
                                    needle_index = np.where(predicted_label == 2)[0]
                                    predicted_box = np.delete(predicted_box, needle_index[-1], 0)
                                    predicted_scores = np.delete(predicted_scores, needle_index[-1], 0)
                                    predicted_label = np.delete(predicted_label, needle_index[-1], 0)
                                    predicted_mask = np.delete(predicted_mask, needle_index[-1], 0)
                                    predicted_needle_num = (np.array(np.where(predicted_label == 2))).shape[1]
                                else:
                                    break

                            electrode_correct_number = 0
                            needle_correct_number = 0
                            gt_mask_list = np.array(temp_label_list[0]['masks'].detach().cpu())
                            gt_class_list = np.array(temp_label_list[0]['label'].detach().cpu())
                            electrode_mask_index = np.where(gt_class_list == 1)
                            if electrode_mask_index[0].size != 0:
                                electrode_mask = np.sum(gt_mask_list[electrode_mask_index], axis=0)
                            else:
                                electrode_mask = None
                            needle_mask_index = np.where(gt_class_list == 2)
                            if needle_mask_index[0].size != 0:
                                needle_mask = np.sum(gt_mask_list[needle_mask_index], axis=0)
                            else:
                                needle_mask = None

                            probabolity_low = 0.5
                            for gt_instance_index in range(0, gt_class_list.shape[0]):
                                gtmask_single = gt_mask_list[gt_instance_index]
                                gt_location = np.where(gtmask_single >= probabolity_low)
                                gt_center = np.array([np.mean(gt_location[0]), np.mean(gt_location[1])])
                                for predicted_instance_index in range(0, predicted_label.shape[0]):
                                    predicted_mask_single = predicted_mask[predicted_instance_index][0]
                                    predicted_location = np.where(predicted_mask_single >= probabolity_low)
                                    if not predicted_location[0].shape[0] == 0:
                                        if (electrode_mask is not None) and (
                                                predicted_label[predicted_instance_index] == 1):
                                            predicted_center = np.array([np.mean(predicted_location[0]),
                                                                         np.mean(predicted_location[1])])
                                            # if not (np.isnan(predicted_center)).any():
                                            #     if electrode_mask[int(predicted_center[0]), int(predicted_center[1])] != 0:
                                            # print(np.sum(np.abs(gt_center-predicted_center)))
                                            if np.sum(np.abs(gt_center - predicted_center)) <= main_args.electrode_error:
                                                electrode_correct_number += 1
                            for predicted_instance_index in range(0, predicted_label.shape[0]):
                                predicted_mask_single = predicted_mask[predicted_instance_index][0]
                                predicted_location = np.where(predicted_mask_single >= probabolity_low)
                                if not predicted_location[0].shape[0] == 0:
                                    if (needle_mask is not None) and (
                                            predicted_label[predicted_instance_index] == 2):
                                        # --predicted curve fit
                                        [predicted_scope, predicted_res] = np.polyfit(predicted_location[0],
                                                                                      predicted_location[1], 1)
                                        predicted_location_unique_row = np.unique(predicted_location[0])
                                        predicted_location_unique_col = (
                                                                                predicted_scope * predicted_location_unique_row) + predicted_res
                                        # --mask curve fit
                                        mask_location = np.where(needle_mask >= probabolity_low)
                                        [mask_scope, mask_res] = np.polyfit(mask_location[0], mask_location[1], 1)
                                        mask_location_unique_row = np.unique(mask_location[0])
                                        mask_location_unique_col = (
                                                                               mask_scope * mask_location_unique_row) + mask_res
                                        # --predict and curve bottom point
                                        predicted_bottom_point = [predicted_location_unique_row[-1],
                                                                  predicted_location_unique_col[-1]]
                                        mask_bottom_point = [mask_location_unique_row[-1],
                                                             mask_location_unique_col[-1]]
                                        # --calculate angle and bottom position error
                                        error_angle = np.arctan(np.abs((predicted_scope - mask_scope) / (
                                                1 + predicted_scope * mask_scope))) / 3.14 * 180
                                        error_bottom_position = np.sqrt(
                                            (int(predicted_bottom_point[0]) - int(mask_bottom_point[0])) ** 2 +
                                            (int(predicted_bottom_point[1]) - int(mask_bottom_point[1])) ** 2)
                                        if (error_angle <= 5) and (error_bottom_position <= main_args.needle_error):
                                            needle_correct_number += 1

                            # get the totally correct number.
                            if electrode_correct_number == np.max((np.array(np.where(gt_class_list == 1))).shape):
                                electrode_totally_correct_number += 1
                            if needle_correct_number == np.max((np.array(np.where(gt_class_list == 2))).shape):
                                needle_totally_correct_number += 1
                print("electrode_correct_pre_number:", electrode_totally_correct_number)
                print("needle_correct_pre_number:", needle_totally_correct_number)
                if electrode_totally_correct_number > best_ele_correct_number:
                    best_ele_correct_number = electrode_totally_correct_number
                    torch.save(neural_network_model.state_dict(), ('runs/epoch' + str(int(epoch + 1))
                                                                   + '_Loss' + str(running_epoch_loss) + '.pt'))
                if needle_totally_correct_number > best_nee_correct_number:
                    best_nee_correct_number = needle_totally_correct_number
                    torch.save(neural_network_model.state_dict(), ('runs/epoch' + str(int(epoch + 1))
                                                                   + '_Loss' + str(running_epoch_loss) + '.pt'))
        torch.save(neural_network_model.state_dict(), ('runs/final'
                                                       + '_Loss' + str(running_epoch_loss) + '.pt'))
        print("Congratulation! Training completed.")
    elif main_args.eval is True:
        # --def model.
        neural_network_model = maskrcnn_resnet50_fpn_v2(weights='DEFAULT', min_size=main_args.augment_process['resize'][0],
                                                        box_nms_thresh=0.0, box_score_thresh=0.05,
                                                        rpn_nms_thresh=0.5,
                                                        # rpn_score_thresh=0.5,
                                                        rpn_post_nms_top_n_train=200,
                                                        rpn_post_nms_top_n_test=50, support_guide=support_guide,
                                                        residual_rpn_mode=main_args.residual_rpn_mode,
                                                        anchor_free_cluster_loss=main_args.anchor_free_cluster_loss)
        if support_guide is True:
            box_predict_head_ch = 1
            mask_predict_head_ch = 1
        else:
            box_predict_head_ch = 3
            mask_predict_head_ch = 3
        neural_network_model.roi_heads.box_predictor = FastRCNNPredictor(1024, box_predict_head_ch, support_guide=support_guide)
        in_features_mask = neural_network_model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        neural_network_model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                                          hidden_layer, mask_predict_head_ch)
        # load dict
        pretrained_dict = torch.load(main_args.saved_model_filename)
        model_dict = neural_network_model.state_dict()
        not_used_layer = []
        missing_keys = [k for k in model_dict if k not in pretrained_dict and k not in not_used_layer]
        unexpected_keys = [k for k in pretrained_dict if k not in model_dict]
        print("missing keys: ", missing_keys)
        print("unexpected keys: ", unexpected_keys)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        neural_network_model.load_state_dict(model_dict)

        neural_network_model.cuda()
        neural_network_model.eval()
        right_thread_prediction = 0
        total_val_electrode_number = 0
        detect_time = 0
        for val_batch_index, sample in enumerate(val_images_dataloader):
            if len(sample[0]) != 0:
                temp_image_list, temp_label_list, temp_apparatus_list = sample[0], sample[1], sample[2]
                temp_image = np.array(temp_image_list[0].detach().cpu())
                temp_image = np.transpose(temp_image, (1, 2, 0))
                temp_mask_label_image = np.zeros((temp_apparatus_list[0]['raw_image_size'][0],
                                                  temp_apparatus_list[0]['raw_image_size'][1]))
                temp_roi_mask_label_image = np.zeros_like(temp_image)[:, :, 0]
                # cv2.imshow('1', temp_image)
                # cv2.waitKey(0)
                total_val_electrode_number += temp_label_list[0]['masks'].shape[0]
                start_time = time.time()
                if support_guide is True:
                    loss, output = neural_network_model(temp_image_list, temp_label_list,
                                                        ref_electrode_sample, ref_needle_sample)
                else:
                    loss, output = neural_network_model(temp_image_list, temp_label_list)
                detect_time += (time.time()) - start_time

                predicted_box = np.array(output[0]['boxes'].detach().cpu())
                predicted_scores = np.array(output[0]['scores'].detach().cpu())
                predicted_label = np.array(output[0]['label'].detach().cpu())
                predicted_mask = np.array(output[0]['masks'].detach().cpu())
                predicted_mask[np.where(predicted_mask < 0.5)] = 0
                # if we predict more electrode number than we have,
                # we store the one that have the most probabilities.
                predicted_electrode_num = (np.array(np.where(predicted_label == 1))).shape[1]
                predicted_needle_num = (np.array(np.where(predicted_label == 2))).shape[1]
                for time_index in range(0, 100):
                    if predicted_electrode_num > main_args.electrode_num:
                        electrode_index = np.where(predicted_label == 1)[0]
                        predicted_box = np.delete(predicted_box, electrode_index[-1], 0)
                        predicted_scores = np.delete(predicted_scores, electrode_index[-1], 0)
                        predicted_label = np.delete(predicted_label, electrode_index[-1], 0)
                        predicted_mask = np.delete(predicted_mask, electrode_index[-1], 0)
                        predicted_electrode_num = (np.array(np.where(predicted_label == 1))).shape[1]
                    elif predicted_needle_num > 1:
                        needle_index = np.where(predicted_label == 2)[0]
                        predicted_box = np.delete(predicted_box, needle_index[-1], 0)
                        predicted_scores = np.delete(predicted_scores, needle_index[-1], 0)
                        predicted_label = np.delete(predicted_label, needle_index[-1], 0)
                        predicted_mask = np.delete(predicted_mask, needle_index[-1], 0)
                        predicted_needle_num = (np.array(np.where(predicted_label == 2))).shape[1]
                    else:
                        break

                # electrode_time = 1
                # for single_index in range(0, predicted_mask.shape[0]):
                #     # --visualize
                #     # single_box = predicted_box[single_index]-1
                #     # temp_image[int(single_box[1]):int(single_box[3]), int(single_box[0])] = (0, 0, 255)
                #     # temp_image[int(single_box[1]):int(single_box[3]), int(single_box[2])] = (0, 0, 255)
                #     # temp_image[int(single_box[1]), int(single_box[0]):int(single_box[2])] = (0, 0, 255)
                #     # temp_image[int(single_box[3]), int(single_box[0]):int(single_box[2])] = (0, 0, 255)
                #
                #     # cv2.imshow('1', temp_image)
                #     # cv2.waitKey(0)
                #     single_mask = predicted_mask[single_index][0]
                #     single_mask = (single_mask - np.min(single_mask)) / (
                #             np.max(single_mask) - np.min(single_mask) + 0.00000001)
                #     if predicted_label[single_index] == 2:
                #         image_color = (0, 0, 1)
                #         mask_color = 9
                #     else:
                #         image_color = (0, 1, 0)
                #         mask_color = electrode_time
                #         electrode_time += 1
                #     for row_index in range(0, single_mask.shape[0]):
                #         for col_index in range(0, single_mask.shape[1]):
                #             if single_mask[row_index, col_index] != 0:
                #                 temp_image[row_index, col_index] = image_color
                #                 temp_roi_mask_label_image[row_index, col_index] = mask_color
                    # cv2.imshow('1', temp_image)
                    # cv2.waitKey(0)
                raw_image_filename = temp_label_list[0]['image_filename']
                cv2.imwrite(filename=("runs/detect/" + raw_image_filename), img=temp_image*255)
                # roi_rectangular = temp_apparatus_list[0]['roi']
                # temp_mask_label_image[roi_rectangular[0]:roi_rectangular[1],
                #                       roi_rectangular[2]:roi_rectangular[3]] = temp_roi_mask_label_image
                # cv2.imwrite(filename=("runs/detect/mask_" + raw_image_filename), img=temp_mask_label_image)

                # save txt file that includes the correct number.
                electrode_correct_number = 0
                electrode_totally_correct_number = 0
                needle_correct_number = 0
                needle_totally_correct_number = 0
                gt_mask_list = np.array(temp_label_list[0]['masks'].detach().cpu())
                gt_class_list = np.array(temp_label_list[0]['label'].detach().cpu())
                electrode_mask_index = np.where(gt_class_list == 1)
                if electrode_mask_index[0].size != 0:
                    electrode_sum_mask = np.sum(gt_mask_list[electrode_mask_index], axis=0)
                else:
                    electrode_sum_mask = None
                needle_mask_index = np.where(gt_class_list == 2)
                if needle_mask_index[0].size != 0:
                    needle_sum_mask = np.sum(gt_mask_list[needle_mask_index], axis=0)
                else:
                    needle_sum_mask = None

                probabolity_low = 0.5
                for gt_instance_index in range(0, gt_mask_list.shape[0]):
                    gtmask_single = gt_mask_list[gt_instance_index]
                    gt_location = np.where(gtmask_single >= probabolity_low)
                    gt_center = np.array([np.mean(gt_location[0]), np.mean(gt_location[1])])
                    for predicted_instance_index in range(0, predicted_label.shape[0]):
                        predicted_mask_single = predicted_mask[predicted_instance_index][0]
                        predicted_location = np.where(predicted_mask_single >= probabolity_low)
                        if not predicted_location[0].shape[0] == 0:
                            if (electrode_sum_mask is not None) and (predicted_label[predicted_instance_index] == 1):
                                predicted_center = np.array([np.mean(predicted_location[0]),
                                                            np.mean(predicted_location[1])])
                                if np.sum(np.abs(gt_center-predicted_center)) <= main_args.electrode_error:
                                    electrode_correct_number += 1
                                    # --visualize
                                    temp_image[int(gt_center[0])-10:int(gt_center[0])+10,
                                                    int(gt_center[1])-2:int(gt_center[1])+2] = np.array([0, 1, 0])
                                    temp_image[int(gt_center[0])-2:int(gt_center[0])+2,
                                                    int(gt_center[1])-10:int(gt_center[1])+10] = np.array([0, 1, 0])
                                    # temp_image[int(predicted_center[0])-10:int(predicted_center[0])+10,
                                    #                 int(predicted_center[1])-2:int(predicted_center[1])+2] = np.array([0, 1, 0])
                                    # temp_image[int(predicted_center[0])-2:int(predicted_center[0])+2,
                                    #                 int(predicted_center[1])-10:int(predicted_center[1])+10] = np.array([0, 1, 0])

                print(electrode_correct_number)
                for predicted_instance_index in range(0, predicted_label.shape[0]):
                    predicted_mask_single = predicted_mask[predicted_instance_index][0]
                    predicted_location = np.where(predicted_mask_single >= probabolity_low)
                    if not predicted_location[0].shape[0] == 0:
                        if (needle_sum_mask is not None) and (predicted_label[predicted_instance_index] == 2):
                            # --predicted curve fit
                            [predicted_scope, predicted_res] = np.polyfit(predicted_location[0], predicted_location[1], 1)
                            predicted_location_unique_row = np.unique(predicted_location[0])
                            predicted_location_unique_col = (predicted_scope * predicted_location_unique_row) + predicted_res
                            # --mask curve fit
                            mask_location = np.where(needle_sum_mask >= probabolity_low)
                            [mask_scope, mask_res] = np.polyfit(mask_location[0], mask_location[1], 1)
                            mask_location_unique_row = np.unique(mask_location[0])
                            mask_location_unique_col = (mask_scope * mask_location_unique_row) + mask_res
                            # --predict and curve bottom point
                            predicted_bottom_point = [predicted_location_unique_row[-1], predicted_location_unique_col[-1]]
                            mask_bottom_point = [mask_location_unique_row[-1], mask_location_unique_col[-1]]
                            # --calculate angle and bottom position error
                            error_angle = np.arctan(np.abs((predicted_scope - mask_scope) / (
                                        1 + predicted_scope * mask_scope))) / 3.14 * 180
                            error_bottom_position = np.sqrt((int(predicted_bottom_point[0])-int(mask_bottom_point[0]))**2 +
                                                            (int(predicted_bottom_point[1])-int(mask_bottom_point[1]))**2)
                            # --visualize
                            temp_image = np.transpose(temp_image, (1, 0, 2))
                            for point_index in range(0, mask_location_unique_col.shape[0]):
                                # temp_image[int(predicted_location_unique_col[point_index])-2:
                                #            int(predicted_location_unique_col[point_index])+2,
                                #            int(predicted_location_unique_row[point_index])] = (0, 0, 1)
                                temp_image[int(mask_location_unique_col[point_index])-2:int(mask_location_unique_col[point_index])+2,
                                mask_location_unique_row[point_index]] = (0, 0, 1)
                            temp_image = np.transpose(temp_image, (1, 0, 2))
                            # cv2.imshow("1", temp_temp_image)
                            # cv2.waitKey(0)
                            for row_index in range(-10, 10):
                                # temp_image[int(predicted_bottom_point[0]) + row_index,
                                #            int(predicted_bottom_point[1])-2:int(predicted_bottom_point[1])+2] = np.array([0, 1, 1])
                                temp_image[int(mask_bottom_point[0]) + row_index,
                                           int(mask_bottom_point[1])-2:int(mask_bottom_point[1])+2] = np.array([0, 1, 1])
                            for col_index in range(-10, 10):
                                # temp_image[int(predicted_bottom_point[0])-2:int(predicted_bottom_point[0])+2,
                                #            int(predicted_bottom_point[1])+col_index] = np.array([0, 1, 1])
                                temp_image[int(mask_bottom_point[0])-2:int(mask_bottom_point[0])+2,
                                           int(mask_bottom_point[1]) + col_index] = np.array([0, 1, 1])
                            if (error_angle <= 5) and (error_bottom_position <= main_args.needle_error):
                                needle_correct_number += 1
                            print("angle:", error_angle)
                            print("position:", error_bottom_position)
                cv2.imwrite("runs/detect/" + raw_image_filename.split(".png")[0]
                            + "_needle_detect" + '.png', temp_image * 255)
                # get the totally correct number.
                if electrode_correct_number == 8:
                    electrode_totally_correct_number += 1
                if (needle_correct_number != 1) and (needle_correct_number != 0):
                    raise ValueError("!!!!!!!!")
                if needle_correct_number == np.max((np.array(np.where(gt_class_list == 2))).shape):
                    needle_totally_correct_number += 1
                # save txt file that includes correct number
                file = open(("runs/detect/" + raw_image_filename.split(".png")[0]
                             + "_condition_1_high_light_0.0" + '.txt'),
                            'w')
                file.write("support class:" + str(1) + "\n")
                file.write("correct number:" + str(electrode_correct_number) + "\n")
                file.write("totally correct number:" + str(electrode_totally_correct_number) + "\n")
                file.close()
                file = open(("runs/detect/" + raw_image_filename.split(".png")[0]
                             + "_condition_7_high_light_0.0" + '.txt'),
                            'w')
                file.write("support class:" + str(2) + "\n")
                file.write("correct number:" + str(needle_correct_number) + "\n")
                file.write("totally correct number:" + str(needle_totally_correct_number) + "\n")
                file.close()
        print("detection image number: ", val_batch_index+1)
        print("detection time: ", detect_time)

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Locate micro-electrode and needle.")
    parser.add_argument("--electrode_num", default=8)
    parser.add_argument("--dataset_dir", default='../dataset/multi_thread_flexible_electrode/')
    parser.add_argument("--crap_image_method", default="label", help='label means using rectangular to crap,'
                                                                     'yolo means using yolo detect result to crap')
    parser.add_argument("--square_crap", default=True, help="crap image as square.")

    # --proposed method
    parser.add_argument("--residual_rpn_mode", default='1x', help="None, '1x', '2x', '3x', '2x_share', '3x_share'."
                                                                  "Proposed method uses 1x.")
    parser.add_argument("--anchor_free_cluster_loss", default=True, help="True or False")

    # -- train (eval) neural network setting
    parser.add_argument("--finetune", default=False, help="train neural network")
    parser.add_argument("--eval", default=True, help="Only evaluation, not including training.")
    parser.add_argument("--Batchsize", default=4, help="The batch size of training neural network."
                                                       "The evaluation batch is 1.")
    parser.add_argument("--Epoch", default=300)
    parser.add_argument("--augment_process", default={'resize': [641, 641]})
    parser.add_argument("--saved_model_filename", default="saved_model/4th/4th.pt")
    parser.add_argument("--sample_dataset", default=SampleOwnDataset, help="SampleOwnDataset -- OWN,"
                                                                           "SampleROCMDataset -- ROCM")
    # evaluation metric
    parser.add_argument("--electrode_error", default=10, help="uint is pixels")
    parser.add_argument("--needle_error", default=10, help="uint is pixels")
    # other try
    parser.add_argument("--support_guide", default=False, help="use fixed anchor to guide segment")
    args = parser.parse_args()
    main(main_args=args)
