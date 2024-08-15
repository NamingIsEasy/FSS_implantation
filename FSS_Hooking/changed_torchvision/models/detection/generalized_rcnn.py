"""
Implements the Generalized R-CNN framework
"""
import glob
import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn, Tensor
import torchvision
import cv2
import numpy as np

from ...utils import _log_api_usage_once


def add_dicts(dict1, dict2):
    result = {}
    for key in dict1.keys():
        if key in dict2:
            result[key] = dict1[key] + dict2[key]
        else:
            result[key] = dict1[key]
    for key in dict2.keys():
        if key not in result:
            result[key] = dict2[key]
    return result


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN.

    Args:
        backbone (nn.Module):
        rpn (nn.Module):
        roi_heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, backbone: nn.Module, rpn: nn.Module, roi_heads, transform: nn.Module,
                 support_guide) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        # used only on torchscript mode
        self._has_warned = False

        # --own test code
        self.support_guide = support_guide

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    def forward(self, images, targets=None, ref_ele_sample=None, ref_nee_sample=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `label` and `mask` (for Mask R-CNN models).

        """
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    if isinstance(boxes, torch.Tensor):
                        torch._assert(
                            len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                            f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                        )
                    else:
                        torch._assert(False, f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)
        if self.support_guide is True:
            ref_ele_images, ref_ele_targets = self.transform(ref_ele_sample[0], ref_ele_sample[1])
            ref_nee_images, ref_nee_targets = self.transform(ref_nee_sample[0], ref_nee_sample[1])

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    print(target)
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}.",
                    )

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        if self.support_guide is True:
            ref_ele_features = self.backbone(ref_ele_images.tensors)
            ref_nee_features = self.backbone(ref_nee_images.tensors)
            ref_ele_box_origin = ref_ele_targets[0]['boxes']
            ref_ele_box = [ref_ele_box_origin/4]
            ref_nee_box_origin = ref_nee_targets[0]['boxes']
            ref_nee_box = [ref_nee_box_origin/4]

        # --own test code
        if self.support_guide is True:
            ele_prototype = ref_ele_features['0'][0]
            ele_prototype_instance_feature_list = []
            for instance_index in range(0, ref_ele_targets[0]['label'].shape[0]):
                if ref_ele_targets[0]['label'][instance_index] == 1:
                    ele_prototype_mask = ref_ele_targets[0]['masks'][instance_index]
                    resized_prototype_mask = nn.AdaptiveMaxPool2d(ele_prototype.shape[-2:])(
                        ele_prototype_mask.type(torch.float).unsqueeze(0).unsqueeze(0))[0][0]
                    ele_prototype_instance_feature_list.append(torch.mean(
                        ele_prototype[:, torch.where(resized_prototype_mask != 0)[0],
                        torch.where(resized_prototype_mask != 0)[1]], dim=-1).unsqueeze(0))
            ele_prototype_instance_feature_list = torch.cat(ele_prototype_instance_feature_list, dim=0)

            nee_prototype = ref_nee_features['0'][0]
            nee_prototype_instance_feature_list = []
            for instance_index in range(0, ref_nee_targets[0]['label'].shape[0]):
                if ref_nee_targets[0]['label'][instance_index] == 2:
                    nee_prototype_mask = ref_nee_targets[0]['masks'][instance_index]
                    resized_prototype_mask = nn.AdaptiveMaxPool2d(nee_prototype.shape[-2:])(
                        nee_prototype_mask.type(torch.float).unsqueeze(0).unsqueeze(0))[0][0]
                    nee_prototype_instance_feature_list.append(torch.mean(
                        nee_prototype[:, torch.where(resized_prototype_mask != 0)[0],
                        torch.where(resized_prototype_mask != 0)[1]], dim=-1).unsqueeze(0))
            nee_prototype_instance_feature_list = torch.cat(nee_prototype_instance_feature_list, dim=0)

        proposals, proposal_losses, proposals_scores = self.rpn(images, features, targets)
        # visualize
        if not self.training:
            image_filename = target['image_filename']
            np_images = np.array(images.tensors.detach().cpu())[0]
            np_images = np.transpose(np_images, (1, 2, 0))
            for proposal_index in range(0, proposals[0].shape[0]):
                single_proposal = proposals[0][proposal_index]
                np_images[int(single_proposal[1]):int(single_proposal[3]-1), int(single_proposal[0]-1)] = 1
                np_images[int(single_proposal[1]):int(single_proposal[3]-1), int(single_proposal[2]-1)] = 1
                np_images[int(single_proposal[1]-1), int(single_proposal[0]-1):int(single_proposal[2]-1)] = 1
                np_images[int(single_proposal[3]-1), int(single_proposal[0]-1):int(single_proposal[2]-1)] = 1
            cv2.imwrite("runs/detect/" + "proposals_" + image_filename, np_images*255)

        if self.support_guide is True:
            detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets,
                                                         ref_ele_features=ref_ele_features, ref_ele_box=ref_ele_box,
                                                         ref_ele_image_shapes=ref_ele_images.image_sizes, ref_ele_targets=ref_ele_targets,
                                                         ref_nee_features=ref_nee_features, ref_nee_box=ref_nee_box,
                                                         ref_nee_image_shapes=ref_nee_images.image_sizes, ref_nee_targets=ref_nee_targets)
        else:
            detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        # visualize
        if not self.training:
            np_images = np.array(images.tensors.detach().cpu())[0]
            np_images = np.transpose(np_images, (1, 2, 0))
            for proposal_index in range(0, detections[0]['boxes'].shape[0]):
                if detections[0]['label'][proposal_index] == 1:
                    color = (0, 255, 0)
                elif detections[0]['label'][proposal_index] == 2:
                    color = (0, 0, 255)
                else:
                    raise ValueError("The predicted box label is not 1 or 2!!!")
                single_proposal = detections[0]['boxes'][proposal_index]
                line_half_width = 2
                np_images[int(single_proposal[1]-line_half_width):int(single_proposal[3]+line_half_width),
                int(single_proposal[0] - line_half_width):int(single_proposal[0] + line_half_width)] = color

                np_images[int(single_proposal[1]-line_half_width):int(single_proposal[3]+line_half_width),
                int(single_proposal[2] - line_half_width):int(single_proposal[2] + line_half_width)] = color

                np_images[int(single_proposal[1] - line_half_width):int(single_proposal[1] + line_half_width),
                int(single_proposal[0]):int(single_proposal[2])] = color

                np_images[int(single_proposal[3]-line_half_width):int(single_proposal[3]+line_half_width),
                int(single_proposal[0]):int(single_proposal[2])] = color
            cv2.imwrite("runs/detect/" + "boxes_" + image_filename, np_images*255)

        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        return losses, detections