#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import math
import torch


class IOUloss:
    """ Calculate IoU loss.
    """
    def __init__(self, box_format='xywh', iou_type='ciou', reduction='none', eps=1e-7):
        """ Setting of the class.
        Args:
            box_format: (string), must be one of 'xywh' or 'xyxy'.
            iou_type: (string), can be one of 'ciou', 'diou', 'giou' or 'siou'
            reduction: (string), specifies the reduction to apply to the output, must be one of 'none', 'mean','sum'.
            eps: (float), a value to avoid devide by zero error.
        """
        self.box_format = box_format
        self.iou_type = iou_type.lower()
        self.reduction = reduction
        self.eps = eps

    def __call__(self, box1, box2):
        """ calculate iou. box1 and box2 are torch tensor with shape [M, 4] and [Nm 4].
        """
        box2 = box2.T
        if self.box_format == 'xyxy':
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
        elif self.box_format == 'xywh':
            b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
            b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
            b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
            b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

        # Intersection area
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

        # Union Area
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + self.eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + self.eps
        union = w1 * h1 + w2 * h2 - inter + self.eps
        iou = inter / union

        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if self.iou_type == 'giou':
            c_area = cw * ch + self.eps  # convex area
            iou = iou - (c_area - union) / c_area
        elif self.iou_type in ['diou', 'ciou']:
            c2 = cw ** 2 + ch ** 2 + self.eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if self.iou_type == 'diou':
                iou = iou - rho2 / c2
            elif self.iou_type == 'ciou':
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + self.eps))
                iou = iou - (rho2 / c2 + v * alpha)
        elif self.iou_type == 'siou':
            # SIoU Loss https://arxiv.org/pdf/2205.12740.pdf
            s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5
            s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5
            sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
            sin_alpha_1 = torch.abs(s_cw) / sigma
            sin_alpha_2 = torch.abs(s_ch) / sigma
            threshold = pow(2, 0.5) / 2
            sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
            angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
            rho_x = (s_cw / cw) ** 2
            rho_y = (s_ch / ch) ** 2
            gamma = angle_cost - 2
            distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
            omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
            omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
            shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
            iou = iou - 0.5 * (distance_cost + shape_cost)
        loss = 1.0 - iou

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss


def pairwise_bbox_iou(box1, box2, box_format='xywh'):
    """Calculate iou.
    This code is based on https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/utils/boxes.py
    """
    if box_format == 'xyxy':
        lt = torch.max(box1[:, None, :2], box2[:, :2])
        rb = torch.min(box1[:, None, 2:], box2[:, 2:])
        area_1 = torch.prod(box1[:, 2:] - box1[:, :2], 1)
        area_2 = torch.prod(box2[:, 2:] - box2[:, :2], 1)

    elif box_format == 'xywh':
        lt = torch.max(
            (box1[:, None, :2] - box1[:, None, 2:] / 2),
            (box2[:, :2] - box2[:, 2:] / 2),
        )
        rb = torch.min(
            (box1[:, None, :2] + box1[:, None, 2:] / 2),
            (box2[:, :2] + box2[:, 2:] / 2),
        )

        area_1 = torch.prod(box1[:, 2:], 1)
        area_2 = torch.prod(box2[:, 2:], 1)
    valid = (lt < rb).type(lt.type()).prod(dim=2)
    inter = torch.prod(rb - lt, 2) * valid
    return inter / (area_1[:, None] + area_2 - inter)


# SAHI
import logging

import numpy as np

from sahi.prediction import ObjectPrediction
from sahi.utils.import_utils import is_available
from sahi.utils.torch import is_torch_cuda_available
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
from typing import Any, Dict, List, Optional

from models.common import DetectMultiBackend, AutoShape
from utils.torch_utils import select_device

logger = logging.getLogger(__name__)


class DetectionModel:
    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[Any] = None,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
        mask_threshold: float = 0.5,
        confidence_threshold: float = 0.3,
        category_mapping: Optional[Dict] = None,
        category_remapping: Optional[Dict] = None,
        load_at_init: bool = True,
        image_size: int = None,
    ):
        """
        Init object detection/instance segmentation model.
        Args:
            model_path: str
                Path for the instance segmentation model weight
            config_path: str
                Path for the mmdetection instance segmentation model config file
            device: str
                Torch device, "cpu" or "cuda"
            mask_threshold: float
                Value to threshold mask pixels, should be between 0 and 1
            confidence_threshold: float
                All predictions with score < confidence_threshold will be discarded
            category_mapping: dict: str to str
                Mapping from category id (str) to category name (str) e.g. {"1": "pedestrian"}
            category_remapping: dict: str to int
                Remap category ids based on category names, after performing inference e.g. {"car": 3}
            load_at_init: bool
                If True, automatically loads the model at initalization
            image_size: int
                Inference input size.
        """
        self.model_path = model_path
        self.config_path = config_path
        self.model = None
        self.device = device
        self.mask_threshold = mask_threshold
        self.confidence_threshold = confidence_threshold
        self.category_mapping = category_mapping
        self.category_remapping = category_remapping
        self.image_size = image_size
        self._original_predictions = None
        self._object_prediction_list_per_image = None

        # automatically set device if its None
        if not (self.device):
            self.device = "cuda:0" if is_torch_cuda_available() else "cpu"

        # automatically load model if load_at_init is True
        if load_at_init:
            if model:
                self.set_model(model)
            else:
                self.load_model()

    def load_model(self):
        """
        This function should be implemented in a way that detection model
        should be initialized and set to self.model.
        (self.model_path, self.config_path, and self.device should be utilized)
        """
        raise NotImplementedError()

    def set_model(self, model: Any, **kwargs):
        """
        This function should be implemented to instantiate a DetectionModel out of an already loaded model
        Args:
            model: Any
                Loaded model
        """
        raise NotImplementedError()

    def unload_model(self):
        """
        Unloads the model from CPU/GPU.
        """
        self.model = None
        if is_available("torch"):
            from sahi.utils.torch import empty_cuda_cache

            empty_cuda_cache()

    def perform_inference(self, image: np.ndarray):
        """
        This function should be implemented in a way that prediction should be
        performed using self.model and the prediction result should be set to self._original_predictions.
        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted.
        """
        raise NotImplementedError()

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
        full_shape_list: Optional[List[List[int]]] = None,
    ):
        """
        This function should be implemented in a way that self._original_predictions should
        be converted to a list of prediction.ObjectPrediction and set to
        self._object_prediction_list. self.mask_threshold can also be utilized.
        Args:
            shift_amount_list: list of list
                To shift the box and mask predictions from sliced image to full sized image, should
                be in the form of List[[shift_x, shift_y],[shift_x, shift_y],...]
            full_shape_list: list of list
                Size of the full image after shifting, should be in the form of
                List[[height, width],[height, width],...]
        """
        raise NotImplementedError()

    def _apply_category_remapping(self):
        """
        Applies category remapping based on mapping given in self.category_remapping
        """
        # confirm self.category_remapping is not None
        if self.category_remapping is None:
            raise ValueError("self.category_remapping cannot be None")
        # remap categories
        for object_prediction_list in self._object_prediction_list_per_image:
            for object_prediction in object_prediction_list:
                old_category_id_str = str(object_prediction.category.id)
                new_category_id_int = self.category_remapping[old_category_id_str]
                object_prediction.category.id = new_category_id_int

    def convert_original_predictions(
        self,
        shift_amount: Optional[List[int]] = [0, 0],
        full_shape: Optional[List[int]] = None,
    ):
        """
        Converts original predictions of the detection model to a list of
        prediction.ObjectPrediction object. Should be called after perform_inference().
        Args:
            shift_amount: list
                To shift the box and mask predictions from sliced image to full sized image, should be in the form of [shift_x, shift_y]
            full_shape: list
                Size of the full image after shifting, should be in the form of [height, width]
        """
        self._create_object_prediction_list_from_original_predictions(
            shift_amount_list=shift_amount,
            full_shape_list=full_shape,
        )
        if self.category_remapping:
            self._apply_category_remapping()

    @property
    def object_prediction_list(self):
        return self._object_prediction_list_per_image[0]

    @property
    def object_prediction_list_per_image(self):
        return self._object_prediction_list_per_image

    @property
    def original_predictions(self):
        return self._original_predictions


class Yolov5DetectionModel(DetectionModel):
    def load_model(self):
        """
        Detection model is initialized and set to self.model.
        """
        try:
            # device = select_device('')
            model = DetectMultiBackend(self.model_path, device=self.device)
            model = AutoShape(model)  # for file/URI/PIL/cv2/np inputs and NMS
            self.set_model(model)
        except Exception as e:
            raise TypeError("model_path is not a valid yolov5 model path: ", e)

    def set_model(self, model: Any):
        """
        Sets the underlying YOLOv5 model.
        Args:
            model: Any
                A YOLOv5 model
        """

        if model.__class__.__module__ not in ["yolov5.models.common", "models.common"]:
            raise Exception(f"Not a yolov5 model: {type(model)}")

        model.conf = self.confidence_threshold
        self.model = model

        # set category_mapping
        if not self.category_mapping:
            category_mapping = {str(ind): category_name for ind, category_name in enumerate(self.category_names)}
            self.category_mapping = category_mapping

    def perform_inference(self, image: np.ndarray):
        """
        Prediction is performed using self.model and the prediction result is set to self._original_predictions.
        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted. 3 channel image should be in RGB order.
        """

        # Confirm model is loaded
        if self.model is None:
            raise ValueError("Model is not loaded, load it by calling .load_model()")
        if self.image_size is not None:
            prediction_result = self.model(image, size=self.image_size)
        else:
            prediction_result = self.model(image)

        self._original_predictions = prediction_result

    @property
    def num_categories(self):
        """
        Returns number of categories
        """
        return len(self.model.names)

    @property
    def has_mask(self):
        """
        Returns if model output contains segmentation mask
        """
        has_mask = self.model.with_mask
        return has_mask

    @property
    def category_names(self):
        return self.model.names

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
        full_shape_list: Optional[List[List[int]]] = None,
    ):
        """
        self._original_predictions is converted to a list of prediction.ObjectPrediction and set to
        self._object_prediction_list_per_image.
        Args:
            shift_amount_list: list of list
                To shift the box and mask predictions from sliced image to full sized image, should
                be in the form of List[[shift_x, shift_y],[shift_x, shift_y],...]
            full_shape_list: list of list
                Size of the full image after shifting, should be in the form of
                List[[height, width],[height, width],...]
        """
        original_predictions = self._original_predictions

        # compatilibty for sahi v0.8.15
        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)

        # handle all predictions
        object_prediction_list_per_image = []
        for image_ind, image_predictions_in_xyxy_format in enumerate(original_predictions.xyxy):
            shift_amount = shift_amount_list[image_ind]
            full_shape = None if full_shape_list is None else full_shape_list[image_ind]
            object_prediction_list = []

            # process predictions
            for prediction in image_predictions_in_xyxy_format.cpu().detach().numpy():
                x1 = int(prediction[0])
                y1 = int(prediction[1])
                x2 = int(prediction[2])
                y2 = int(prediction[3])
                bbox = [x1, y1, x2, y2]
                score = prediction[4]
                category_id = int(prediction[5])
                category_name = self.category_mapping[str(category_id)]

                # fix negative box coords
                bbox[0] = max(0, bbox[0])
                bbox[1] = max(0, bbox[1])
                bbox[2] = max(0, bbox[2])
                bbox[3] = max(0, bbox[3])

                # fix out of image box coords
                if full_shape is not None:
                    bbox[0] = min(full_shape[1], bbox[0])
                    bbox[1] = min(full_shape[0], bbox[1])
                    bbox[2] = min(full_shape[1], bbox[2])
                    bbox[3] = min(full_shape[0], bbox[3])

                # ignore invalid predictions
                if not (bbox[0] < bbox[2]) or not (bbox[1] < bbox[3]):
                    logger.warning(f"ignoring invalid prediction with bbox: {bbox}")
                    continue

                object_prediction = ObjectPrediction(
                    bbox=bbox,
                    category_id=category_id,
                    score=score,
                    bool_mask=None,
                    category_name=category_name,
                    shift_amount=shift_amount,
                    full_shape=full_shape,
                )
                object_prediction_list.append(object_prediction)
            object_prediction_list_per_image.append(object_prediction_list)

        self._object_prediction_list_per_image = object_prediction_list_per_image


from sahi_utils import Yolov5DetectionModel
from sahi.predict import get_sliced_prediction

yolov5_model_path = 'weights/crop_best.pt'

detection_model = Yolov5DetectionModel(
    model_path=yolov5_model_path,
    confidence_threshold=0.3,
    device="cpu", # or 'cpu'
)


result = get_sliced_prediction(
    "data/images/cam1_1567911451.793079_3.jpg",
    detection_model,
    slice_height = 256,
    slice_width = 256,
    overlap_height_ratio = 0.2,
    overlap_width_ratio = 0.2
)


result.export_visuals(export_dir="result/")
