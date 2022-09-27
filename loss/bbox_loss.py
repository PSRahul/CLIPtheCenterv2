import torch
import torch.nn.functional as F
import segmentation_models_pytorch
from network.models.SMP_DeepLab.utils import transpose_and_gather_output_array
import numpy as np

def calculate_bbox_loss_with_heatmap(predicted_bbox, groundtruth_bbox, flattened_index, num_objects, device):
    ################# DEBUG

    predicted_width = predicted_bbox[:, 0, :, :].flatten(start_dim=1, end_dim=- 1)
    predicted_height = predicted_bbox[:, 1, :, :].flatten(start_dim=1, end_dim=- 1)

    groundtruth_width = groundtruth_bbox[:, 0, :, :].flatten(start_dim=1, end_dim=- 1)
    groundtruth_height = groundtruth_bbox[:, 1, :, :].flatten(start_dim=1, end_dim=- 1)

    bbox_loss_width = torch.nn.functional.mse_loss(input=predicted_width.float(),
                                                   target=groundtruth_width.float(),
                                                   reduction='mean')
    bbox_loss_height = torch.nn.functional.mse_loss(input=predicted_height.float(),
                                                    target=groundtruth_height.float(),
                                                    reduction='mean')
    bbox_loss = bbox_loss_height + bbox_loss_width
    return bbox_loss



def calculate_bbox_loss_without_heatmap(predicted_bbox, groundtruth_bbox, flattened_index, num_objects, device):
    predicted_bbox = transpose_and_gather_output_array(predicted_bbox, flattened_index)
    bbox_loss = F.mse_loss(predicted_bbox.float(), groundtruth_bbox.float(), reduction="mean")
    return bbox_loss


