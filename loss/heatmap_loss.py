import torchvision
import torch
import segmentation_models_pytorch
from network.models.EfficientnetConv2DT.utils import gather_output_array, transpose_and_gather_output_array
import torch.nn.functional as F

heatmap_loss_fn = segmentation_models_pytorch.losses.FocalLoss(mode="binary")





def calculate_heatmap_loss(predicted_heatmap, groundtruth_heatmap):
    # heatmap_loss = torchvision.ops.sigmoid_focal_loss(inputs=predicted_heatmap, targets=groundtruth_heatmap,
    #                                   reduction="mean")
    # gradient_mask = torch.zeros_like(groundtruth_heatmap)
    # gradient_mask[groundtruth_heatmap != 0] = 1
    predicted_heatmap = predicted_heatmap.unsqueeze(dim=1)
    groundtruth_heatmap = groundtruth_heatmap.unsqueeze(dim=1)

    heatmap_loss = heatmap_loss_fn(predicted_heatmap.float(), groundtruth_heatmap.float())
    # predicted_heatmap = predicted_heatmap * gradient_mask
    # groundtruth_heatmap = groundtruth_heatmap * gradient_mask
    # heatmap_loss = torch.nn.functional.mse_loss(input=predicted_heatmap.float(), target=groundtruth_heatmap.float(),
    #                                            reduction='mean')
    return heatmap_loss

def calculate_heatmap_loss(predicted_heatmap, groundtruth_heatmap):

    heatmap_loss = torch.nn.functional.mse_loss(input=predicted_heatmap.float(), target=groundtruth_heatmap.float(),
                                                reduction='mean')
    return heatmap_loss

def calculate_heatmap_scatter_loss(predicted_heatmap, groundtruth_heatmap, flattened_index, num_objects, device):
    object_boolean_mask = torch.zeros((flattened_index.shape), device="cuda")
    for i in range(object_boolean_mask.shape[0]):
        object_boolean_mask[i, 0:int(num_objects[i])] = 1
    object_boolean_mask = object_boolean_mask.unsqueeze(2).expand_as(predicted_heatmap).float()
    predicted_heatmap, groundtruth_heatmap = predicted_heatmap * object_boolean_mask, groundtruth_heatmap * object_boolean_mask
    bbox_loss = F.smooth_l1_loss(predicted_heatmap, groundtruth_heatmap, reduction="sum")
    bbox_loss = bbox_loss / object_boolean_mask.sum()
    return bbox_loss