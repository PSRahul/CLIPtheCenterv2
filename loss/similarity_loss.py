import torch.nn.functional as F
import torch
import torchvision


def calculate_embedding_loss(predicted_embedding, groundtruth_embedding,flattened_index, num_objects):
    object_boolean_mask = torch.zeros((flattened_index.shape), device="cuda")
    for i in range(object_boolean_mask.shape[0]):
        object_boolean_mask[i, 0:int(num_objects[i])] = 1
    object_boolean_mask=object_boolean_mask.view((object_boolean_mask.shape[0]*object_boolean_mask.shape[1]))
    #predicted_embedding = predicted_embedding.view((flattened_index.shape[0], flattened_index.shape[1], 512))
    #groundtruth_embedding = groundtruth_embedding.view((flattened_index.shape[0], flattened_index.shape[1], 512))

    object_boolean_mask = object_boolean_mask.unsqueeze(1).expand_as(predicted_embedding.float())
    predicted_embedding, groundtruth_embedding = predicted_embedding * object_boolean_mask, groundtruth_embedding * object_boolean_mask

    target = torch.zeros((flattened_index.shape), device="cuda")
    for i in range(target.shape[0]):
        target[i, 0:int(num_objects[i])] = 1
    target = target.view((target.shape[0] * target.shape[1]))
    embedding_loss = F.cosine_embedding_loss(input1=predicted_embedding, input2=groundtruth_embedding,
                                             target=target,reduce=True)
    embedding_loss = embedding_loss / num_objects.sum()
    return embedding_loss
