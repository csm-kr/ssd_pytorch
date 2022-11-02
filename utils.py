import os
import math
import torch
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
from torchvision.ops.boxes import nms as torchvision_nms


# for voc label
voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
voc_label_map = {k: v for v, k in enumerate(voc_labels)}
voc_label_map['background'] = 20
voc_rev_label_map = {v: k for k, v in voc_label_map.items()}  # Inverse mapping
np.random.seed(0)
voc_color_array = np.random.randint(256, size=(21, 3)) / 255  # In plt, rgb color space's range from 0 to 1

# for coco label
coco_labels = ('person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
               'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
               'wine glass', 'cup', 'fork', 'knife', 'spoon',
               'bowl', 'banana', 'apple', 'sandwich', 'orange',
               'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
               'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse',
               'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
               'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

coco_label_map = {k: v for v, k in enumerate(coco_labels)}  # {0 ~ 79 : 'person' ~ 'toothbrush'}
coco_label_map['background'] = 80                                # {80 : 'background'}
coco_rev_label_map = {v: k for k, v in coco_label_map.items()}  # Inverse mapping
np.random.seed(1)
coco_color_array = np.random.randint(256, size=(81, 3)) / 255  # In plt, rgb color space's range from 0 to 1


def bar_custom(current, total, width=30):
    avail_dots = width-2
    shaded_dots = int(math.floor(float(current) / total * avail_dots))
    percent_bar = '[' + '■'*shaded_dots + ' '*(avail_dots-shaded_dots) + ']'
    progress = "%d%% %s [%d / %d byte]" % (current / total * 100, percent_bar, current, total)
    return progress


def cxcy_to_xy(cxcy):

    x1y1 = cxcy[..., :2] - cxcy[..., 2:] / 2
    x2y2 = cxcy[..., :2] + cxcy[..., 2:] / 2
    return torch.cat([x1y1, x2y2], dim=1)


def xy_to_cxcy(xy):

    cxcy = (xy[..., 2:] + xy[..., :2]) / 2
    wh = xy[..., 2:] - xy[..., :2]
    return torch.cat([cxcy, wh], dim=1)


def xy_to_cxcy2(xy):
    wh = xy[..., 2:] - xy[..., :2]
    cxcy = xy[..., :2] + 0.5 * wh
    return torch.cat([cxcy, wh], dim=1)


def find_jaccard_overlap(set_1, set_2, eps=1e-5):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection + eps  # (n1, n2)

    return intersection / union  # (n1, n2)


def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)  # 0 혹은 양수로 만드는 부분
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)  # 둘다 양수인 부분만 존재하게됨!


def encode(gt_cxywh, anc_cxywh):
    tg_cxy = (gt_cxywh[:, :2] - anc_cxywh[:, :2]) / anc_cxywh[:, 2:]
    tg_wh = torch.log(gt_cxywh[:, 2:] / anc_cxywh[:, 2:])
    tg_cxywh = torch.cat([tg_cxy, tg_wh], dim=1)
    return tg_cxywh


def decode(tcxcy, center_anchor):
    cxcy = tcxcy[:, :2] * center_anchor[:, 2:] + center_anchor[:, :2]
    wh = torch.exp(tcxcy[:, 2:]) * center_anchor[:, 2:]
    cxywh = torch.cat([cxcy, wh], dim=1)
    return cxywh


def resume(opts, model, optimizer, scheduler):
    if opts.start_epoch != 0:
        # take pth at epoch - 1

        f = os.path.join(opts.log_dir, opts.name, 'saves', opts.name + '.{}.pth.tar'.format(opts.start_epoch - 1))
        device = torch.device('cuda:{}'.format(opts.gpu_ids[opts.rank]))
        checkpoint = torch.load(f=f,
                                map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])                              # load model state dict
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])                      # load optim state dict
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])                      # load sched state dict
        if opts.rank == 0:
            print('\nLoaded checkpoint from epoch %d.\n' % (int(opts.start_epoch) - 1))
    else:
        if opts.rank == 0:
            print('\nNo check point to resume.. train from scratch.\n')
    return model, optimizer, scheduler


def nms(boxes, scores, iou_threshold=0.5, top_k=200):

    # 1. num obj
    num_boxes = len(boxes)

    # 2. get sorted scores, boxes
    sorted_scores, idx_scores = scores.sort(descending=True)
    sorted_boxes = boxes[idx_scores]

    # 3. iou
    iou = find_jaccard_overlap(sorted_boxes, sorted_boxes)
    keep = torch.ones(num_boxes, dtype=torch.bool)

    # 4. suppress boxes except max boxes
    for each_box_idx, iou_for_each_box in enumerate(iou):
        if keep[each_box_idx] == 0:  # 이미 없는것
            continue

        # 압축조건
        suppress = iou_for_each_box > iou_threshold  # 없앨 아이들
        keep[suppress] = 0
        keep[each_box_idx] = 1  # 자기자신은 살린당.

    return keep, sorted_scores, sorted_boxes


def detect_objects(priors_cxcy, predicted_locs, predicted_scores, min_score, max_overlap, top_k, n_classes=21):
    device = predicted_locs.get_device()
    """
    batch 1 에 대한 boxes 와 labels 와 scores 를 찾는 함수
    :param priors_cxcy: [8732, 4]
    :param predicted_locs: [1, 8732, 4]
    :param predicted_scores: [1, 8732, 21]
    :return:
    after nms, remnant object is num_objects <= 200
    image_boxes: [num_objects, 4]
    image_labels:[num_objects]
    image_scores:[num_objects]
    """

    batch_size = predicted_locs.size(0)
    n_priors = priors_cxcy.size(0)
    predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 8732, n_classes)

    assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

    # Decode object coordinates from the form we regressed predicted boxes to
    decoded_locs = cxcy_to_xy(
        gcxgcy_to_cxcy(predicted_locs[0], priors_cxcy))  # (8732, 4), these are fractional pt. coordinates

    # Lists to store boxes and scores for this image
    image_boxes = list()
    image_labels = list()
    image_scores = list()

    # Check for each class
    for c in range(1, n_classes):
        # Keep only predicted boxes and scores where scores for this class are above the minimum score
        class_scores = predicted_scores[0][:, c]  # (8732)
        score_above_min_score = class_scores > min_score  # torch.uint8 (byte) tensor, for indexing
        n_above_min_score = score_above_min_score.sum().item()
        if n_above_min_score == 0:
            continue
        class_scores = class_scores[score_above_min_score]  # (n_qualified), n_min_score <= 8732
        class_decoded_locs = decoded_locs[score_above_min_score]  # (n_qualified, 4)

        sorted_scores, idx_scores = class_scores.sort(descending=True)
        sorted_boxes = class_decoded_locs[idx_scores]
        sorted_boxes = sorted_boxes.clamp(0, 1)  # 0 ~ 1 로 scaling 해줌 --> 조금 오르려나? 78.30 --> 78.45 로 오름!

        num_boxes = len(sorted_boxes)
        keep_idx = torchvision_nms(boxes=sorted_boxes, scores=sorted_scores, iou_threshold=max_overlap)
        keep_ = torch.zeros(num_boxes, dtype=torch.bool)
        keep_[keep_idx] = 1  # int64 to bool
        keep = keep_

        # Store only unsuppressed boxes for this class
        image_boxes.append(sorted_boxes[keep])
        image_labels.append(torch.LongTensor((keep).sum().item() * [c]).to(device))
        image_scores.append(sorted_scores[keep])

    # If no object in any class is found, store a placeholder for 'background'
    if len(image_boxes) == 0:
        image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
        image_labels.append(torch.LongTensor([0]).to(device))
        image_scores.append(torch.FloatTensor([0.]).to(device))

    # Concatenate into single tensors
    image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
    image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
    image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
    n_objects = image_scores.size(0)

    # Keep only the top k objects --> 다구하고 200 개를 자르는 것은 느리지 않은가?
    if n_objects > top_k:
        image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
        image_scores = image_scores[:top_k]  # (top_k)
        image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
        image_labels = image_labels[sort_ind][:top_k]  # (top_k)

    return image_boxes, image_labels, image_scores  # lists of length batch_size


def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):

    cxcy = gcxgcy[:, :2] * priors_cxcy[:, 2:] + priors_cxcy[:, :2]
    wh = torch.exp(gcxgcy[:, 2:]) * priors_cxcy[:, 2:]
    return torch.cat([cxcy, wh], dim=1)


def cxcy_to_gcxgcy(cxcy, priors_cxcy):

    gcxcy = (cxcy[:, :2] - priors_cxcy[:, :2]) / priors_cxcy[:, 2:]
    gwh = torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:])
    return torch.cat([gcxcy, gwh], dim=1)


def init_for_distributed(rank, opts):

    # 1. setting for distributed training
    opts.rank = rank
    local_gpu_id = int(opts.gpu_ids[opts.rank])
    torch.cuda.set_device(local_gpu_id)
    if opts.rank is not None:
        print("Use GPU: {} for training".format(local_gpu_id))

    # # 2. init_process_group
    dist.init_process_group(backend='nccl',
                            init_method='tcp://127.0.0.1:23456',
                            world_size=opts.world_size,
                            rank=opts.rank)

    # if put this function, the all processes block at all.
    torch.distributed.barrier()
    # convert print fn iif rank is zero
    setup_for_distributed(opts.rank == 0)
    print(opts)
    return


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
