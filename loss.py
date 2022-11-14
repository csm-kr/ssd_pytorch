import torch
import torch.nn as nn
from utils import cxcy_to_xy, encode, xy_to_cxcy, find_jaccard_overlap


class TargetMaker(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, gt_boxes, gt_labels, center_anchor, positive_threshold=0.5):

        batch_size = len(gt_labels)
        num_anchors = center_anchor.size(0)
        device_ = gt_labels[0].get_device()

        # 1. make container
        gt_locations = torch.zeros((batch_size, num_anchors, 4), dtype=torch.float).to(device_)  # (N, 8732, 4)
        gt_classes = torch.zeros((batch_size, num_anchors), dtype=torch.long).to(device_)   # (N, 8732)
        gt_masks = torch.zeros((batch_size, num_anchors), dtype=torch.bool).to(device_)     # (N, 8732)

        # 2. make corner anchors
        center_anchor = center_anchor.to(device_)
        corner_anchor = cxcy_to_xy(center_anchor)

        # 3. 각 이미지에 대한
        for i in range(batch_size):
            boxes = gt_boxes[i]
            labels = gt_labels[i]
            num_objects = boxes.size()[0]

            # 3-1. iou구한다.
            iou = find_jaccard_overlap(corner_anchor, boxes)   # [num_anchors, num_objects] - [8732, 20]

            # condition 1 - maximum iou
            _, obj_idx = iou.max(dim=0)  # [num_obj]

            # condition 2 - iou that higher than 0.5
            # iou of maximum obj set 1.0 for satisfying both condition 1 and condition 2.
            for obj in range(len(obj_idx)):
                iou[obj_idx[obj]][obj] = 1.

            positive_anchors = iou >= positive_threshold         # [num_anchors, num_objects] \in [0, 1]
            positive_anchors, _ = positive_anchors.max(dim=1)
            gt_masks[i] = positive_anchors                       # [num_anchors] - [8732]

            _, max_anchors_idx = iou.max(dim=1)                  # [num_anchors]

            # label
            gt_classes_ = labels[max_anchors_idx] + 1            # ** for background, each labels plus 1 **
            gt_classes[i] = gt_classes_ * positive_anchors       # ** if not positive anchors, labels make zero **

            # box
            gt_center_locations = xy_to_cxcy(boxes[max_anchors_idx])
            gt_locations_ = encode(gt_center_locations, center_anchor)
            gt_locations[i] = gt_locations_

        return gt_classes, gt_locations, gt_masks


class MultiBoxLoss(nn.Module):
    def __init__(self, neg_pos_ratio=3, alpha=1.):
        super(MultiBoxLoss, self).__init__()

        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)
        self.smooth_l1 = nn.SmoothL1Loss()
        self.target_maker = TargetMaker()

    def forward(self, pred, gt_boxes, gt_labels, center_anchor):

        pred_cls = pred[0]
        pred_loc = pred[1]

        device_ = pred_cls.get_device()
        batch_size = pred_loc.size(0)
        num_classes = pred_cls.size(2)

        num_anchors = int(center_anchor.size(0))
        assert num_anchors == pred_loc.size(1) == pred_cls.size(1)  # 67995 --> 120087

        center_anchor = center_anchor.to(device_)

        # build targets
        gt_cls, gt_loc, gt_masks = self.target_maker(gt_boxes, gt_labels, center_anchor)

        # localization loss - smooth l1 loss
        num_positives = gt_masks.sum(dim=1)                              # each batches, num of pos anchors (e.g) [2, 8]
        loc_loss = self.smooth_l1(pred_loc[gt_masks], gt_loc[gt_masks])  # (), scalar

        # classification loss - cross entropy
        cls_loss_all = self.cross_entropy(pred_cls.view(-1, num_classes), gt_cls.view(-1))  # (N * 8732)
        cls_loss_all = cls_loss_all.view(batch_size, num_anchors)  # (N, 8732)  # positive bbox 를 위한 resize

        # ** hard negative mining **
        # about pos anchors  (eg. 10 number only)
        cls_loss_pos = cls_loss_all[gt_masks]  # (sum(n_positives))  # positive masking
        # about neg anchors
        cls_loss_neg = cls_loss_all.clone()  # (N, 8732)  #  new allocated cls loss
        # hard negative mining
        cls_loss_neg[gt_masks] = 0.                                   # (N, 8732), positive priors are ignored (never in top n_hard_negatives)
        cls_loss_neg, _ = cls_loss_neg.sort(dim=1, descending=True)   # (N, 8732), sorted by decreasing hardness
        nim_hard_negatives = self.neg_pos_ratio * num_positives       # make number of each batch of hard samples using ratio
        hardness_ranks = torch.LongTensor(range(num_anchors)).unsqueeze(0).expand_as(cls_loss_neg).to(device_)
        # make a row 0 to 8732 (N, 8732) shape's tensor to index hard negative samples

        hard_negative_mask = hardness_ranks < nim_hard_negatives.unsqueeze(1)  # (N, 8732)
        # remains only top-k hard negative samples indices.

        cls_loss_hard_neg = cls_loss_neg[hard_negative_mask]  # it means a network knows zero is background
        cls_loss = (cls_loss_hard_neg.sum() + cls_loss_pos.sum()) / num_positives.sum().float()  # pos + neg loss

        # TOTAL LOSS
        loc_loss = self.alpha * loc_loss
        total_loss = (cls_loss + loc_loss)
        return total_loss, (cls_loss, loc_loss)


if __name__ == '__main__':
    device = torch.device('cuda')
    from models.model import SSD
    img = torch.FloatTensor(2, 3, 300, 300).to(device)
    ssd = SSD(in_chs=3, num_classes=21).to(device)
    pred = ssd(img)
    pred_cls = pred[0]
    pred_loc = pred[1]
    print(pred_cls.size())
    print(pred_loc.size())
    gt = [torch.Tensor([[0.426, 0.158, 0.788, 0.997]]).to(device),
          torch.Tensor([[0.002, 0.090, 0.998, 0.867]]).to(device)]
    label = [torch.Tensor([15.]).to(device),
             torch.Tensor([1.]).to(device)]
    print(pred)

    loss = MultiBoxLoss()
    print(loss(pred, gt, label, ssd.anchors))

