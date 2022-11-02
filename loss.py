import torch
import torch.nn as nn
from utils import cxcy_to_xy, encode, xy_to_cxcy, find_jaccard_overlap


class MultiBoxLoss(nn.Module):
    def __init__(self, threshold=0.5, neg_pos_ratio=3, alpha=10., ):
        super(MultiBoxLoss, self).__init__()

        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha
        self.focal_loss = False

        self.smooth_l1 = nn.L1Loss()
        # self.smooth_l1 = nn.SmoothL1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)

    def forward(self, pred, gt_boxes, gt_labels, center_anchor):

        pred_cls = pred[0]
        pred_loc = pred[1]
        device_ = pred_cls.get_device()
        batch_size = pred_loc.size(0)
        n_classes = pred_cls.size(2)

        n_priors = int(center_anchor.size(0))
        assert n_priors == pred_loc.size(1) == pred_cls.size(1)  # 67995 --> 120087

        center_anchor = center_anchor.to(device_)
        corner_anchor = cxcy_to_xy(center_anchor)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device_)  # (N, 8732, 4)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(device_)   # (N, 8732)
        batch_postivie_default_box = torch.zeros((batch_size, n_priors), dtype=torch.bool).to(device_)  # (N, 8732)

        # batch 에 따라 cls, loc 가 모두 다르니 batch 로 나누어 준다.
        for i in range(batch_size):
            boxes = gt_boxes[i]
            labels = gt_labels[i]
            n_objects = boxes.size()[0]
            # ------------------------------------- my code -------------------------------------
            # step1 ) positive default box
            iou = find_jaccard_overlap(boxes, corner_anchor)   # [ num_obj, num_default_box ]

            # condition 1 - maximum iou
            _, obj_idx = iou.max(dim=1)  # [num_obj]

            # condition 2 - iou that higher than 0.5
            # iou of maximum obj set 1.0 for satisfying both condition 1 and condition 2.
            for obj in range(len(obj_idx)):
                iou[obj][obj_idx[obj]] = 1.

            positive_prior = iou >= self.threshold  # > 0.5

            positive_prior, _ = positive_prior.max(dim=0)
            batch_postivie_default_box[i] = positive_prior

            # step 2 ) cls label
            _, max_prior_idx = iou.max(dim=0)  # iou 에서 max 인 cls 를 찾으려는 부분 idx

            # 모든 default box 의 label 을 구하고 positive prior box 만 연산
            # FIXME positive prior 이 float32
            true_classes_ = (labels[max_prior_idx] + 1) * positive_prior.type(torch.long)
            # --> long 으로
            true_classes[i] = true_classes_

            # step 3 ) loc label
            # 모든 default box 의 g^^ 을 구하고 loss 다룰때만 연산 --> cls 는 back ground 가 0 이었기 때문에 곱!
            # b = xy_to_cxcy(boxes[max_prior_idx])
            true_locs_ = xy_to_cxcy(boxes[max_prior_idx])
            # bbox
            true_locs_ = encode(true_locs_, center_anchor)
            true_locs[i] = true_locs_

        ###################################################
        # location loss
        ###################################################
        positive_priors = batch_postivie_default_box  # B, 8732 : for positive anchors, smooth l1
        n_positives = positive_priors.sum(dim=1)      # B       : each batches, num of positive sample (e.g) [2, 8]
        loc_loss = self.smooth_l1(pred_loc[positive_priors], true_locs[positive_priors])  # (), scalar

        ###################################################
        # classification loss
        ###################################################
        # ---------------- original ssd loss ----------------
        if not self.focal_loss:
            # about whole anchors 90 - 112
            conf_loss_all = self.cross_entropy(pred_cls.view(-1, n_classes), true_classes.view(-1))  # (N * 8732)
            conf_loss_all = conf_loss_all.view(batch_size, n_priors)  # (N, 8732)  # positive bbox 를 위한 resize

            # about pos anchors  (eg. 10 number only)
            conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))  # positive masking

            # about neg anchors
            conf_loss_neg = conf_loss_all.clone()  # (N, 8732)  #  new allocated conf loss

            # hard negative mining
            conf_loss_neg[
                positive_priors] = 0.  # (N, 8732), positive priors are ignored (never in top n_hard_negatives)
            conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  # (N, 8732), sorted by decreasing hardness

            n_hard_negatives = self.neg_pos_ratio * n_positives  # make number of each batch of hard samples using ratio
            hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device_)
            # make a row 0 to 8732 (N, 8732) shape's tensor to index hard negative samples

            hard_negative_mask = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 8732)
            # remains only top-k hard negative samples indices.

            conf_loss_hard_neg = conf_loss_neg[hard_negative_mask]  # it means a network knows zero is background
            conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()  # pos + neg loss

        # ---------------- focal classification ssd loss ----------------
        elif self.focal_loss:
            # focal loss : 117 - 135
            # focal loss 는 positive negative 상관하지 않는다. --> 마지막에 나눠줄때만 ㅎㅎ
            # 1) ture_classes 를 one_hot 으로 바꾼다. (B, 8732, 21)
            # 2) softmax 를 계산한다.                 (B, 8732, 21)
            # 3) a_t 는

            alpha = 0.25
            gamma = 2

            # cvt to one hot encoding
            y = torch.eye(21).to(device_)  # [D,D]
            targets = y[true_classes]

            targets = targets[..., 1:]    # remove background labels
            pred_cls = pred_cls[..., 1:]  # remove background prediction

            alpha_factor = torch.ones(targets.shape).to(device_) * alpha
            a_t = torch.where((targets == 1), alpha_factor, 1. - alpha_factor)
            pred_cls = torch.sigmoid(pred_cls).clamp(1e-4, 1.0 - 1e-4)
            p_t = torch.where(targets == 1, pred_cls, 1 - pred_cls)  # p_t
            from torch.nn import functional as F
            # ce = F.binary_cross_entropy(pred_cls, targets)
            ce = -torch.log(p_t)
            conf_loss = (a_t * torch.pow(1 - p_t, gamma) * ce).sum() / n_positives.sum()  # focal loss

        # TOTAL LOSS
        loc_loss = self.alpha * loc_loss
        total_loss = (conf_loss + loc_loss)
        return total_loss, (conf_loss, loc_loss)


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
    loss = MultiBoxLoss()
    print(loss(pred, gt, label, ssd.anchors))