import cv2
import math
import torch
import numpy as np
import torch.nn as nn
from models.vgg import VGG
from torchvision.ops import nms
from models.anchor import create_anchors
from utils.util import cxcy_to_xy, decode


class SSD(nn.Module):
    def __init__(self, in_chs, num_classes, loss_type='multi', pretrained=True):
        super().__init__()
        print("SSD Detector's the number of classes is {}".format(num_classes))

        self.anchors = create_anchors()

        self.in_chs = in_chs
        self.backbone = nn.ModuleList(list(VGG(in_chs, pretrained).features.children()))  # nn.ModuleList
        self.vgg = self.backbone
        self.loss_type = loss_type
        self.num_classes = num_classes
        self.extras = nn.ModuleList([nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=2),
                                     nn.ReLU(inplace=True),
                                     # conv8

                                     nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=2),
                                     nn.ReLU(inplace=True),
                                     # conv9

                                     nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
                                     nn.ReLU(inplace=True),
                                     # conv10

                                     nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
                                     nn.ReLU(inplace=True)]  # conv11
                                    )

        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))  # there are 512 channels in conv4_3_feats
        # nn.init.xavier_normal_(self.rescale_factors)
        nn.init.constant_(self.rescale_factors, 20)

        self.loc = nn.ModuleList([nn.Conv2d(in_channels=512, out_channels=4 * 4, kernel_size=3, padding=1),
                                  nn.Conv2d(in_channels=1024, out_channels=6 * 4, kernel_size=3, padding=1),
                                  nn.Conv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1),
                                  nn.Conv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
                                  nn.Conv2d(in_channels=256, out_channels=4 * 4, kernel_size=3, padding=1),
                                  nn.Conv2d(in_channels=256, out_channels=4 * 4, kernel_size=3, padding=1)])

        self.cls = nn.ModuleList([nn.Conv2d(in_channels=512, out_channels=4 * self.num_classes, kernel_size=3, padding=1),
                                  nn.Conv2d(in_channels=1024, out_channels=6 * self.num_classes, kernel_size=3, padding=1),
                                  nn.Conv2d(in_channels=512, out_channels=6 * self.num_classes, kernel_size=3, padding=1),
                                  nn.Conv2d(in_channels=256, out_channels=6 * self.num_classes, kernel_size=3, padding=1),
                                  nn.Conv2d(in_channels=256, out_channels=4 * self.num_classes, kernel_size=3, padding=1),
                                  nn.Conv2d(in_channels=256, out_channels=4 * self.num_classes, kernel_size=3, padding=1)])

        self.init_conv2d()
        print("num_params : ", self.count_parameters())

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def init_conv2d(self):
        for c in self.cls.children():
            if isinstance(c, nn.Conv2d):

                if self.loss_type == 'multi':
                    # for origin ssd
                    nn.init.xavier_uniform_(c.weight)
                    nn.init.uniform_(c.bias, 0.)

                elif self.loss_type == 'focal':

                    # for focal loss
                    pi = 0.01
                    b = - math.log((1 - pi) / pi)
                    nn.init.constant_(c.bias, b)
                    nn.init.normal_(c.weight, std=0.01)

        for c in self.extras.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.uniform_(c.bias, 0.)

        for c in self.loc.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.uniform_(c.bias, 0.)

    def forward(self, x):

        assert self.in_chs == x.size(1), 'must same to the channel of x and in_chs'

        # result 를 담는 list
        cls = []
        loc = []

        # 6개의 multi scale feature 를 담는 list
        scale_features = []
        for i in range(23):
            x = self.vgg[i](x)  # conv 4_3 relu

        # l2 norm technique
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()     # (N, 1, 38, 38)
        conv4_3 = x / norm                                  # (N, 512, 38, 38)
        conv4_3 = conv4_3 * self.rescale_factors            # (N, 512, 38, 38)

        scale_features.append(conv4_3)

        for i in range(23, len(self.vgg)):
            x = self.vgg[i](x)   # conv 7 relu
        conv7 = x
        scale_features.append(conv7)

        for i in range(len(self.extras)):
            x = self.extras[i](x)
            if i % 4 == 3:
                scale_features.append(x)

        batch_size = scale_features[0].size(0)
        for (x, c, l) in zip(scale_features, self.cls, self.loc):
            cls.append(c(x).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.num_classes))  # permute  --> view
            loc.append(l(x).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4))   # permute  --> view

        # 이전과 비교
        cls = torch.cat([c for c in cls], dim=1)
        loc = torch.cat([l for l in loc], dim=1)
        return cls, loc

    def predict(self, pred, center_anchor, opts):
        pred_cls = pred[0]
        pred_reg = pred[1]

        pred_cls = torch.softmax(pred_cls, dim=-1).squeeze()
        pred_bbox = cxcy_to_xy(decode(pred_reg.squeeze(), center_anchor))
        pred_bbox = pred_bbox.clamp(min=0, max=1)

        bbox, label, score = self._suppress(pred_bbox, pred_cls, opts)
        return bbox, label, score

    def _suppress(self, raw_cls_bbox, raw_prob, opts):
        bbox = list()
        label = list()
        score = list()

        # skip cls_id = 0 because it is the background class
        for l in range(1, opts.num_classes):
            prob_l = raw_prob[:, l]                  # [8732, 21]
            mask = prob_l > opts.thres               # [8732] - torch.bool
            cls_bbox_l = raw_cls_bbox[mask]
            prob_l = prob_l[mask]
            keep = nms(cls_bbox_l, prob_l, iou_threshold=0.45)
            bbox.append(cls_bbox_l[keep].cpu().numpy())
            label.append((l - 1) * np.ones((len(keep),)))
            score.append(prob_l[keep].cpu().numpy())
            '''
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.num_classes, 4))[:, l, :]
            prob_l = raw_prob[:, l]
            mask = prob_l > opts.thres
            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]
            keep = nms(cls_bbox_l, prob_l, iou_threshold=0.3)
            bbox.append(cls_bbox_l[keep].cpu().numpy())
            label.append((l - 1) * np.ones((len(keep),)))
            score.append(prob_l[keep].cpu().numpy())
            '''

        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)

        # get top k
        n_objects = score.shape[0]
        top_k = opts.top_k
        if n_objects > opts.top_k:
            sort_ind = score.argsort(axis=0)[::-1]    # [::-1] means descending
            score = score[sort_ind][:top_k]           # (top_k)
            bbox = bbox[sort_ind][:top_k]             # (top_k, 4)
            label = label[sort_ind][:top_k]           # (top_k)
        return bbox, label, score


if __name__ == '__main__':
    img = torch.randn([2, 3, 300, 300]).cuda()
    model = SSD(in_chs=3, num_classes=21, pretrained=True).cuda()
    output = model(img)
    print(output[0].size())
    print(output[1].size())