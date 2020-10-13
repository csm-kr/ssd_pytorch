import torch
from math import sqrt
from collections import OrderedDict
from config import device


def create_anchor_boxes():
    """
    Create the 8732 prior (default) boxes for the SSD300, as defined in the paper.
    :return: prior boxes in center-size coordinates, a tensor of dimensions (8732, 4)
    """
    fmap_dims = {'conv4_3': 38,
                 'conv7': 19,
                 'conv8_2': 10,
                 'conv9_2': 5,
                 'conv10_2': 3,
                 'conv11_2': 1}
    fmap_dims = OrderedDict(sorted(fmap_dims.items(), key=lambda t: t[1], reverse=True))  # 내림차순
    # value 를 기준으로 sorted 함!

    obj_scales = {'conv4_3': 0.1,
                  'conv7': 0.2,
                  'conv8_2': 0.375,
                  'conv9_2': 0.55,
                  'conv10_2': 0.725,
                  'conv11_2': 0.9}

    aspect_ratios = {'conv4_3': [1., 2., 0.5],
                     'conv7': [1., 2., 3., 0.5, .333],
                     'conv8_2': [1., 2., 3., 0.5, .333],
                     'conv9_2': [1., 2., 3., 0.5, .333],
                     'conv10_2': [1., 2., 0.5],
                     'conv11_2': [1., 2., 0.5]}

    fmaps = list(fmap_dims.keys())
    prior_boxes = []

    for k, fmap in enumerate(fmaps):
        for i in range(fmap_dims[fmap]):
            for j in range(fmap_dims[fmap]):
                cx = (j + 0.5) / fmap_dims[fmap]
                cy = (i + 0.5) / fmap_dims[fmap]

                for ratio in aspect_ratios[fmap]:
                    prior_boxes.append([cx, cy, obj_scales[fmap] * sqrt(ratio), obj_scales[fmap] / sqrt(ratio)])

                    # For an aspect ratio of 1, use an additional prior whose scale is the geometric mean of the
                    # scale of the current feature map and the scale of the next feature map
                    if ratio == 1.:
                        try:
                            additional_scale = sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])
                        # For the last feature map, there is no "next" feature map
                        except IndexError:
                            additional_scale = 1.
                        prior_boxes.append([cx, cy, additional_scale, additional_scale])

    prior_boxes = torch.FloatTensor(prior_boxes).to(device)  # (8732, 4)
    prior_boxes.clamp_(0, 1)  # (8732, 4)
    return prior_boxes


if __name__ == "__main__":
    de_box = create_anchor_boxes()
    de_b = create_anchor_boxes()
    print(de_box.size())
    for db in de_box:
        print(db)

    print(torch.equal(de_box, de_b))