import torch
from math import sqrt
from collections import OrderedDict


def create_anchors():
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
    center_anchors = []

    for k, fmap in enumerate(fmaps):
        for i in range(fmap_dims[fmap]):
            for j in range(fmap_dims[fmap]):
                cx = (j + 0.5) / fmap_dims[fmap]
                cy = (i + 0.5) / fmap_dims[fmap]

                for ratio in aspect_ratios[fmap]:
                    center_anchors.append([cx, cy, obj_scales[fmap] * sqrt(ratio), obj_scales[fmap] / sqrt(ratio)])

                    # For an aspect ratio of 1, use an additional prior whose scale is the geometric mean of the
                    # scale of the current feature map and the scale of the next feature map
                    if ratio == 1.:
                        try:
                            additional_scale = sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])
                        # For the last feature map, there is no "next" feature map
                        except IndexError:
                            additional_scale = 1.
                        center_anchors.append([cx, cy, additional_scale, additional_scale])

    center_anchors = torch.FloatTensor(center_anchors)   # (8732, 4)
    center_anchors.clamp_(0, 1)                          # (8732, 4)

    visualization = True
    num_vis_anchors = 1000
    if visualization:
        from utils import cxcy_to_xy, xy_to_cxcy
        from matplotlib.patches import Rectangle
        import matplotlib.pyplot as plt

        # original
        corner_anchors = cxcy_to_xy(center_anchors)

        # center anchor clamp 방식!
        corner_anchors = cxcy_to_xy(center_anchors).clamp(0, 1)
        center_anchors = xy_to_cxcy(corner_anchors)

        size = 300
        img = torch.ones([size, size, 3], dtype=torch.float32)
        axes = plt.axes()
        axes.set_xlim([- 1 / 3 * size, size + 1 / 3 * size])
        axes.set_ylim([- 1 / 3 * size, size + 1 / 3 * size])
        plt.imshow(img)

        for anchor in corner_anchors[:num_vis_anchors]:
            x1 = anchor[0] * size
            y1 = anchor[1] * size
            x2 = anchor[2] * size
            y2 = anchor[3] * size

            plt.gca().add_patch(Rectangle(xy=(x1, y1),
                                          width=x2 - x1,
                                          height=y2 - y1,
                                          linewidth=1,
                                          edgecolor=[0, 1, 0],
                                          facecolor='none'
                                          ))
        plt.show()

    return center_anchors


if __name__ == "__main__":
    anchors = create_anchors()  # torch [8732, 4] 0 ~ 1
