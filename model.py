import torch.nn as nn
import torch
import math
import torchvision
from config import device


class VGG(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(64, 64, 3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(2, 2),
                                      # conv1

                                      nn.Conv2d(64, 128, 3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 128, 3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(2, 2),
                                      # conv2

                                      nn.Conv2d(128, 256, 3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(256, 256, 3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(256, 256, 3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                                      # conv3

                                      nn.Conv2d(256, 512, 3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(512, 512, 3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(512, 512, 3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      # conv4

                                      nn.Conv2d(512, 512, 3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(512, 512, 3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(512, 512, 3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                                      # conv5

                                      nn.Conv2d(512, 1024, 3, padding=6, dilation=6),
                                      nn.ReLU(inplace=True),
                                      # conv6

                                      nn.Conv2d(1024, 1024, 1, padding=0),
                                      nn.ReLU(inplace=True),
                                      # conv7
                                      )
        self.init_conv2d()

        if pretrained:

            std = torchvision.models.vgg16(pretrained=True).features.state_dict()
            model_dict = self.features.state_dict()
            pretrained_dict = {k: v for k, v in std.items() if k in model_dict}  # 여기서 orderdict 가 아니기 때문에
            model_dict.update(pretrained_dict)
            self.features.load_state_dict(model_dict)

    def init_conv2d(self):
        for c in self.features.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, x):
        x = self.features(x)
        return x


class SSD(nn.Module):
    def __init__(self, base, n_classes=21, loss_type='multi'):
        super().__init__()
        print("SSD Detector's the number of classes is {}".format(n_classes))
        self.vgg = nn.ModuleList(list(base.features.children()))  # nn.ModuleList
        self.loss_type = loss_type
        self.n_classes = n_classes
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
        nn.init.constant_(self.rescale_factors, int(n_classes - 1))

        self.loc = nn.ModuleList([nn.Conv2d(in_channels=512, out_channels=4 * 4, kernel_size=3, padding=1),
                                  nn.Conv2d(in_channels=1024, out_channels=6 * 4, kernel_size=3, padding=1),
                                  nn.Conv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1),
                                  nn.Conv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
                                  nn.Conv2d(in_channels=256, out_channels=4 * 4, kernel_size=3, padding=1),
                                  nn.Conv2d(in_channels=256, out_channels=4 * 4, kernel_size=3, padding=1)])

        self.cls = nn.ModuleList([nn.Conv2d(in_channels=512, out_channels=4 * self.n_classes, kernel_size=3, padding=1),
                                  nn.Conv2d(in_channels=1024, out_channels=6 * self.n_classes, kernel_size=3, padding=1),
                                  nn.Conv2d(in_channels=512, out_channels=6 * self.n_classes, kernel_size=3, padding=1),
                                  nn.Conv2d(in_channels=256, out_channels=6 * self.n_classes, kernel_size=3, padding=1),
                                  nn.Conv2d(in_channels=256, out_channels=4 * self.n_classes, kernel_size=3, padding=1),
                                  nn.Conv2d(in_channels=256, out_channels=4 * self.n_classes, kernel_size=3, padding=1)])

        self.init_conv2d()
        print("num_params : ", self.count_parameters())

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
            cls.append(c(x).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.n_classes))  # permute  --> view
            loc.append(l(x).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4))   # permute  --> view

        # 이전과 비교
        cls = torch.cat([c for c in cls], dim=1)
        loc = torch.cat([l for l in loc], dim=1)

        return loc, cls

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":

    vgg = VGG(pretrained=True).to(device)
    img = torch.FloatTensor(2, 3, 300, 300).to(device)
    ssd = SSD(vgg).to(device)
    cls, loc = ssd(img)
    print(cls.size())
    print(loc.size())

