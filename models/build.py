import torch
from models.model import SSD
from torch.nn.parallel import DistributedDataParallel as DDP


def build_model(opts):
    if opts.distributed:
        model = SSD(in_chs=opts.in_chs, num_classes=opts.num_classes, pretrained=opts.pretrained)
        model = model.cuda(int(opts.gpu_ids[opts.rank]))
        model = DDP(module=model,
                    device_ids=[int(opts.gpu_ids[opts.rank])],
                    find_unused_parameters=False)
    else:
        # IF DP
        model = SSD(in_chs=opts.in_chs, num_classes=opts.num_classes, pretrained=opts.pretrained)
        model = torch.nn.DataParallel(module=model, device_ids=[int(id) for id in opts.gpu_ids])
    return model




