from losses.loss import MultiBoxLoss


def build_loss(opts):
    criterion = MultiBoxLoss()
    return criterion

