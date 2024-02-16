import numpy as np
import torch.nn as nn
from monai.losses import GeneralizedDiceLoss, DiceCELoss, DiceLoss


def get_loss(args):
    if args.loss == "GeneralizedDiceLoss":
        loss = GeneralizedDiceLoss(
            include_background=args.include_background,
            to_onehot_y=True,
            softmax=args.use_softmax,
            smooth_nr=args.smooth_nr,
            smooth_dr=args.smooth_dr
        )
    elif args.loss == "DiceCELoss":
        loss = DiceCELoss(include_background=args.include_background,
                          to_onehot_y=True,
                          softmax=args.use_softmax,
                          sigmoid=args.use_sigmoid,
                          squared_pred=False,
                          lambda_dice=args.dice_prop,
                          lambda_ce=1 - args.dice_prop,
                          smooth_nr=args.smooth_nr,
                          smooth_dr=args.smooth_dr,
                          reduction=args.loss_reduction)

    elif args.loss == "Dice":
        loss = DiceLoss(include_background=args.include_background,
                        to_onehot_y=True,
                        softmax=args.use_softmax,
                        smooth_nr=args.smooth_nr,
                        smooth_dr=args.smooth_dr,
                        reduction=args.loss_reduction)

    if (args.do_ds or args.loss == "DeepSupervision"):
        ################# Here we wrap the loss for deep supervision ############
        # we need to know the number of outputs of the network
        net_numpool = args.net_numpool

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
        weights[~mask] = 0
        weights = weights / weights.sum()
        ds_loss_weights = weights
        # now wrap the loss
        return MultipleOutputLoss2(loss, ds_loss_weights)

    return loss


class MultipleOutputLoss2(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        """
        super(MultipleOutputLoss2, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss

    def forward(self, x, y):
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors

        l = weights[0] * self.loss(x[0], y[0])
        for i in range(1, len(x)):
            if weights[i] != 0:
                l += weights[i] * self.loss(x[i], y[i])
        return l


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0,
                            self.sum / self.count,
                            self.sum)
