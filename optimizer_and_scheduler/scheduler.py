import torch

from optimizer_and_scheduler.lr_scheduler import LinearWarmupCosineAnnealingLR


def get_scheduler(args, optimizer):
    if args.lrschedule == 'warmup_cosine':
        scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                                  warmup_epochs=args.warmup_epochs,
                                                  max_epochs=args.epochs)
    elif args.lrschedule == 'cosine_anneal':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=args.epochs)
    else:
        scheduler = None
    return scheduler
