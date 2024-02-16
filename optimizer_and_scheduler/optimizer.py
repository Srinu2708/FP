import torch


def get_optimizer(args, model):
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.optim_lr,
                                     weight_decay=args.reg_weight,
                                     amsgrad=args.amsgrad)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=args.optim_lr,
                                      weight_decay=args.reg_weight)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.optim_lr,
                                    momentum=args.momentum,
                                    nesterov=True,
                                    weight_decay=args.reg_weight)
    return optimizer


def poly_learning_rate(optimizer, epoch, max_epoch, initial_lr, exponent=0.9):
    optimizer.param_groups[0]['lr'] = initial_lr * (1 - epoch / max_epoch) ** exponent


def optimizer_to_device(optim, device):
    # move optimizer to device
    for param in optim.state.values():
        # Not sure if there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
