import yaml


def get_config(args):
    with open(args.config_file, 'r') as f:
        yaml_config = yaml.load(f, Loader=yaml.FullLoader)
    for key, value in yaml_config.items():
        if args.__contains__(key):
            args.__setattr__(key, value)
    return args
