import torch
from monai.networks.nets import SwinUNETR
from torch import nn

from networks.nnformer.nnFormer import nnFormer
from networks.nnunet.generic_UNet import Generic_UNet
from networks.segmentationnet.segmentationNet import UNet3D as SegmentationNet
from networks.transbts.TransBTS_downsample8x_skipconnection import TransBTS
from networks.unetr.unetr import UNETR
from networks.vtunet.vtunet import VTUNet
from utils.utils import InitWeights_He


def get_model(args):
    if args.model_name == 'unetr':
        model = UNETR(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            feature_size=args.feature_size,
            hidden_size=args.hidden_size,
            mlp_dim=args.mlp_dim,
            num_heads=args.num_heads,
            pos_embed=args.pos_embed,
            norm_name=args.norm_name,
            conv_block=True,
            res_block=True,
            dropout_rate=args.dropout_rate)

    elif args.model_name == 'vtunet':
        model = VTUNet(args, num_classes=args.out_channels,
                       embed_dim=args.embed_dim,
                       win_size=args.win_size)
        model.load_from(args)

    elif args.model_name == 'transbts':
        model = TransBTS(args, dataset='brats', _conv_repr=True, _pe_type="learned",
                         patch_dim=args.patch_dim)

    elif args.model_name == 'nnunet':
        conv_op = nn.Conv3d
        dropout_op = nn.Dropout3d
        norm_op = nn.InstanceNorm3d
        net_nonlin = nn.LeakyReLU
        model = Generic_UNet(args.in_channels,
                             32,
                             args.out_channels,
                             args.net_numpool,
                             num_conv_per_stage=2,
                             feat_map_mul_on_downscale=2,
                             conv_op=conv_op,
                             norm_op=norm_op,
                             norm_op_kwargs={'eps': 1e-05, 'affine': True},
                             dropout_op=dropout_op,
                             dropout_op_kwargs={'p': 0, 'inplace': True},
                             nonlin=net_nonlin,
                             nonlin_kwargs={'negative_slope': 0.01, 'inplace': True},
                             deep_supervision=args.do_ds,
                             dropout_in_localization=False,
                             final_nonlin=lambda x: x,
                             weightInitializer=InitWeights_He(1e-2),
                             pool_op_kernel_sizes=[[2, 2, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 1]],
                             conv_kernel_sizes=[[3, 3, 1], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
                             upscale_logits=False,
                             convolutional_pooling=True,
                             convolutional_upsampling=True)

    elif args.model_name == 'nnformer':
        model = nnFormer(crop_size=[args.roi_x, args.roi_y, args.roi_z],
                         embedding_dim=args.embed_dim,
                         input_channels=args.in_channels,
                         num_classes=args.out_channels,
                         conv_op=nn.Conv3d,
                         depths=args.depths,
                         num_heads=args.nn_former_num_heads,
                         patch_size=args.patch_size,
                         window_size=args.window_size,
                         down_stride=args.down_stride,
                         deep_supervision=args.do_ds,
                         relative_mult=args.relative_mult)

    elif args.model_name == 'swinunetr':
        model = SwinUNETR(img_size=(args.roi_x, args.roi_y, args.roi_z),
                          in_channels=args.in_channels,
                          out_channels=args.out_channels,
                          feature_size=args.feature_size,
                          drop_rate=0.0,
                          attn_drop_rate=0.0,
                          dropout_path_rate=args.dropout_path_rate,
                          use_checkpoint=True,
                          )

        if args.pretrained_weight is not None:
            model_dict = torch.load(args.pretrained_weight)
            state_dict = model_dict["state_dict"]
            # fine-tuning
            if "module." in list(state_dict.keys())[0]:
                print("Tag 'module.' found in state dict - fixing!")
                for key in list(state_dict.keys()):
                    state_dict[key.replace("module.", "")] = state_dict.pop(key)
            if "swin_vit" in list(state_dict.keys())[0]:
                print("Tag 'swin_vit' found in state dict - fixing!")
                for key in list(state_dict.keys()):
                    state_dict[key.replace("swin_vit", "swinViT")] = state_dict.pop(key)
            # We now load model weights, setting param `strict` to False, i.e.:
            # this load the encoder weights (Swin-ViT, SSL pre-trained), but leaves
            # the decoder weights untouched (CNN UNet decoder).
            model.load_state_dict(state_dict, strict=False)
            print("Using pretrained self-supervised Swin UNETR backbone weights !")

    else:
        model = SegmentationNet(in_channel=args.in_channels, out_channel=args.out_channels)

    return model
