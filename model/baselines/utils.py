import numpy as np
import torch
import torch.nn as nn
import pdb


def get_model(args, pretrain=False, n_in=1, n_out=6):
    if args.model == 'vnet':
        from .dim3 import VNet
        if pretrain:
            raise ValueError('No pretrain model available')
        return VNet(inChans=n_in, outChans=n_out, scale=[[1,2,2], [2,2,2], [2,2,2], [2,2,2]], baseChans=16)
    
    elif args.model == 'resunet':
        from .dim3 import UNet
        if pretrain:
            raise ValueError('No pretrain model available')
        return UNet(in_ch=n_in, base_ch=32, num_classes=n_out,
               scale=[[2,2,2], [2,2,2], [2,2,2], [2,2,2]], 
               kernel_size=[[3,3,3], [3,3,3], [3,3,3], [3,3,3], [3,3,3]],
               block='BasicBlock',
               norm='in')
        
    elif args.model == 'unet':
        from .dim3 import UNet
        return UNet(in_ch=n_in, base_ch=32, num_classes=n_out,
               scale=[[2,2,2], [2,2,2], [2,2,2], [2,2,2]], 
               kernel_size=[[3,3,3], [3,3,3], [3,3,3], [3,3,3], [3,3,3]],
               block='SingleConv',
               norm='in')
    
    elif args.model == 'unet++':
        from .dim3 import UNetPlusPlus
        return UNetPlusPlus(in_ch=n_in, base_ch=32,
                            num_classes=n_out, scale=[[1,2,2], [1,2,2], [2,2,2], [2,2,2]], 
                            norm='in', kernel_size=[[1,3,3], [1,3,3], [3,3,3], [3,3,3], [3,3,3]], 
                            block='BasicBlock')
        
    elif args.model == 'attention_unet':
        from .dim3 import AttentionUNet
        return AttentionUNet(in_ch=1, base_ch=32, 
                      scale=[[2,2,2], [2,2,2], [2,2,2], [2,2,2]], 
                      kernel_size=[[3,3,3], [3,3,3], [3,3,3], [3,3,3], [3,3,3]],
                      num_classes=n_out, block='BasicBlock', norm='in')

    elif args.model == 'medformer':
        from .dim3 import MedFormer
        
        return MedFormer(in_chan=n_in, num_classes=n_out, base_chan=32, conv_block='BasicBlock',
                         map_size=[2, 6, 6], 
                         scale=[[1,2,2], [1,2,2], [2,2,2], [2,2,2]], 
                         conv_num=[2,0,0,0, 0,0,2,2], trans_num=[0,2,2,2, 2,2,0,0], 
                         num_heads=[1,4,4,4, 4,4,1,1], fusion_depth=2, fusion_dim=256, 
                         fusion_heads=4, expansion=4, attn_drop=0., proj_drop=0., 
                         proj_type='depthwise', norm='in', act='gelu', 
                         kernel_size=[[1,3,3], [1,3,3], [3,3,3], [3,3,3], [3,3,3]])
        
    elif args.model == 'unetr':
        from .dim3 import UNETR
        model = UNETR(in_channels=n_in, out_channels=n_out, img_size=[128,128,64], 
                      feature_size=16, hidden_size=768, mlp_dim=3072, 
                      num_heads=12, pos_embed='perceptron', norm_name='instance', res_block=True)
        return model
    
    elif args.model == 'swinunetr':
        from monai.networks.nets import SwinUNETR
        model = SwinUNETR(
        img_size=(128, 128, 64),
        in_channels=1,
        out_channels=n_out,
        feature_size=48,
        use_checkpoint=True,
        )
        return model
    
    elif args.model == 'swinattunet':
        from .dim3.swin_att_unet import SwinAttUNET
        model = SwinAttUNET(img_size=(128, 128, 64),
                    in_channels=1, 
                    out_channels=n_out,
                    feature_size=48,
                    use_checkpoint=True,
                    scale=[[2,2,2], [2,2,2], [2,2,2], [2,2,2]],
                    kernel_size=[[3,3,3], [3,3,3], [3,3,3], [3,3,3], [3,3,3]],
                    block='BasicBlock',
                    if_dpsv = False
                    )
        return model
    
    elif args.model == 'swinattunet_t2':
        from .dim3.swin_att_unet_t2 import SwinAttUNET
        model = SwinAttUNET(img_size=(128, 128, 64),
                    in_channels=1, 
                    out_channels=n_out,
                    feature_size=36,
                    use_checkpoint=True,
                    scale=[[2,2,2], [2,2,2], [2,2,2], [2,2,2]],
                    kernel_size=[[3,3,3], [3,3,3], [3,3,3], [3,3,3], [3,3,3]],
                    block='BasicBlock',
                    if_dpsv = False
                    )
        return model