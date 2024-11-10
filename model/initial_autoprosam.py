from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from .Med_SAM.image_encoder import ImageEncoderViT_3d_v2 as ImageEncoderViT_3d
from .Med_SAM.mask_decoder import VIT_MLAHead_h as VIT_MLAHead
from .Med_SAM.prompt_encoder import AutomaticPromptEncoder
from functools import partial
import torch
import torch.nn as nn
    
def init_network(args, n_out=14, device=None):    
    sam = sam_model_registry["vit_b"](checkpoint="./ckpt/sam_vit_b_01ec64.pth")
    mask_generator = SamAutomaticMaskGenerator(sam)
    img_encoder = ImageEncoderViT_3d(
        depth=12,
        embed_dim=768,
        img_size=1024,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=12,
        patch_size=16,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=[2, 5, 8, 11],
        window_size=14,
        cubic_window_size=8,
        out_chans=256,
        num_slice = 16)
    img_encoder.load_state_dict(mask_generator.predictor.model.image_encoder.state_dict(), strict=False)
    del sam
    img_encoder.to(device)

    for p in img_encoder.parameters():
        p.requires_grad = False
    img_encoder.depth_embed.requires_grad = True
    for p in img_encoder.slice_embed.parameters():
        p.requires_grad = True
    for i in img_encoder.blocks:
        for p in i.norm1.parameters():
            p.requires_grad = True
        for p in i.adapter.parameters():
            p.requires_grad = True
        for p in i.norm2.parameters():
            p.requires_grad = True
        i.attn.rel_pos_d = nn.parameter.Parameter(0.5 * (i.attn.rel_pos_h + i.attn.rel_pos_w), requires_grad=True)
    for i in img_encoder.neck_3d:
        for p in i.parameters():
            p.requires_grad = True

    prompt_encoder = AutomaticPromptEncoder(in_ch=256, base_ch=16, num_classes=256,
        scale=[[2,2,2], [2,2,2], [2,2,2], [2,2,2]], 
        kernel_size=[[3,3,3], [3,3,3], [3,3,3], [3,3,3], [3,3,3]],
        block='SingleConv',
        norm='in')
    prompt_encoder.to(device)
    
    mask_decoder = VIT_MLAHead(img_size=96, num_classes=n_out)
    mask_decoder.to(device)
    return img_encoder, prompt_encoder, mask_decoder
