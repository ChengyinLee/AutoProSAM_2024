import os
import random
import numpy as np
import time
import pandas as pd
import glob

from monai.transforms import (AsDiscrete,AsDiscreted,EnsureChannelFirstd,Compose,CropForegroundd,LoadImaged,
    Orientationd,RandCropByPosNegLabeld,SaveImaged,ScaleIntensityRanged,ScaleIntensity,Spacingd,Invertd,RandAffined, MapTransform, AddChanneld)
from monai.transforms import (RandFlipd,
    RandShiftIntensityd,
    RandRotate90d)
from monai.transforms import (ThresholdIntensityd, NormalizeIntensityd, ScaleIntensityd)
from monai.networks.layers import Norm 
# from monai.inferers import sliding_window_inference
from training.inference.inference_my import sliding_window_inference_my
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch, PersistentDataset

from monai.metrics import compute_hausdorff_distance, DiceMetric
from monai.losses import DiceLoss, DiceCELoss

import torch
import warnings

from .utils import (
    configure_logger,
    save_configure,
    AverageMeter,
    ProgressMeter,
)
warnings.filterwarnings("ignore", category=UserWarning)

from training.utils import exp_lr_scheduler_with_warmup, log_evaluation_result, get_optimizer
import torch.nn.functional as F
from script_utils import save_checkpoint
from torch.optim import AdamW
import yaml

#get data path lists
def get_data(base_folder="/home/cli6/MyData/amos_ct_3d_tgt_dir/", 
             batch_size=1, return_test=False, num_samples=1, ten_percent=False, fast_val=False):
    
    # training list
    with open(os.path.join(base_folder, 'list', 'dataset.yaml'), 'r') as f:
            img_name_list_train = yaml.load(f, Loader=yaml.SafeLoader)
    
    # validation list
    img_name_list_val = [13, 70, 292, 280, 29, 334, 257, 357, 326, 191, 238, 310, 373, 202, 247, 255, 228, 328, 363, 200, 
                     56, 144, 290, 308, 208, 316, 216, 204, 304, 85, 189, 140, 40, 123, 286, 176, 284, 150, 117, 174, 
                     206, 218, 318, 365, 377, 87, 372, 311, 203, 356, 339, 244, 344, 90, 293, 128, 155, 136, 63, 112, 
                     34, 283, 157, 73, 61, 313, 325, 258, 409, 346, 106, 18, 22, 41, 287, 399, 333, 233, 250, 342, 
                     309, 278, 223, 323, 194, 352, 364, 219, 207, 368, 8, 108, 167, 51, 132, 385, 32, 289, 397, 120]
    
    
    
    
    files = glob.glob(f'{base_folder}/*.nii.gz')

    images = sorted([s for s in files if '_gt' not in s])
    segs = sorted([s for s in files if '_gt' in s])

    train_files = [{"image": f'{base_folder}/{name}.nii.gz', "label": f'{base_folder}/{name}_gt.nii.gz'} for name in img_name_list_train]
    val_files = [{"image": f'{base_folder}/{name}.nii.gz', "label": f'{base_folder}/{name}_gt.nii.gz'} for name in img_name_list_val]
    test_files = val_files
    
    # debug
    if fast_val:
        val_files = val_files[:20]
        print(f'20 samples are used for validation. (fast validation)')

    if ten_percent and fast_val:
        train_files = train_files[:int(len(train_files)*0.1)]
        val_files = val_files[:int(len(val_files)*0.1)]
        test_files = test_files[:int(len(test_files)*0.1)]
        print(f'10% of data is used for training, validation and testing. (debuging)')
        
    
    # for debug
    train_files = train_files
    val_files = val_files[:20]
    
    print('Training files:', len(train_files),'\nValidation files:', len(val_files), '\nTest files:', len(test_files))
    # print(val_files)
    
    
    train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ThresholdIntensityd(keys=['image'], threshold=-991, above=True, cval=-991),
        ThresholdIntensityd(keys=['image'], threshold=362, above=False, cval=362),
        NormalizeIntensityd(keys=['image'], subtrahend=50.0, divisor=141.0),  
        # ScaleIntensityd(keys=["image"] ),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(128, 128, 128),
            pos=2,
            neg=1,
            num_samples=num_samples,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[1],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[2],
            prob=0.10,
        ),
        RandRotate90d(
            keys=["image", "label"],
            prob=0.10,
            max_k=3,
        ),
        # RandShiftIntensityd(
        #     keys=["image"],
        #     offsets=0.10,
        #     prob=0.50,
        # ),
    ]
)
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ThresholdIntensityd(keys=['image'], threshold=-991, above=True, cval=-991),
            ThresholdIntensityd(keys=['image'], threshold=362, above=False, cval=362),
            NormalizeIntensityd(keys=['image'], subtrahend=50.0, divisor=141.0),  
            # ScaleIntensityd(keys=["image"]),
        ]
    )

    test_transforms = val_transforms
    
    
    if not return_test:
        train_ds = CacheDataset(data=train_files, transform=train_transforms,cache_rate=1.0)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        

        val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
        return train_loader, val_loader
    else:
        test_ds = CacheDataset(data=test_files, transform=test_transforms, cache_rate=1.0)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
        return test_loader
        
    # return train_ds, val_ds
    
        

# validaiton function
def validation(img_encoder, prompt_encoder, mask_decoder, val_loader, n_out=16):
    post_label = AsDiscrete(to_onehot=n_out)
    post_pred = AsDiscrete(argmax=True, to_onehot=n_out)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    # dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            val_inputs = batch["image"]
            val_labels = batch["label"]
            with torch.cuda.amp.autocast():
                val_outputs = sliding_window_inference_my(val_inputs, (128, 128, 128), 1, img_encoder, prompt_encoder, mask_decoder, sw_device='cuda', device='cpu')
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [
                post_label(val_label_tensor) for val_label_tensor in val_labels_list
            ]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [
                post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
            ]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
        mean_dice_val = dice_metric.aggregate().item()
        dice_metric.reset()
        # del val_inputs
        # del val_labels
    return mean_dice_val

def train_net(img_encoder, prompt_encoder, mask_decoder, train_loader, val_loader, device, args,
              logger=None, writer=None):
    max_epochs = args.max_epochs
    val_interval = args.val_interval
    
    #Variables to keep track of the net with the best evaluation metric
    epoch_best = 1
    dice_val_best = -1
    
    # criterion = DiceCELoss(to_onehot_y=True, softmax=True, lambda_dice=0.5, lambda_ce=0.5)
    criterion = DiceCELoss(include_background=True, to_onehot_y=True, softmax=True, lambda_dice=0.5, lambda_ce=0.5)
    encoder_opt = AdamW([i for i in img_encoder.parameters() if i.requires_grad==True], lr=args.base_lr, weight_decay=0)
    feature_opt = AdamW(prompt_encoder.parameters(), lr=args.base_lr, weight_decay=0)
    decoder_opt = AdamW([i for i in mask_decoder.parameters() if i.requires_grad == True], lr=args.base_lr, weight_decay=0)
    
    patch_size = args.rand_crop_size[0]
    # training loop
    for epoch in range(max_epochs):
        logger.info(f"epoch {epoch + 1}/{max_epochs}, val_interval:{val_interval}")
        encoder_scheduler = exp_lr_scheduler_with_warmup(encoder_opt, init_lr=args.base_lr, epoch=epoch, warmup_epoch=5, max_epoch=max_epochs)
        feature_scheduler = exp_lr_scheduler_with_warmup(feature_opt, init_lr=args.base_lr, epoch=epoch, warmup_epoch=5, max_epoch=max_epochs)
        decoder_scheduler = exp_lr_scheduler_with_warmup(decoder_opt, init_lr=args.base_lr, epoch=epoch, warmup_epoch=5, max_epoch=max_epochs)
        
        logger.info(f"Current lr in encoder: {encoder_scheduler:.4e}")
        writer.add_scalar('LR encoder', encoder_scheduler, epoch+1)
        img_encoder.train()
        prompt_encoder.train()
        mask_decoder.train()
        
        # train_epoch(train_loader, net, optimizer, device, epoch, writer, logger, criterion)
        batch_time = AverageMeter("Time", ":6.2f")
        epoch_loss = AverageMeter("Loss", ":.2f")
        for idx, batch_data in enumerate(train_loader):
            global_step = idx + epoch * len(train_loader) # global steps
    
            inputs, labels = batch_data["image"],  batch_data["label"] #torch.Size([1, 1, 128, 128, 128])
            out = F.interpolate(inputs.float(), scale_factor=512 / patch_size, mode='trilinear')
            out = out.repeat(1,3,1,1,1)
            input_batch = out.to(device)
            input_batch = input_batch[0].transpose(0, 1)
            batch_features, feature_list = img_encoder(input_batch)
            feature_list.append(batch_features) # 4xtorch.Size([1, 256, 32, 32, 32]), from shallow to deep features
            feature_list = feature_list[::-1] # reverse the list, from deep to shallow features
            # get point based prompts, positive and/or negtive
            new_feature = []
            # using prompt_encoder to get the feaures [3 of previous, one of the encoded with twoway attention]
            for i, feature in enumerate(feature_list):
                if i == 3:
                    new_feature.append(
                        prompt_encoder(feature) #torch.Size([1, 256, 32, 32, 32])
                    )
                else:
                    new_feature.append(feature)
            img_resize = F.interpolate(inputs.permute(0, 1, 3, 4, 2).to(device), scale_factor=64/patch_size,
                mode='trilinear') # torch.Size([1, 1, 64, 64, 64])
            new_feature.append(img_resize)
            masks = mask_decoder(new_feature, 2, patch_size//64)
            masks = masks.permute(0, 1, 4, 2, 3) 
            
            labels = labels.to(device)
            loss = criterion(masks, labels)
            
            encoder_opt.zero_grad()
            decoder_opt.zero_grad()
            feature_opt.zero_grad()
            loss.backward()
            
            epoch_loss.update(loss.item(), inputs.shape[0])          
            logger.info(f"epoch: {epoch}, global_step: {global_step}, iter_loss: {loss.item()}")        
            writer.add_scalar('train_loss', loss.item(), global_step)
            
            torch.nn.utils.clip_grad_norm_(img_encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(mask_decoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(prompt_encoder.parameters(), 1.0)
            encoder_opt.step()
            feature_opt.step()
            decoder_opt.step()
            # break
        
        logger.info(f"epoch {epoch + 1} average loss: {epoch_loss.avg:.7f}")
        writer.add_scalar('train_loss_epoch', epoch_loss.avg, epoch+1)
        
        ########################################################################################
        # Evaluation, save checkpoint and log training info
        
        if (epoch + 1)  % val_interval == 0:
            img_encoder.eval()
            prompt_encoder.eval()
            mask_decoder.eval()
            
            with torch.no_grad():
                dice_val = validation(img_encoder, prompt_encoder, mask_decoder, val_loader)
                logger.info(f"epoch {epoch + 1} average dice: {dice_val:.7f}")
                writer.add_scalar('dice_val', dice_val, epoch+1)
                if dice_val > dice_val_best:
                    dice_val_best = dice_val
                    epoch_best = epoch + 1
                    # torch.save(
                    # net.state_dict(), os.path.join(args.log_path, "best_metric_net.pth")
                    # )
                    is_best = True
                    save_checkpoint({"epoch": epoch + 1,
                        "best_val_dice": dice_val_best,
                         "encoder_dict": img_encoder.state_dict(),
                         "decoder_dict": mask_decoder.state_dict(),
                         "feature_dict": prompt_encoder.state_dict(),
                         "encoder_opt": encoder_opt.state_dict(),
                         "feature_opt": feature_opt.state_dict(),
                         "decoder_opt": decoder_opt.state_dict()
                         },
                        is_best=is_best,
                        checkpoint=args.log_path)
                    
                    logger.info(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                    )
                
    return epoch+1, dice_val_best, epoch_best 