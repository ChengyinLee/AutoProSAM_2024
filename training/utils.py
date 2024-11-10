import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import optim

def get_optimizer(args, net):
    if args.optimizer == 'sgd':
        return optim.SGD(net.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        return optim.Adam(net.parameters(), lr=args.base_lr, betas=args.betas, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        return optim.AdamW(net.parameters(), lr=args.base_lr, betas=args.betas, weight_decay=args.weight_decay, eps=1e-5) # larger eps has better stability during AMP training


def log_evaluation_result(writer, dice_list, ASD_list, HD_list, name, epoch, args):
    C = dice_list.shape[0]

    writer.add_scalar('Dice/%s_AVG'%name, dice_list.mean(), epoch+1)
    for idx in range(C):
        writer.add_scalar('Dice/%s_Dice%d'%(name, idx+1), dice_list[idx], epoch+1)
    writer.add_scalar('ASD/%s_AVG'%name, ASD_list.mean(), epoch+1)
    for idx in range(C):
        writer.add_scalar('ASD/%s_ASD%d'%(name, idx+1), ASD_list[idx], epoch+1)
    writer.add_scalar('HD/%s_AVG'%name, HD_list.mean(), epoch+1)
    for idx in range(C):
        writer.add_scalar('HD/%s_HD%d'%(name, idx+1), HD_list[idx], epoch+1)

def unwrap_model_checkpoint(net, ema_net, args):
    net_state_dict = net.module if args.distributed else net 
    net_state_dict = net_state_dict._orig_mod.state_dict() if args.torch_compile else net_state_dict.state_dict()
    if args.ema:
        if args.distributed:
            ema_net_state_dict = ema_net.module.state_dict()
        else:   
            ema_net_state_dict = ema_net.state_dict()
    else:       
        ema_net_state_dict = None 

    return net_state_dict, ema_net_state_dict

def filter_validation_results(dice_list, ASD_list, HD_list, args):
    if args.dataset == 'amos_mr':
        # the validation set of amos_mr doesn't have the last two organs, so elimiate them
        dice_list, ASD_list, HD_list = dice_list[:-2], ASD_list[:-2], HD_list[:-2]

    return dice_list, ASD_list, HD_list

def multistep_lr_scheduler_with_warmup(optimizer, init_lr, epoch, warmup_epoch, lr_decay_epoch, max_epoch, gamma=0.1):

    if epoch >= 0 and epoch <= warmup_epoch:
        lr = init_lr * 2.718 ** (10*(float(epoch) / float(warmup_epoch) - 1.))
        if epoch == warmup_epoch:
            lr = init_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    flag = False
    for i in range(len(lr_decay_epoch)):
        if epoch == lr_decay_epoch[i]:
            flag = True
            break

    if flag == True:
        lr = init_lr * gamma**(i+1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    else:
        return optimizer.param_groups[0]['lr']

    return lr

def exp_lr_scheduler_with_warmup(optimizer, init_lr, epoch, warmup_epoch, max_epoch):

    if epoch >= 0 and epoch <= warmup_epoch and warmup_epoch != 0:
        lr = init_lr * 2.718 ** (10*(float(epoch) / float(warmup_epoch) - 1.))
        if epoch == warmup_epoch:
            lr = init_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    else:
        lr = init_lr * (1 - epoch / max_epoch)**0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return lr




def update_ema_variables(model, ema_model, alpha, global_step):
    
    alpha = min((1 - 1 / (global_step + 1)), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    for ema_buffer, m_buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.copy_(m_buffer)



@torch.no_grad()
def concat_all_gather(tensor):
    """ 
    Performs all_gather operation on the provided tensor
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


@torch.no_grad()
def remove_wrap_arounds(tensor, ranks):
    """ 
    Due to the DistributedSampler will pad samples for evenly distribute
    samples to gpus, the padded samples need to be removed for right
    evaluation. Need to turn shuffle to False for the dataloader.
    """
    if ranks == 0:
        return tensor

    world_size = dist.get_world_size()
    single_length = len(tensor) // world_size
    output = []

    for rank in range(world_size):
        sub_tensor = tensor[rank * single_length : (rank+1) * single_length]
        if rank >= ranks:
            output.append(sub_tensor[:-1])
        else:
            output.append(sub_tensor)

    output = torch.cat(output)

    return output

import os
import logging
import torch
import torch.distributed as dist
import pdb

LOG_FORMAT = "[%(levelname)s] %(asctime)s %(filename)s:%(lineno)s %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

def configure_logger(rank, log_path=None):
    if log_path:
        log_dir = os.path.dirname(log_path)
        os.makedirs(log_dir, exist_ok=True)

    # only master process will print & write
    level = logging.INFO if rank in {-1, 0} else logging.WARNING
    handlers = [logging.StreamHandler()]
    if rank in {0, -1} and log_path:
        handlers.append(logging.FileHandler(log_path, "w"))

    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=LOG_DATEFMT,
        handlers=handlers,
        force=True,
    )


def save_configure(args):
    if hasattr(args, "distributed"):
        if (args.distributed and is_master(args)) or (not args.distributed):
            with open(f"{args.cp_dir}/config.txt", 'w') as f:
                for name in args.__dict__:
                    f.write(f"{name}: {getattr(args, name)}\n")
    else:
        with open(f"{args.cp_dir}/config.txt", 'w') as f:
            for name in args.__dict__:
                f.write(f"{name}: {getattr(args, name)}\n")
            
def resume_load_optimizer_checkpoint(optimizer, args):
    assert args.load != False, "Please specify the load path with --load"
    
    checkpoint = torch.load(args.load)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def resume_load_model_checkpoint(net, ema_net, args):
    assert args.load != False, "Please specify the load path with --load"
    
    checkpoint = torch.load(args.load)
    net.load_state_dict(checkpoint['model_state_dict'])
    args.start_epoch = checkpoint['epoch']

    if args.ema:
        ema_net.load_state_dict(checkpoint['ema_model_state_dict'])



class AverageMeter(object):
    """ Computes and stores the average and current value """

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
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
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)



class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1)) 
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]" 


def is_master(args):
    return args.rank % args.ngpus_per_node == 0
