# 
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class ConvNeXt_Downsample_first(nn.Sequential):
    def __init__(self,
                 c1,
                 c2):
        super(ConvNeXt_Downsample_first, self).__init__(
            nn.Conv2d(c1, c2, kernel_size=4, stride=4),
            LayerNorm(c2, eps=1e-6, data_format="channels_first")
        )

class ConvNeXt_Downsample_other(nn.Sequential):
    def __init__(self,
                 c1,
                 c2
                 ):
        super(ConvNeXt_Downsample_other, self).__init__(
            LayerNorm(c1, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(c1, c2, kernel_size=2, stride=2)
        )

class ConvNeXtBlock(nn.Module):
    def __init__(self, c1, c2, drop_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(c1, c2, kernel_size=7, padding=3, groups=c2)  # depthwise conv
        self.norm = LayerNorm(c2, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(c2, 4 * c2)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * c2, c2)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((c2,)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]

        x = shortcut + self.drop_path(x)
        return x

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out
        
nc: 80  # number of classes
depth_multiple: 0.33  # model depth multiple
width_multiple: 0.33  # layer channel multiple
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32

# YOLOv5 v6.0 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, ConvNeXt_Downsample_first, [96]],  # 0-P2/4    160
   [-1, 3, ConvNeXtBlock, [96]],  #                       160
   [-1, 1, ConvNeXt_Downsample_other, [192]], # P2/8      80
   [-1, 3, ConvNeXtBlock, [192]],  #                      80
   [-1, 1, ConvNeXt_Downsample_other, [384]], # P3/16     40
   [-1, 9, ConvNeXtBlock, [384]],  #                     40
   [-1, 1, ConvNeXt_Downsample_other, [768]], # P4/32     20
   [-1, 3, ConvNeXtBlock, [768]],  #                      20
   [-1, 1, SPPF, [768, 5]],  # 8
  ]

# YOLOv5 v6.0 head
head:
  [[-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 5], 1, Concat, [1]],  # cat backbone P3
   [-1, 3, C3, [512, False]],  # 12

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 3], 1, Concat, [1]],  # cat backbone P3
   [-1, 3, C3, [256, False]],  # 16 (P3/8-small)

   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 13], 1, Concat, [1]],  # cat head P4
   [-1, 3, C3, [512, False]],  # 19 (P4/16-medium)

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 9], 1, Concat, [1]],  # cat head P5
   [-1, 3, C3, [1024, False]],  # 22 (P5/32-large)

   [[16, 19, 22], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]
        

import os
import sys
import json
import argparse
import logging
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from utils.WarmUpLR import GradualWarmupScheduler, WarmUpLR

from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter
from utils.dataloder import Dataloader
from utils.utils import read_split_data, read_imagenet_train, read_imagenet_val
from nets.resnet import resnet50
from tqdm import tqdm
# from torchvision.models.resnet import resnet50

def set_logging(name=None, verbose=True):
    for h in logging.root.handlers:
        logging.root.removeHandler(h)
    rank = int(os.getenv('RANK', -1))
    logging.basicConfig(format="%(message)s", level=logging.INFO if (verbose and rank in (-1, 0)) else logging.WARNING)
    return logging.getLogger(name)

def main():
    LOGGER = set_logging(__name__)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    LOGGER.info("using {} device.".format(device))

    batch_size = opt.batch_size
    epochs = opt.epochs

    LOGGER.info(opt)
    LOGGER.info('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter(log_dir=opt.tensorboard_dir)
    if os.path.exists(opt.output_dir) is False:
        os.makedirs(opt.output_dir)

    train_info, num_classes = read_imagenet_train(opt.dataset_dir + '/train')
    val_info = read_imagenet_val(opt.dataset_dir + '/test')
    train_images_path, train_images_label = train_info
    val_images_path, val_images_label = val_info

    assert opt.num_classes == num_classes
    LOGGER.warning('Dataset num_classes {}, but input {}'.format(opt.num_classes, num_classes))

    # -------------------------------------------------------------------- #
    # CIFAR-100      mean: [0.507, 0.487, 0.441] std: [0.267, 0.256, 0.276]
    # ImageNet-1K     mean: [0.485, 0.456, 0.406] std: [0.229, 0.224, 0.225]
    # Park-train     mean: [0.380, 0.376, 0.361] std: [0.208, 0.202, 0.201]
    # Park-test      mean: [0.369, 0.361, 0.333] std: [0.221, 0.214, 0.212]
    # -------------------------------------------------------------------- #
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(384),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.38, 0.376, 0.361], [0.208, 0.202, 0.201])]),
        "val": transforms.Compose([transforms.Resize(384),
                                   transforms.CenterCrop(384),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.38, 0.376, 0.361], [0.208, 0.202, 0.201])])}

    gen_train_data = Dataloader(images_path=train_images_path,
                                images_class=train_images_label,
                                transform=data_transform['train'])
    gen_val_data = Dataloader(images_path=val_images_path,
                              images_class=val_images_label,
                              transform=data_transform['val'])

    train_num = len(gen_train_data)
    val_num = len(gen_val_data)
    LOGGER.info("using {} images for training, {} images for validation.".format(train_num, val_num))


    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    LOGGER.info('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(gen_train_data,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=nw,
                                               collate_fn=gen_train_data.collate_fn)

    validate_loader = torch.utils.data.DataLoader(gen_val_data,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=nw,
                                                  collate_fn=gen_val_data.collate_fn)

    LOGGER.info("using {} batch for training, {} batch for validation.".format(len(train_loader),
                                                                           len(validate_loader)))

    # create model
    net = resnet50(num_classes=opt.num_classes)
    net.to(device)
    net = nn.DataParallel(net)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=opt.lr, momentum=0.937, weight_decay=5e-4)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)
    lf = lambda x: ((1 + math.cos(x * math.pi / opt.epochs)) / 2) * (1 - opt.lrf) + opt.lrf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    top1_best_acc = 0.0
    top1_val_acc = 0.0
    top5_best_acc = 0.0
    top5_val_acc = 0.0
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        LOGGER.info(('\n' + '%10s' * 4) % ('Epoch', 'gpu_mem', 'loss', 'lr'))
        pbar_train = enumerate(train_loader)
        pbar_train = tqdm(pbar_train, total=len(train_loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

        for step, data in pbar_train:
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
            pbar_train.set_description(('%10s' * 2 + '%10.4g' * 2) % (
                f'{epoch + 1}/{opt.epochs}', mem, loss,
                round(optimizer.param_groups[0]['lr'], 5)))

        scheduler.step()

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        acc_top5 = 0.0
        with torch.no_grad():
            LOGGER.info(('%11s' * 3) % ('train_loss', 'acc-top1', 'acc-top5'))
            pbar_val = enumerate(validate_loader)
            pbar_val = tqdm(pbar_val, total=len(validate_loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
            for i, val_data in pbar_val:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]

                maxk = max((1, 5))
                y_resize = val_labels.to(device).view(-1, 1)
                _, pred = outputs.topk(maxk, 1, True, True)

                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                acc_top5 += torch.eq(pred, y_resize).sum().item()

                pbar_val.set_description(('%10.4g' * 3) % ((round(running_loss / len(train_loader), 3)),
                                         round(acc / val_num, 3), round(acc_top5 / val_num, 3)))

        top1_val_acc = acc / val_num
        top5_val_acc = acc_top5 / val_num

        tags = ["loss", "acc-top1", "acc-top5", "learning_rate"]
        tb_writer.add_scalar(tags[0], running_loss / len(train_loader), epoch)
        tb_writer.add_scalar(tags[1], top1_val_acc, epoch)
        tb_writer.add_scalar(tags[2], top5_val_acc, epoch)
        tb_writer.add_scalar(tags[3], optimizer.param_groups[0]["lr"], epoch)

        torch.save(net.state_dict(), opt.output_dir + '/last.pth')

        if top5_val_acc > top5_best_acc:
            top5_best_acc = top5_val_acc

        if top1_val_acc > top1_best_acc:
            top1_best_acc = top1_val_acc
            torch.save(net.state_dict(), opt.output_dir + '/best.pth')

    LOGGER.info('Finished Training')
    LOGGER.info('Best Accuracy - Top-1 %10.4g' % top1_best_acc)
    LOGGER.info('Last Accuracy - Top-1 %10.4g' % top1_val_acc)
    LOGGER.info('Best Accuracy - Top-5 %10.4g' % top5_best_acc)
    LOGGER.info('Last Accuracy - Top-5 %10.4g' % top5_val_acc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lrf', type=float, default=0.1)
    # 是否启用SyncBatchNorm
    parser.add_argument('--syncBN', type=bool, default=True)
    parser.add_argument('--dataset_dir', type=str, default="../dataset/")
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    parser.add_argument('--freeze_layers', type=bool, default=False)
    # 不要改该参数，系统会自动分配
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--tensorboard_dir', default='./tensorboard')
    parser.add_argument('--output_dir', default='./output')
    opt = parser.parse_args()

    main()
