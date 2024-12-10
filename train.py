#!/usr/bin/env python3
""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import argparse
import importlib
import json
import logging
import os
import time
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from functools import partial
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# import apex

import torch
import torch.nn as nn
import torchvision.utils
# import yaml
from torch.nn.parallel import DistributedDataParallel as NativeDDP

# import torch.optim as optim
import torch.optim as optim

import yaml

from timm import utils
# from timm.data import create_dataset
from timm.data import  create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.layers import convert_splitbn_model, convert_sync_batchnorm, set_fast_norm
from timm.loss import JsdCrossEntropy, SoftTargetCrossEntropy, BinaryCrossEntropy, LabelSmoothingCrossEntropy
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, model_parameters
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2, scheduler_kwargs
from timm.utils import ApexScaler, NativeScaler

from torch.utils.tensorboard import SummaryWriter


try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False


try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

try:
    from functorch.compile import memory_efficient_fusion
    has_functorch = True
except ImportError as e:
    has_functorch = False

# functorch 是 PyTorch 的一个库，专注于为神经网络提供高效的向量化操作和批处理操作。它提供了一些高级的自动化梯度、自动微分和内存优化功能。
# memory_efficient_fusion 可能是 functorch 提供的一种优化机制，减少内存占用和提升计算效率。

has_compile = hasattr(torch, 'compile')


_logger = logging.getLogger('train')

# 这是用于解析 --config 参数的第一个arg parser，目的是用于加载YAML文件
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

# 定义数据转换操作，类似于图像标准化等操作
transform = transforms.Compose([
    transforms.ToTensor(),  # 将PIL图像或numpy数组转换为Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 对数据进行标准化处理
])

# # 加载训练集和验证集
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
#
# valset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
# valloader = DataLoader(valset, batch_size=128, shuffle=False, num_workers=4)
#


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# argparse 是Python标准库中的模块，用于解析命令行参数。在深度学习训练中，可以通过命令行传递模型参数、数据集路径等。
# 使用 argparse.ArgumentParser，我们可以很容易地解析命令行中的各种参数，并在代码中使用这些参数。

# 数据集参数
group = parser.add_argument_group('Dataset parameters')
# 使用 argparse.ArgumentParser 的 add_argument_group() 方法，创建一个参数组，命名为 “Dataset parameters”。将与数据集相关的参数放在一起，方便分类和文档化。
# Keep this argument outside the dataset group because it is positional.

parser.add_argument('data', nargs='?', metavar='DIR', const=None,
                    help='path to dataset (positional is *deprecated*, use --data-dir)')
# 定义了一个位置参数 data，用于指定数据集的路径。
# nargs='?' 表示该参数是可选的。
# metavar='DIR' 用来在帮助信息中显示位置参数的提示（指向数据集目录）。
# const=None 指定默认值为空。
# 注意： 这个位置参数已经被标注为“过时”，建议使用 --data-dir 替代它。

parser.add_argument('--data-dir', metavar='DIR',
                    help='path to dataset (root dir)')
# 这是推荐的用于指定数据集根目录的选项。
# 例如：--data-dir /path/to/dataset 会指定数据集的根目录

parser.add_argument('--dataset', metavar='NAME', default='',
                    help='dataset type + name ("<type>/<name>") (default: ImageFolder or ImageTar if empty)')
# 这是推荐的用于指定数据集根目录的选项。
# 例如：--data-dir /path/to/dataset 会指定数据集的根目录

group.add_argument('--train-split', metavar='NAME', default='train',
                   help='dataset train split (default: train)')
# --train-split 参数用于指定训练集的分割名称，默认值为 train。
# 例如：对于某些数据集，训练数据集可能存在不同的分割，指定该参数可以选择不同的分割进行训练。

group.add_argument('--val-split', metavar='NAME', default='validation',
                   help='dataset validation split (default: validation)')
# --val-split 参数用于指定验证集的分割名称，默认值为 validation。
# 与训练集类似，可以选择不同的分割来验证模型。

parser.add_argument('--train-num-samples', default=None, type=int,
                    metavar='N', help='Manually specify num samples in train split, for IterableDatasets.')
# --train-num-samples 用于手动指定训练集中的样本数量，特别适用于 IterableDatasets 类型的数据集。
# 该参数的默认值为 None，意味着如果不指定，则使用整个数据集。

parser.add_argument('--val-num-samples', default=None, type=int,
                    metavar='N', help='Manually specify num samples in validation split, for IterableDatasets.')
# --val-num-samples 类似于 --train-num-samples，用于手动指定验证集中的样本数量。

group.add_argument('--dataset-download', action='store_true', default=False,
                   help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
# --dataset-download 是一个布尔类型的参数，当指定时（store_true），将允许数据集的下载操作。默认值为 False。

group.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                   help='path to class to idx mapping file (default: "")')
# --class-map 参数用于指定类别到索引的映射文件的路径，帮助将类别名称映射到模型可以理解的标签索引。
# 如果未指定，默认值为空字符串 ""。

group.add_argument('--input-img-mode', default=None, type=str,
                   help='Dataset image conversion mode for input images.')
# --input-img-mode 参数允许用户指定输入图像的转换模式（例如，RGB 或 L 模式）。

group.add_argument('--input-key', default=None, type=str,
                   help='Dataset key for input images.')
# --input-key 参数用于从数据集中提取输入图像的键名称（例如，如果数据集是字典格式）

group.add_argument('--target-key', default=None, type=str,
                   help='Dataset key for target labels.')
# --target-key 参数用于从数据集中提取目标标签的键名称（例如，对于字典格式的数据集）。

# 模型参数
group = parser.add_argument_group('Model parameters')
# 这行代码使用 argparse.ArgumentParser 的 add_argument_group() 方法，创建一个参数组，专门用于定义与模型相关的参数。通过将参数进行分组，有助于代码的可读性和易维护性。

group.add_argument('--model', default='resnet50', type=str, metavar='MODEL',
                   help='Name of model to train (default: "resnet50")')
# --model 参数指定要训练的模型名称，默认值是 resnet50。通过这个参数，用户可以在运行脚本时指定不同的模型。
# 例如：可以通过 --model efficientnet 来指定不同的神经网络模型。

group.add_argument('--pretrained', action='store_true', default=False,
                   help='Start with pretrained version of specified network (if avail)')
# --pretrained 是一个布尔型参数，表示是否使用预训练模型。如果指定了这个参数（store_true），模型将加载预训练的权重，默认为 False。

group.add_argument('--pretrained-path', default=None, type=str,
                   help='Load this checkpoint as if they were the pretrained weights (with adaptation).')
# --pretrained-path 允许用户指定一个预训练模型权重的路径。这些权重可以作为初始权重加载到模型中，并进行微调。

group.add_argument('--initial-checkpoint', default=None, type=str, metavar='PATH',
                   help='Load this checkpoint into model after initialization (default: none)')
# --initial-checkpoint 参数用于指定初始化后加载的模型检查点路径（默认值为空）。这是为了加载已经训练过的模型权重，继续训练或进行模型微调。

group.add_argument('--resume', default=None, type=str, metavar='PATH',
                   help='Resume full model and optimizer state from checkpoint (default: none)')
# --resume 参数用于从一个现有的检查点恢复模型的训练状态，包括模型的权重和优化器的状态。通过恢复之前的状态，可以在中断之后继续训练。

group.add_argument('--no-resume-opt', action='store_true', default=False,
                   help='prevent resume of optimizer state when resuming model')
# --no-resume-opt 用来控制是否恢复优化器的状态。默认为 False，表示当恢复模型时，优化器的状态也会恢复。如果指定了该参数，优化器的状态将不会被恢复。

group.add_argument('--num-classes', type=int, default=None, metavar='N',
                   help='number of label classes (Model default if None)')
# --num-classes 参数用于指定模型的分类数（类别数）。如果没有指定这个参数，模型会使用其默认的类别数。
# 例如：CIFAR-100 有 100 个类别，ImageNet 有 1000 个类别。

group.add_argument('--gp', default=None, type=str, metavar='POOL',
                   help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
# --gp 参数用于指定全局池化（Global Pooling）的类型，选项包括 fast, avg, max, avgmax, avgmaxc 等。如果不指定，将使用模型默认的池化方式。

# group.add_argument('--img-size', type=int, default=None, metavar='N',
#                    help='Image size (default: None => model default)')

group.add_argument('--img-size', type=int, default=32, metavar='N', help='Input image size (CIFAR-100 images are 32x32)')
# --img-size 参数定义输入图像的大小，默认值为 32，这适用于 CIFAR-100 数据集，因为 CIFAR-100 的图像尺寸为 32x32 像素。
# 对于其他数据集，这个参数可以根据需要进行调整，例如：ImageNet 通常使用 224x224 的图像尺寸。

group.add_argument('--in-chans', type=int, default=None, metavar='N',
                   help='Image input channels (default: None => 3)')
# --in-chans 参数用于指定输入图像的通道数，默认值为 None，这意味着模型将使用默认的 3 个通道（RGB 图像）。如果是单通道灰度图像，可以将其设置为 1。

group.add_argument('--input-size', default=None, nargs=3, type=int,
                   metavar='N N N',
                   help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
# --input-size 用于指定输入图像的维度，包括通道数、图像的高度和宽度（如：3 224 224 代表 3 个通道、224x224 大小的图像）。
# 如果没有提供这个参数，模型将使用默认的输入大小。

group.add_argument('--crop-pct', default=None, type=float,
                   metavar='N', help='Input image center crop percent (for validation only)')
# --crop-pct 参数控制验证时的图像中心裁剪百分比。裁剪后的图像尺寸将根据这个百分比进行缩小，以便输入模型进行验证。

group.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                   help='Override mean pixel value of dataset')
# --mean 参数允许用户覆盖数据集的均值值。默认情况下，模型会使用数据集自带的均值进行标准化。

group.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                   help='Override std deviation of dataset')
# --std 参数允许用户覆盖数据集的标准差值。这通常在图像预处理过程中用于标准化图像像素值。

group.add_argument('--interpolation', default='', type=str, metavar='NAME',
                   help='Image resize interpolation type (overrides model)')
# --interpolation 参数用于指定图像缩放时的插值方法，例如 bilinear 或 bicubic 等。如果不指定，将使用模型默认的插值方法。

# group.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
#                    help='Input batch size for training (default: 128)')

group.add_argument('--batch-size', type=int, default=256, metavar='N', help='Input batch size for CIFAR-100 (default: 256)')
# --batch-size 参数定义训练时的批量大小。这里默认为 256，这适用于 CIFAR-100 数据集。不同的数据集和硬件资源可能需要调整这个值以优化训练效率。

group.add_argument('-vb', '--validation-batch-size', type=int, default=None, metavar='N',
                   help='Validation batch size override (default: None)')
# 这个参数用于覆盖验证时的批量大小。默认情况下，验证集的批量大小与训练集相同，但是如果指定了这个参数，它会优先使用这里提供的值。
# 用途：在验证过程中可能不需要使用与训练相同的批量大小，可以通过这个参数调整验证时的效率

group.add_argument('--channels-last', action='store_true', default=False,
                   help='Use channels_last memory layout')
# 如果使用了这个参数，模型将会使用 channels_last 的内存布局（即将通道数放在最后），这在某些硬件和情况下可以提高训练效率。
# 用途：这是为了支持混合精度和提高 GPU 性能。

group.add_argument('--fuser', default='', type=str,
                   help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')")
# --fuser 参数允许用户选择 JIT fuser（Just-In-Time 编译器），这是优化器的一个配置。可以选择如 te, old, nvfuser 等选项。
# 用途：JIT fuser 会影响计算图的编译和执行效率。不同的 fuser 适合不同的硬件和场景。

group.add_argument('--grad-accum-steps', type=int, default=1, metavar='N',
                   help='The number of steps to accumulate gradients (default: 1)')
# 这个参数控制梯度累积的步数。默认是 1，即每个 batch 都会更新梯度。如果指定了一个大于 1 的值，梯度会在多步之后再进行更新。
# 用途：在显存有限的情况下，使用梯度累积可以实现较大的虚拟批量。

group.add_argument('--grad-checkpointing', action='store_true', default=False,
                   help='Enable gradient checkpointing through model blocks/stages')
# 这个参数启用模型块或阶段之间的梯度检查点，用于节省显存。
# 用途：这是为了通过推迟某些计算来节省显存，特别是在深层神经网络中。

group.add_argument('--fast-norm', default=False, action='store_true',
                   help='enable experimental fast-norm')
# 这是一个实验性选项，用于启用更快的归一化操作。
# 用途：优化归一化操作的速度，提高训练效率

group.add_argument('--model-kwargs', nargs='*', default={}, action=utils.ParseKwargs)
# 这个参数允许传递任意关键字参数（kwargs）给模型的构造函数。
# 用途：可以用来向模型传递自定义的配置参数，比如某些模型特有的超参数。

group.add_argument('--head-init-scale', default=None, type=float,
                   help='Head initialization scale')
# --head-init-scale 控制最后一层的初始化缩放比例（通常是分类层）。
# 用途：在某些情况下，可以通过调整这个缩放比例来稳定训练。

group.add_argument('--head-init-bias', default=None, type=float,
                   help='Head initialization bias value')
# 用于控制最后一层的偏置初始化值。
# 用途：调整偏置的初始化以避免模型过快地收敛到某些特定的预测。

group.add_argument('--torchcompile-mode', type=str, default=None,
                    help="torch.compile mode (default: None).")
# 启用 torch.compile 模式，指定要使用的编译后端，默认是 inductor。
# 用途：可以加速模型执行，特别是对于复杂计算图或需要优化的模型架构。

# 脚本与代码生成
scripting_group = group.add_mutually_exclusive_group()
scripting_group.add_argument('--torchscript', dest='torchscript', action='store_true',
                             help='torch.jit.script the full model')
# 启用 torch.jit.script，这会对模型进行脚本化编译。
# 用途：torchscript 可以将模型序列化并加快推理过程，非常适合部署阶段。

scripting_group.add_argument('--torchcompile', nargs='?', type=str, default=None, const='inductor',
                             help="Enable compilation w/ specified backend (default: inductor).")
# 启用 torch.compile 并指定编译的后端，默认是 inductor。
# 用途：使用 PyTorch 的编译功能，加速模型训练和推理。



# 设备与分布式训练
group = parser.add_argument_group('Device parameters')
group.add_argument('--device', default='cuda', type=str,
                    help="Device (accelerator) to use.")
# 指定训练过程中使用的设备，默认是 cuda（即使用 GPU）。如果用户希望在 CPU 上运行，可以将其改为 cpu。
# 用途：允许在不同硬件（如 GPU、CPU）上运行训练。

group.add_argument('--amp', action='store_true', default=False,
                   help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
# 启用自动混合精度（AMP）训练。AMP 使用 NVIDIA Apex 或 PyTorch 的原生 AMP 来实现混合精度。
# 用途：混合精度可以通过在不损失模型性能的情况下减少计算资源，提高训练速度。

group.add_argument('--amp-dtype', default='float16', type=str,
                   help='lower precision AMP dtype (default: float16)')
# 控制混合精度训练中的低精度数据类型，默认是 float16。还可以选择 bfloat16。
# 用途：不同的数据类型对硬件有不同的优化效果。bfloat16 对 TPU 等硬件更友好。

group.add_argument('--amp-impl', default='native', type=str,
                   help='AMP impl to use, "native" or "apex" (default: native)')
# 指定 AMP 的实现方式，可以是 native（PyTorch 原生）或 apex（NVIDIA 提供的 Apex 库）。
# 用途：根据用户安装的库和硬件特性，选择最适合的 AMP 实现。

group.add_argument('--no-ddp-bb', action='store_true', default=False,
                   help='Force broadcast buffers for native DDP to off.')
# 关闭分布式数据并行（DDP）中的广播缓冲区。
# 用途：在某些分布式训练的配置下，可能不需要启用缓冲区广播，这个参数可以关闭它。

group.add_argument('--synchronize-step', action='store_true', default=False,
                   help='torch.cuda.synchronize() end of each step')
# 每个训练步骤后，调用 torch.cuda.synchronize()，确保 GPU 上的操作完成。
# 用途：用于调试和确保在分布式训练中的同步。

group.add_argument("--local_rank", default=0, type=int)
# 控制本地进程的排序编号，常用于分布式训练。
# 用途：分布式训练时，每个进程会被分配一个唯一的 local_rank，以协调它们的操作。

parser.add_argument('--device-modules', default=None, type=str, nargs='+',
                    help="Python imports for device backend modules.")
# 指定需要导入的设备相关模块。
# 用途：支持自定义设备的模块导入，方便在不同硬件上灵活运行


# 优化器参数
group = parser.add_argument_group('Optimizer parameters')
group.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                   help='Optimizer (default: "sgd")')
# 指定训练时使用的优化器类型，默认是 SGD（随机梯度下降）。
# 用途：用户可以选择不同的优化器（如 Adam、AdamW 等），适应不同的训练需求。

parser.add_argument('--optimizer_type', type=str, default='adam', help='选择优化器类型，例如: adam 或 sgd')
#指定优化器类型

group.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                   help='Optimizer Epsilon (default: None, use opt default)')
# eps 是优化器中的小数值，主要用于防止分母为零的情况。
# 用途：对收敛稳定性有重要影响，尤其是对于一些具有动量的优化器。

group.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                   help='Optimizer Betas (default: None, use opt default)')
# 对于某些优化器（如 Adam），betas 控制动量的衰减因子。
# 用途：用户可以自定义 beta 值以更好地控制优化器对梯度动量的处理。

group.add_argument('--momentum', type=float, default=0.9, metavar='M',
                   help='Optimizer momentum (default: 0.9)')
# 动量参数，用于动量型优化器（如 SGD with momentum）。
# 用途：帮助在训练过程中保持梯度方向，提高收敛速度。

group.add_argument('--weight-decay', type=float, default=2e-5,
                   help='weight decay (default: 2e-5)')
# 权重衰减因子，通常用于正则化，防止模型过拟合。
# 用途：帮助在训练过程中减少模型的复杂性，提高泛化能力。

group.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                   help='Clip gradient norm (default: None, no clipping)')
# 设置梯度裁剪的上限，防止梯度爆炸问题。
# 用途：在深度神经网络中，梯度值可能会变得非常大，导致不稳定。通过梯度裁剪可以有效地避免这个问题。

group.add_argument('--clip-mode', type=str, default='norm',
                   help='Gradient clipping mode. One of ("norm", "value", "agc")')
# 指定梯度裁剪的模式，可以是 norm、value 或 agc。
# 用途：控制梯度裁剪的方式，比如通过 L2 范数进行裁剪或者直接设定梯度值的范围。

group.add_argument('--layer-decay', type=float, default=None,
                   help='layer-wise learning rate decay (default: None)')
# 指定层级学习率衰减，即对于网络的不同层设置不同的学习率。
# 用途：通常用在微调过程中，可以让较低层保持较低的学习率，保护其预训练的特征。

group.add_argument('--opt-kwargs', nargs='*', default={}, action=utils.ParseKwargs)
# 用于传递任意关键字参数给优化器。
# 用途：灵活性非常大，可以根据需要传递一些额外的参数，扩展优化器的功能。

# 学习率调度参数
group = parser.add_argument_group('Learning rate schedule parameters')


group.add_argument('--sched', type=str, default='cosine', metavar='SCHEDULER',
                   help='LR scheduler (default: "cosine"')
# 指定学习率调度器，默认是 cosine（余弦退火）。
# 用途：通过学习率调度器控制学习率随时间的变化，余弦退火可以使学习率在训练过程中逐渐减小，从而帮助模型更稳定地收敛。

group.add_argument('--sched-on-updates', action='store_true', default=False,
                   help='Apply LR scheduler step on update instead of epoch end.')
# 设置学习率调度器是否在每次更新时进行步进，而不是在每个 epoch 结束时。
# 用途：在一些场景中，使用每次更新后调度学习率可以获得更精细的学习率调整。

group.add_argument('--lr', type=float, default=None, metavar='LR',
                   help='learning rate, overrides lr-base if set (default: None)')
#
# group.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate for CIFAR-100')
# 设置学习率的值。对于 CIFAR-100 数据集，默认学习率是 0.01。
# 用途：用户可以调整初始学习率来影响训练速度和稳定性。

group.add_argument('--lr-base', type=float, default=0.1, metavar='LR',
                   help='base learning rate: lr = lr_base * global_batch_size / base_size')
# 基础学习率，通常根据全局批量大小和一些其他因素来计算具体学习率。
# 用途：在大规模训练（如使用分布式训练）时，基础学习率的设定和全局批量大小相结合，可以确保学习过程的稳定。

group.add_argument('--lr-base-size', type=int, default=256, metavar='DIV',
                   help='base learning rate batch size (divisor, default: 256).')
# 基础学习率批量大小，这个参数用于调整学习率与批量大小之间的比例。
# 用途：用于根据批量大小调整学习率，保持训练稳定性。

group.add_argument('--lr-base-scale', type=str, default='', metavar='SCALE',
                   help='base learning rate vs batch_size scaling ("linear", "sqrt", based on opt if empty)')
# 基础学习率的比例缩放方式，可以是 linear 或 sqrt。
# 用途：这决定了学习率与批量大小之间的缩放方式。线性比例更适合批量变化不大的情况，而平方根比例则适合大批量的情况下。

group.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                   help='learning rate noise on/off epoch percentages')
# 用于为学习率添加噪声，提升模型的泛化能力。
# 用途：添加一些随机扰动，帮助模型跳出局部最优点。

group.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                   help='learning rate noise limit percent (default: 0.67)')
# 控制学习率噪声的上线百分比。
# 用途：细化对学习率噪声的控制。

group.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                   help='learning rate noise std-dev (default: 1.0)')
# 学习率噪声的标准差。
# 用途：用于控制噪声的强度。

group.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                   help='learning rate cycle len multiplier (default: 1.0)')
# 学习率循环长度的倍乘因子。
# 用途：在有学习率循环策略时，可以通过这个参数增加或减少循环的长度。

group.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT',
                   help='amount to decay each learning rate cycle (default: 0.5)')
# 每次学习率循环后，学习率衰减的倍数。
# 用途：控制学习率循环中的衰减步长，有助于使模型收敛得更加平稳。

group.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                   help='learning rate cycle limit, cycles enabled if > 1')
# 控制学习率循环的次数，循环次数大于 1 才会启用学习率循环。
# 用途：设置学习率循环的上限，防止学习率无限循环。

group.add_argument('--lr-k-decay', type=float, default=1.0,
                   help='learning rate k-decay for cosine/poly (default: 1.0)')
# 学习率衰减因子，用于控制余弦或多项式调度器。
# 用途：进一步控制学习率在衰减过程中变化的形状。

group.add_argument('--warmup-lr', type=float, default=1e-5, metavar='LR',
                   help='warmup learning rate (default: 1e-5)')
# 学习率预热时的学习率初始值。
# 用途：在训练初期，使用较小的学习率可以使模型更稳定地开始学习。

group.add_argument('--min-lr', type=float, default=0, metavar='LR',
                   help='lower lr bound for cyclic schedulers that hit 0 (default: 0)')
# 学习率最小值，适用于周期性调度器（如余弦调度器）。
# 用途：确保学习率不会衰减到接近于零，避免模型完全不学习。

group.add_argument('--epochs', type=int, default=300, metavar='N',
                   help='number of epochs to train (default: 66)')
# 训练的总 epoch 数，默认是 66。
# 用途：用户可以设置训练的时长来确保模型达到预期的性能。

group.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                   help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
# 控制每个 epoch 数据集重复的倍数。
# 用途：数据重复训练可以帮助模型更好地拟合，但需要小心避免过拟合。

group.add_argument('--start-epoch', default=None, type=int, metavar='N',
                   help='manual epoch number (useful on restarts)')
# 学习率预热阶段的 epoch 数量。
# 用途：预热期的学习率从小到大逐渐增加，可以让模型在稳定阶段前更好地调整参数。

group.add_argument('--decay-milestones', default=[90, 180, 270], type=int, nargs='+', metavar="MILESTONES",
                   help='list of decay epoch indices for multistep lr. must be increasing')
# 指定学习率衰减的 epoch 时刻，值为一个整数列表。
# 默认情况下，如果用户没有提供值，则会使用 [90, 180, 270] 作为默认的衰减点。nargs='+' 确保用户可以传递多个 epoch 值，type=int 确保这些值必须是整数。

group.add_argument('--decay-epochs', type=float, default=90, metavar='N',
                   help='epoch interval to decay LR')
# 学习率预热阶段的 epoch 数量。
# 用途：预热期的学习率从小到大逐渐增加，可以让模型在稳定阶段前更好地调整参数。

group.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                   help='epochs to warmup LR, if scheduler supports')
# 是否将预热阶段从学习率衰减计划中剥离。
# 用途：可以更灵活地调整学习率预热和学习率衰减之间的关系。

group.add_argument('--warmup-prefix', action='store_true', default=False,
                   help='Exclude warmup period from decay schedule.'),
# 是否将预热阶段从学习率衰减计划中剥离。
# 用途：可以更灵活地调整学习率预热和学习率衰减之间的关系。

group.add_argument('--cooldown-epochs', type=int, default=0, metavar='N',
                   help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
# 在循环结束后，学习率在最低点的冷却阶段的 epoch 数。
# 用途：有助于在最后阶段稳定模型，使其尽可能接近最优点。

group.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                   help='patience epochs for Plateau LR scheduler (default: 10)')
# 在平稳学习率策略下，如果模型效果没有提升，允许继续训练的 epoch 数。
# 用途：防止学习率提前衰减，在模型未达到期望性能时保持学习率不变。

group.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                   help='LR decay rate (default: 0.1)')
# 学习率衰减因子，用于调整每次衰减的步长。
# 用途：通过设置衰减率，灵活控制学习率衰减的力度。


# 数据增强和正则化参数
group = parser.add_argument_group('Augmentation and regularization parameters')


group.add_argument('--no-aug', action='store_true', default=False,
                   help='Disable all training augmentation, override other train aug args')
# 作用：禁用所有训练数据的增强操作。
# 用途：如果设为 True，将关闭所有数据增强，这在测试或特殊训练情况下可能很有用。

group.add_argument('--train-crop-mode', type=str, default=None,
                   help='Crop-mode in train'),
# 作用：指定训练时裁剪的模式。
# 用途：可以设定图像裁剪的模式，比如 center（中心裁剪）或者 random（随机裁剪），控制输入的图像尺寸。

group.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                   help='Random resize scale (default: 0.08 1.0)')
#作用：控制随机缩放的比例范围。
# 用途：随机调整图像的尺寸，以一定比例范围缩放（0.08 到 1.0），从而提高模型的鲁棒性和泛化能力。

group.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                   help='Random resize aspect ratio (default: 0.75 1.33)')
#作用：控制图像随机裁剪的宽高比范围。
# 用途：通过随机调整宽高比（比如从 0.75 到 1.33），模型可以看到更多样化的视角，减少对某些特定几何形状的依赖。

group.add_argument('--hflip', type=float, default=0.5,
                   help='Horizontal flip training aug probability')
#作用：水平翻转图像的概率，默认值是 0.5。
# 用途：水平翻转是常用的数据增强手段，有助于平衡数据中的左右偏差，提高模型的泛化能力。

group.add_argument('--vflip', type=float, default=0.,
                   help='Vertical flip training aug probability')
# 作用：垂直翻转图像的概率，默认值是 0.0（不使用）。
# 用途：虽然垂直翻转通常不适用于自然场景，但在某些情况下（如物体检测任务）可能有用。

group.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                   help='Color jitter factor (default: 0.4)')
# 作用：颜色扰动的幅度因子，默认值为 0.4。
# 用途：随机改变亮度、对比度、饱和度等，以防止模型过度依赖于特定的颜色特征。

group.add_argument('--color-jitter-prob', type=float, default=None, metavar='PCT',
                   help='Probability of applying any color jitter.')
# 作用：控制应用颜色扰动的概率。
# 用途：有时候需要指定颜色扰动的应用概率，以增加训练过程中的随机性。

group.add_argument('--grayscale-prob', type=float, default=None, metavar='PCT',
                   help='Probability of applying random grayscale conversion.')
# 作用：随机将图像转换为灰度的概率。
# 用途：灰度化可以使模型不依赖于特定颜色特征，尤其对于某些不依赖颜色信息的任务很有用。

group.add_argument('--gaussian-blur-prob', type=float, default=None, metavar='PCT',
                   help='Probability of applying gaussian blur.')
# 作用：随机对图像应用高斯模糊的概率。
# 用途：模糊效果用于模拟不同的相机或环境条件，提高模型应对模糊输入的能力。

group.add_argument('--aa', type=str, default=None, metavar='NAME',
                   help='Use AutoAugment policy. "v0" or "original". (default: None)'),
# 作用：指定是否使用 AutoAugment 策略，v0 或 original。
# 用途：AutoAugment 是一种自动搜索增强策略的技术，可以在训练过程中选择最有效的数据增强方法。

group.add_argument('--aug-repeats', type=float, default=0,
                   help='Number of augmentation repetitions (distributed training only) (default: 0)')
# 作用：指定数据增强的重复次数。
# 用途：在分布式训练中重复增强操作，以增加模型接触的样本多样性。

group.add_argument('--aug-splits', type=int, default=0,
                   help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
# 作用：指定数据增强的分割次数。
# 用途：在一些特殊的训练策略（如 Jensen-Shannon Divergence loss）中可能需要将增强的数据分为多个部分。

group.add_argument('--jsd-loss', action='store_true', default=False,
                   help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
# 作用：启用 Jensen-Shannon Divergence (JSD) 加上交叉熵 (Cross Entropy, CE) 的组合损失。
# 用途：使用 JSD 损失有助于模型在使用数据增强时更好地进行多视角融合，使得模型更加稳健。通常需要配合 --aug-splits 参数使用。

group.add_argument('--bce-loss', action='store_true', default=False,
                   help='Enable BCE loss w/ Mixup/CutMix use.')
# 作用：启用带有 Mixup/CutMix 数据增强的二值交叉熵 (Binary Cross Entropy, BCE) 损失。
# 用途：在一些场景中使用二值交叉熵比普通交叉熵更有效，特别是当目标标签是平滑的，或者是多标签问题时。

group.add_argument('--bce-sum', action='store_true', default=False,
                   help='Sum over classes when using BCE loss.')
# 作用：在使用 BCE 损失时，对所有类的损失求和。
# 用途：通过将多个类别的损失求和，使得在多类情况下 BCE 损失的计算更加合适。

group.add_argument('--bce-target-thresh', type=float, default=None,
                   help='Threshold for binarizing softened BCE targets (default: None, disabled).')
# 作用：设置用于二值化 BCE 目标的阈值。
# 用途：在某些情况下，可以对 BCE 损失的目标进行二值化操作，以更好地处理一些复杂任务。

group.add_argument('--bce-pos-weight', type=float, default=None,
                   help='Positive weighting for BCE loss.')
# 作用：对 BCE 损失中的正样本进行加权。
# 用途：适用于处理不平衡的数据集，使得正样本在训练过程中占据更大的权重，降低类别不平衡对模型训练的影响。

group.add_argument('--reprob', type=float, default=0., metavar='PCT',
                   help='Random erase prob (default: 0.)')
# 作用：随机擦除的概率。
# 用途：随机擦除增强是为了随机擦除图像中的部分区域，从而让模型在面对部分遮挡时也能取得良好表现。

group.add_argument('--remode', type=str, default='pixel',
                   help='Random erase mode (default: "pixel")')
# 作用：指定随机擦除的模式，比如用单个像素值 (pixel) 或者其他模式。
# 用途：随机擦除增强的细节控制，如擦除后的区域是用随机像素值填充还是用特定颜色填充。

group.add_argument('--recount', type=int, default=1,
                   help='Random erase count (default: 1)')
# 作用：指定随机擦除增强时需要执行的次数。
# 用途：控制随机擦除的次数，增加或减少被遮挡区域的个数。

group.add_argument('--resplit', action='store_true', default=False,
                   help='Do not random erase first (clean) augmentation split')
# 作用：指定随机擦除增强时需要执行的次数。作用：在第一次增强时不进行随机擦除。
# 用途：保证某些增强分割（如 clean augmentation split）不受随机擦除影响。

group.add_argument('--mixup', type=float, default=0.0,
                   help='mixup alpha, mixup enabled if > 0. (default: 0.)')
# 作用：mixup 数据增强的 alpha 参数，用于控制混合强度。
# 用途：mixup 是一种数据增强技术，它将两个不同的样本进行线性混合，并混合它们的标签，有助于使模型不那么敏感于具体的样本。

group.add_argument('--cutmix', type=float, default=0.0,
                   help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
# 作用：指定 cutmix 的最小和最大比例，以覆盖 alpha 值。作用：cutmix 数据增强的 alpha 参数，用于控制混合强度。
# 用途：cutmix 是另一种数据增强策略，通过切割并混合两个样本，将它们不同的部分合并。与 mixup 类似，它也可以提升模型的泛化能力。

group.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                   help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
# 作用：指定 cutmix 的最小和最大比例，以覆盖 alpha 值。
# 用途：可以进一步精确控制 cutmix 的剪切和混合比例。

group.add_argument('--mixup-prob', type=float, default=1.0,
                   help='Probability of performing mixup or cutmix when either/both is enabled')
# 作用：执行 mixup 或 cutmix 数据增强的概率。
# 用途：可以对增强的应用概率进行控制，以增加训练过程中增强的随机性。

group.add_argument('--mixup-switch-prob', type=float, default=0.5,
                   help='Probability of switching to cutmix when both mixup and cutmix enabled')
# 作用：当 mixup 和 cutmix 都启用时，执行 cutmix 的概率。
# 用途：在 mixup 和 cutmix 都启用的情况下，这个参数可以控制不同增强策略之间的随机选择。

group.add_argument('--mixup-mode', type=str, default='batch',
                   help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
# 作用：决定如何应用 mixup 或 cutmix 参数。
# 用途： batch：将整个批次作为单位应用 mixup/cutmix。pair：成对混合。elem：针对每个元素进行混合。这些不同的模式有助于产生更多数据增强的多样性。

group.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                   help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
# 作用：指定 mixup 在第几轮迭代后关闭。
# 用途：用于控制数据增强在训练后期是否继续使用，以避免过多的噪声影响模型的收敛。

group.add_argument('--smoothing', type=float, default=0.1,
                   help='Label smoothing (default: 0.1)')
# 作用：标签平滑值，通常用于减少模型的过拟合。
# 用途：通过将标签从硬编码（如 [0, 1]）转换为软编码值（如 [0.05, 0.95]），标签平滑有助于让模型更稳健，减少过拟合。

group.add_argument('--train-interpolation', type=str, default='random',
                   help='Training interpolation (random, bilinear, bicubic default: "random")')
# 选项：random、bilinear、bicubic
# 用途：定义在训练时图像缩放时的插值方式。使用不同的插值方式可能会影响图像的细节表现。

group.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                   help='Dropout rate (default: 0.)')
# 作用：指定 Dropout 的比例。
# 用途：Dropout 是一种正则化方法，随机地将一部分神经元的输出置为零，减少过拟合。

group.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                   help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
# 作用：指定 Drop Connect 的比例，不过这个参数已经不推荐使用，建议用 drop-path 。
# 用途：Drop Connect 类似于 Dropout，不过它作用在连接上，而不是神经元上。

group.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                   help='Drop path rate (default: None)')
# 作用：指定 Drop Path 的比例。
# 用途：Drop Path 是在网络中随机丢弃某些路径的技术，特别适用于深度残差网络（ResNet）和 Transformer 类结构。

group.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                   help='Drop block rate (default: None)')
# 作用：指定 Drop Block 的比例。
# 用途：Drop Block 类似于 Dropout，但作用于整个小的空间块，尤其适用于图像输入的模型。


# Batch norm parameters (only works with gen_efficientnet based models currently)
# Batch Normalization（批量归一化）的参数。
group = parser.add_argument_group('Batch norm parameters', 'Only works with gen_efficientnet based models currently.')


group.add_argument('--bn-momentum', type=float, default=None,
                   help='BatchNorm momentum override (if not None)')
# 作用：覆盖 BatchNorm 层的动量参数。
# 用途：BatchNorm 的动量决定了如何平滑更新均值和方差，修改该参数可以控制归一化更新的平滑程度。

group.add_argument('--bn-eps', type=float, default=None,
                   help='BatchNorm epsilon override (if not None)')
# 作用：覆盖 BatchNorm 层的 epsilon 参数。
# 用途：BatchNorm 中的 epsilon 用于避免分母为零，通常设置一个非常小的值，用于稳定数值计算。

group.add_argument('--sync-bn', action='store_true',
                   help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
# 作用：启用同步 Batch Normalization。
# 用途：在分布式训练中使得不同 GPU 之间的 BatchNorm 统计量同步，适用于分布式数据并行场景。

group.add_argument('--dist-bn', type=str, default='reduce',
                   help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
# 选项：broadcast、reduce、空字符串
# 用途：用于指定在分布式训练中如何共享和同步 BatchNorm 的统计量。

group.add_argument('--split-bn', action='store_true',
                   help='Enable separate BN layers per augmentation split.')
# 作用：为每个数据增强部分启用单独的 BatchNorm 层。
# 用途：在某些复杂的数据增强（如 AugMix）的设置中，需要为不同的数据增强路径使用不同的 BatchNorm，以便统计不同增强方式下的数据分布。



# Model Exponential Moving Average
# 指数移动平均 (Exponential Moving Average, EMA) 参数
group = parser.add_argument_group('Model exponential moving average parameters')
group.add_argument('--model-ema', action='store_true', default=False,
                   help='Enable tracking moving average of model weights.')
# 作用：启用模型权重的指数移动平均。
# 用途：通过跟踪模型权重的移动平均，EMA 提供一个稳定的模型版本，可能比标准训练的模型具有更好的泛化能力。

group.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                   help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
# 作用：将 EMA 操作强制在 CPU 上进行。
# 用途：为了节省 GPU 的内存，可以选择在 CPU 上进行 EMA 计算。如果指定，则只在 rank=0 的节点上追踪 EMA，且不进行 EMA 验证。

group.add_argument('--model-ema-decay', type=float, default=0.9998,
                   help='Decay factor for model weights moving average (default: 0.9998)')
# 作用：模型权重 EMA 的衰减因子。
# 用途：决定 EMA 的更新速度。衰减因子越接近于 1.0，历史模型权重对当前权重的影响就越大，更新就越平滑。

group.add_argument('--model-ema-warmup', action='store_true',
                   help='Enable warmup for model EMA decay.')
# 作用：启用模型 EMA 衰减的热身。
# 用途：在训练初期，模型参数变化较大，启用热身可以让 EMA 逐步生效，避免初期 EMA 权重偏差过大。


# Misc
# 杂项参数


group = parser.add_argument_group('Miscellaneous parameters')


group.add_argument('--seed', type=int, default=42, metavar='S',
                   help='random seed (default: 42)')
# 作用：设置随机数种子。
# 用途：确保实验的可重复性。通过设置相同的种子值，能够让模型每次训练使用相同的随机初始化和数据顺序。

group.add_argument('--worker-seeding', type=str, default='all',
                   help='worker seed mode (default: all)')
# 作用：定义工作线程的种子模式。
# 用途：在多线程数据加载时，定义不同 worker 的种子如何设置，确保数据处理的一致性。

group.add_argument('--log-interval', type=int, default=50, metavar='N',
                   help='how many batches to wait before logging training status')
# 作用：记录训练状态的间隔批次数。
# 用途：每训练 50 个批次，记录一次训练日志，以便监控训练过程。

group.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                   help='how many batches to wait before writing recovery checkpoint')
# 作用：恢复检查点的写入间隔批次数。
# 用途：用于意外中断时保存模型状态，以便之后恢复。

group.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
                   help='number of checkpoints to keep (default: 10)')
# 作用：要保留的检查点的数量。
# 用途：为了节省存储空间，通常只保留最近的几个检查点，而不是全部。

group.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                   help='how many training processes to use (default: 4)')
#作用：用于数据加载的工作线程数量。
# 用途：多线程数据加载可以加快数据的读取速度，提高训练效率。

group.add_argument('--save-images', action='store_true', default=False,
                   help='save images of input bathes every log interval for debugging')
# 作用：在每次日志记录间隔时保存输入批次的图像。
# 用途：在调试时查看输入数据是否正常，可以帮助发现数据预处理问题。

group.add_argument('--pin-mem', action='store_true', default=False,
                   help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
# 作用：在数据加载器中固定 CPU 内存。
# 用途：当数据从 CPU 传输到 GPU 时，固定内存可以提升数据传输效率，特别是在 GPU 数量较多的系统上。

group.add_argument('--no-prefetcher', action='store_true', default=False,
                   help='disable fast prefetcher')
# 作用：禁用快速预取器。
# 用途：在加载数据时，如果不使用预取器，可以减少内存的压力，但是可能导致数据加载的速度变慢。如果数据处理比较复杂或者内存有限，可以考虑禁用预取器。

group.add_argument('--output', default='', type=str, metavar='PATH',
                   help='path to output folder (default: none, current dir)')
# 作用：指定模型输出结果的路径。
# 用途：用于存储训练过程中生成的日志文件、检查点、模型权重等。如果不指定，默认存储在当前目录。

group.add_argument('--experiment', default='', type=str, metavar='NAME',
                   help='name of train experiment, name of sub-folder for output')
# 作用：定义训练实验的名称。
# 用途：为不同实验设置不同的名称，便于区分不同的输出结果。每个实验会在输出文件夹中创建一个对应的子文件夹。

group.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                   help='Best metric (default: "准确率@1"')
# 作用：指定用于评估模型性能的最佳度量指标。
# 用途：在评估模型时，可以使用不同的度量标准，如 top1、top5 等，便于监控不同实验的效果。

group.add_argument('--tta', type=int, default=0, metavar='N',
                   help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
# 作用：定义测试/推理阶段的数据增强因子。
# 用途：推理阶段的数据增强可以通过对输入数据进行多次采样，然后对结果进行平均，提高模型的泛化能力。例如，可以在测试时对图像进行不同的旋转和裁剪，然后得到预测的平均值。

group.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                   help='use the multi-epochs-loader to save time at the beginning of every epoch')
# 作用：启用多时期加载器，以在每个时期开始时节省时间。
# 用途：多时期加载器可以避免每个 epoch 都重新加载数据，而是持续使用相同的数据加载器，减少每个训练周期的初始化时间。

group.add_argument('--log-wandb', action='store_true', default=False,
                   help='log training and validation metrics to wandb')
#作用：将训练和验证的度量指标记录到 Weights and Biases (wandb)。
# 用途：Weights and Biases 是一种流行的实验管理工具，通过这个选项，可以将训练过程中的度量指标记录到 wandb 上，便于可视化和共享实验结果。



# def _parse_args():
#     # Do we have a config file to parse?
#     args_config, remaining = config_parser.parse_known_args()
#     if args_config.config:                                              #检查 args_config 中是否有配置文件路径。如果有，则继续读取这个配置文件，否则跳过。
#         with open(args_config.config, 'r', encoding='utf-8') as f:
#             cfg = yaml.safe_load(f)
#             parser.set_defaults(**cfg)
#
#     # The main arg parser parses the rest of the args, the usual
#     # defaults will have been overridden if config file specified.
#     args = parser.parse_args(remaining)
#     # 使用主解析器parser对剩余的命令行参数进行解析。
#     # 参数的优先级是命令行参数高于配置文件参数，即如果在命令行传递了与配置文件中相同的参数，则命令行参数的值会覆盖配置文件中的默认值。
#
#     # Cache the args as a text string to save them in the output dir later
#     args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
#     return args, args_text




# 假设存在 config.yaml 文件
CONFIG_FILE_PATH = "config.yaml"

def _parse_args():
    # 直接加载配置文件，而不是通过命令行指定
    with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # 使用默认的 argparse 配置
    parser.set_defaults(**cfg)

    # 如果需要进一步从代码中调整参数，可以在这里直接设置
    # 例如，强制设定一些特定的参数
    parser.set_defaults(some_key='some_value')

    # 解析命令行参数，但这里只考虑默认值，不使用命令行输入
    args = parser.parse_args([])

    # 缓存 args 为字符串，用于后续保存到输出文件夹等
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    return args, args_text



def main():

    utils.setup_default_logging()
    args, args_text = _parse_args()

    if args.device_modules:
        for module in args.device_modules:
            importlib.import_module(module)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print('已启用cuda')

    args.prefetcher = not args.no_prefetcher
    args.grad_accum_steps = max(1, args.grad_accum_steps)
    device = utils.init_distributed_device(args)
    if args.distributed:
        _logger.info(
            '分布式多进程训练模式，一个设备分配一个进程'
            f'Process {args.rank}, total {args.world_size}, device {args.device}.')
    else:
        _logger.info( f'在单设备上使用({args.device})进行训练')
    assert args.rank >= 0

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    amp_dtype = torch.float16
    if args.amp:
        if args.amp_impl == 'apex':
            assert has_apex, ('选择 APEX，但没有安装 has_apex 模块')
            use_amp = 'apex'
            assert args.amp_dtype == 'float16'
        else:
            assert has_native_amp, 'Please update PyTorch to a version with native AMP (or use APEX).'
            use_amp = 'native'
            assert args.amp_dtype in ('float16', 'bfloat16')
        if args.amp_dtype == 'bfloat16':
            amp_dtype = torch.bfloat16

    utils.random_seed(args.seed, args.rank)

    if args.fuser:
        utils.set_jit_fuser(args.fuser)
    if args.fast_norm:
        set_fast_norm()

    in_chans = 3
    if args.in_chans is not None:
        in_chans = args.in_chans
    elif args.input_size is not None:
        in_chans = args.input_size[0]

    factory_kwargs = {}
    if args.pretrained_path:
        # merge with pretrained_cfg of model, 'file' has priority over 'url' and 'hf_hub'.
        factory_kwargs['pretrained_cfg_overlay'] = dict(
            file=args.pretrained_path,
            num_classes=-1,  # force head adaptation
        )

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        in_chans=in_chans,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        checkpoint_path=args.initial_checkpoint,
        **factory_kwargs,
        **args.model_kwargs,
    # args.model：指定模型名称，比如resnet50。
    # pretrained = args.pretrained：是否使用预训练权重，取决于args.pretrained。
    # in_chans：输入通道数。
    # num_classes：输出分类类别数。
    # drop_rate、drop_path_rate、drop_block_rate：分别表示dropout、droppath和 drop block的丢弃概率。
    # global_pool：指定全局池化类型。
    # bn_momentum 和 bn_eps：用于控制批归一化层的动量和 epsilon值。
    # scriptable：是否可以通过TorchScript转换。
    # checkpoint_path：加载初始检查点的路径，用于继续训练之前保存的模型。
    # factory_kwargs和args.model_kwargs：通过字典传入额外的模型参数。
    )

    if args.head_init_scale is not None:
        with torch.no_grad():
            model.get_classifier().weight.mul_(args.head_init_scale)
            model.get_classifier().bias.mul_(args.head_init_scale)
            # 如果设置了 head_init_scale，则初始化分类层的权重和偏置。
            # 使用 torch.no_grad() 防止这些操作影响到梯度计算。
            # model.get_classifier()获取模型的分类头部（通常是一个全连接层），对它的weight和bias进行缩放，以便控制初始化的权重大小。
            # 这样做的目的是让分类层的权重在某些情况下更稳定，特别是当你加载预训练模型并希望调整分类任务时。

    if args.head_init_bias is not None:
        nn.init.constant_(model.get_classifier().bias, args.head_init_bias)
        # 如果设置了 head_init_bias，则将分类器的偏置初始化为指定的值。这通常用于通过一个固定值初始化偏置，使模型在某些训练任务中能够更快收敛。

    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes  # FIXME handle model default vs config num_classes more elegantly
    # 检查 args.num_classes 是否为空，如果为空则从模型中获取默认的 num_classes。

    if args.grad_checkpointing:
        model.set_grad_checkpointing(enable=True)
    # 如果开启了梯度检查点选项，则在模型中启用梯度检查点功能

    if utils.is_primary(args):
        _logger.info(
            f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')
    # 如果当前进程是主进程（分布式训练场景下），则输出日志信息，记录模型名称和参数总数量。

    data_config = resolve_data_config(vars(args), model=model, verbose=utils.is_primary(args))
    # resolve_data_config()用于解析数据的相关配置，例如输入图像的大小、均值和标准差等。
    # vars(args)将 args 对象转化为字典格式，以便函数使用。
    # model = model将模型传入，以便根据模型的特性解析合适的数据配置。
    # verbose = utils.is_primary(args)控制是否输出详细的解析信息，只在主进程中输出。


    # setup augmentation batch splits for contrastive loss or split bn
    # 设置用于对比损失或分割 Batch Normalization 的数据增强分割数
    num_aug_splits = 0
    if args.aug_splits > 0:
        if args.aug_splits <= 1:
            raise ValueError('分割数必须大于 1 才有意义，请重新设置 args.aug_splits。')
        num_aug_splits = args.aug_splits

    # enable split bn (separate bn stats per batch-portion)
    # 启用分割批标准化（对每个批次部分使用独立的 BN 统计数据）
    if args.split_bn:
        # 确保分割数大于 1 或者启用了强制重新分割选项
        assert num_aug_splits > 1 or args.resplit
        # 转换模型，使其支持分割 批标准化，分割数至少为 2
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    # move model to GPU, enable channels last layout if set
    # 将模型移动到 GPU 上，如果设置了 channels last 则启用该布局
    model.to(device=device)
    if args.channels_last:
        model.to(memory_format=torch.channels_last)

    # setup synchronized BatchNorm for distributed training
    # 为分布式训练设置同步 BatchNorm
    if args.distributed and args.sync_bn:
        args.dist_bn = ''                                # disable dist_bn when sync BN active 启用同步 BN 时禁用 dist_bn
        assert not args.split_bn                         # 确保没有启用 split BN
        if has_apex and use_amp == 'apex':
            # 使用 Apex AMP 进行同步 BN
            # 警告：这可能无法与包含 BatchNormAct2d 的模型兼容
            model = convert_syncbn_model(model)
        else:
            # 使用 PyTorch 自带的同步 BatchNorm
            model = convert_sync_batchnorm(model)
        # 如果是主程序，记录转换同步 BatchNorm 的日志
        if utils.is_primary(args):
            _logger.info(
                '已将模型转换为使用同步 BatchNorm。警告：如果使用零初始化的 BN 层（默认 ResNet），'
                '可能在启用同步 BN 时出现问题。')


    # 如果启用了 torchscript，将模型转换为 TorchScript 格式
    if args.torchscript:
        assert not args.torchcompile                                                                    # 禁止同时使用 torchscript 和 torchcompile
        assert not use_amp == 'apex','不能将 APEX AMP 与 TorchScript 模型一起使用'
        assert not args.sync_bn, '不能将 SyncBatchNorm 与 TorchScript 模型一起使用'
        model = torch.jit.script(model)

    # 如果没有指定学习率，自动计算学习率
    if not args.lr:
        print('没有指定学习率，将自动计算学习率')
        global_batch_size = args.batch_size * args.world_size * args.grad_accum_steps                   # 计算全局批次大小
        batch_ratio = global_batch_size / args.lr_base_size                                             # 计算批次比率
        if not args.lr_base_scale:
            on = args.opt.lower()
            args.lr_base_scale = 'sqrt' if any([o in on for o in ('ada', 'lamb')]) else 'linear'        # 根据优化器类型设置缩放方式
        if args.lr_base_scale == 'sqrt':
            batch_ratio = batch_ratio ** 0.5                                                            # 如果是 sqrt 缩放，取比率的平方根
        args.lr = args.lr_base * batch_ratio                                                            # 计算最终学习率
        if utils.is_primary(args):
            _logger.info(
                f'从基础学习率 ({args.lr_base})  计算得到学习率  ({args.lr})'
                f'全局批次大小为 ({global_batch_size}) 使用了 {args.lr_base_scale} 缩放方式.')

    optimizer = create_optimizer_v2(
        model,
        **optimizer_kwargs(cfg=args),                   # 优化器配置参数
        **args.opt_kwargs,                              # 其他优化器参数
    )


    # setup automatic mixed-precision (AMP) loss scaling and op casting
    # 设置自动混合精度 (AMP) 的损失缩放和操作类型转换
    amp_autocast = suppress  # 默认不进行任何操作
    loss_scaler = None
    # 如果使用 APEX AMP
    if use_amp == 'apex':                                                                            # APEX 只支持 CUDA 设备
        assert device.type == 'cuda'
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')                          # 使用 APEX 初始化模型和优化器
        loss_scaler = ApexScaler()                                                                   # 使用 APEX 提供的损失缩放器
        if utils.is_primary(args):
            _logger.info('使用 NVIDIA APEX AMP。以混合精度进行训练。')
    # 如果使用 PyTorch 原生 AMP
    elif use_amp == 'native':
        try:
            # 设置自动混合精度的 autocast，根据设备类型和精度类型选择
            amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)
        except (AttributeError, TypeError):
            # 如果 PyTorch 版本低于 1.10，则回退到仅支持 CUDA 的 autocast
            assert device.type == 'cuda'
            amp_autocast = torch.cuda.amp.autocast
        if device.type == 'cuda' and amp_dtype == torch.float16:
            # 如果使用 float16，则需要使用损失缩放器，而 bfloat16 不需要
            loss_scaler = NativeScaler()
        if utils.is_primary(args):
            _logger.info('使用原生 torch AMP 以混合精度进行训练。')
    # 如果没有启用 AMP
    else:
        if utils.is_primary(args):
            _logger.info('未启用 AMP。以 float32 进行训练。')

    # 可选择从检查点恢复训练
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            model,
            args.resume,
            optimizer=None if args.no_resume_opt else optimizer,                                     # 如果不恢复优化器状态，则设置为 None
            loss_scaler=None if args.no_resume_opt else loss_scaler,                                 # 如果不恢复损失缩放器状态，则设置为 None
            log_info=utils.is_primary(args),                                                         # 只在主节点输出日志
        )

    # 设置模型权重的指数移动平均（EMA），这里也可以使用 SWA（Stochastic Weight Averaging）
    model_ema = None
    if args.model_ema:
        # 重要：在 cuda()、DP（数据并行）包装、AMP 之后，但在 DDP（分布式数据并行）之前创建 EMA 模型
        model_ema = utils.ModelEmaV3(
            model,
            decay=args.model_ema_decay,                                                              # EMA 衰减率
            use_warmup=args.model_ema_warmup,                                                        # 是否使用 warmup
            device='cpu' if args.model_ema_force_cpu else None,                                      # 强制将 EMA 模型放在 CPU 上
        )
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)                            # 从检查点恢复 EMA 模型
        if args.torchcompile:
            model_ema = torch.compile(model_ema, backend=args.torchcompile)                         # 编译 EMA 模型

    # 设置分布式训练
    if args.distributed:
        if has_apex and use_amp == 'apex':
            # 首选 APEX DDP，除非启用了原生 AMP
            if utils.is_primary(args):
                _logger.info("使用 NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)                                            # 使用 Apex DDP 并延迟 all-reduce 操作
        else:
            if utils.is_primary(args):
                _logger.info("使用原生的 Torch DistributedDataParallel.")
            model = NativeDDP(model, device_ids=[device], broadcast_buffers=not args.no_ddp_bb)
        # 注意：EMA 模型不需要使用 DDP 包装



    # 如果启用了 torchcompile，对模型进行编译优化
    if args.torchcompile:
        # torch.compile() 必须在 DDP 包装后进行
        assert has_compile, '需要支持 torch.compile() 的版本，可能是 nightly 版本。'
        model = torch.compile(model, backend=args.torchcompile, mode=args.torchcompile_mode)


    if args.data and not args.data_dir:
        args.data_dir = args.data               # 设置数据目录
    if args.input_img_mode is None:
        input_img_mode = 'RGB' if data_config['input_size'][0] == 3 else 'L'                                # 根据输入通道数选择 RGB 或灰度
    else:
        input_img_mode = args.input_img_mode                                                                # 使用指定的输入图像模式


    # 使用CIFAR-100的数据集加载器替换ImageNet
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))  # CIFAR-100的均值和标准差
    ])

    # CIFAR-100训练集
    dataset_train = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    # CIFAR-100验证集
    dataset_eval = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    val_loader = DataLoader(dataset_eval, batch_size=args.validation_batch_size or args.batch_size, shuffle=False,
                            num_workers=args.workers)

    # 设置 Mixup / CutMix 数据增强
    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup,                                          # Mixup 的 alpha 参数
            cutmix_alpha=args.cutmix,                                       # CutMix 的 alpha 参数
            cutmix_minmax=args.cutmix_minmax,                                # CutMix 的最小最大裁剪范围
            prob=args.mixup_prob,                                             # Mixup / CutMix 的应用概率
            switch_prob=args.mixup_switch_prob,                                     # Mixup 和 CutMix 之间的切换概率
            mode=args.mixup_mode,                                           # Mixup 和 CutMix 之间的切换概率
            label_smoothing=args.smoothing,                                 # 标签平滑参数
            num_classes=args.num_classes                                    # 分类类别数
        )
        if args.prefetcher:
            assert not num_aug_splits                                   # collate 冲突（需要在 collate mixup 中支持解交错）
            collate_fn = FastCollateMixup(**mixup_args)                   # 使用快速的 collate mixup 进行数据增强
        else:
            mixup_fn = Mixup(**mixup_args)                              # 使用 Mixup 进行数据增强

    # 包装数据集以支持 AugMix 增强
    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    # create data loaders w/ augmentation pipeline
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']
    loader_train = create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        train_crop_mode=args.train_crop_mode,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        color_jitter_prob=args.color_jitter_prob,
        grayscale_prob=args.grayscale_prob,
        gaussian_blur_prob=args.gaussian_blur_prob,
        auto_augment=args.aa,
        num_aug_repeats=args.aug_repeats,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        device=device,
        use_prefetcher=args.prefetcher,
        use_multi_epochs_loader=args.use_multi_epochs_loader,
        worker_seeding=args.worker_seeding,
    )

    loader_eval = None
    if args.val_split:
        eval_workers = args.workers
        if args.distributed and ('tfds' in args.dataset or 'wds' in args.dataset):
            # FIXME reduces validation padding issues when using TFDS, WDS w/ workers and distributed training
            eval_workers = min(2, args.workers)
        loader_eval = create_loader(
            dataset_eval,
            input_size=data_config['input_size'],
            batch_size=args.validation_batch_size or args.batch_size,
            is_training=False,
            interpolation=data_config['interpolation'],
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=eval_workers,
            distributed=args.distributed,
            crop_pct=data_config['crop_pct'],
            pin_memory=args.pin_mem,
            device=device,
            use_prefetcher=args.prefetcher,
        )

# 使用增强管道创建数据加载器
#     train_interpolation = args.train_interpolation
#     if args.no_aug or not train_interpolation:
#         train_interpolation = data_config['interpolation']  # 如果未指定插值方法或禁用了增强，则使用默认插值方法
#
#     # 创建训练数据加载器
#     loader_train = create_loader(
#         dataset_train,
#         input_size=data_config['input_size'],  # 输入图像尺寸
#         batch_size=args.batch_size,  # 批次大小
#         is_training=True,  # 标记为训练模式
#         no_aug=args.no_aug,  # 是否禁用增强
#         re_prob=args.reprob,  # 随机擦除概率
#         re_mode=args.remode,  # 随机擦除模式
#         re_count=args.recount,  # 随机擦除次数
#         re_split=args.resplit,  # 是否进行擦除分割
#         train_crop_mode=args.train_crop_mode,  # 训练裁剪模式
#         scale=args.scale,  # 随机裁剪比例
#         ratio=args.ratio,  # 宽高比
#         hflip=args.hflip,  # 水平翻转概率
#         vflip=args.vflip,  # 垂直翻转概率
#         color_jitter=args.color_jitter,  # 颜色抖动幅度
#         aug_repeats=args.aug_repeats,  # 增强重复次数
#         auto_augment=args.aa,  # 是否使用自动增强
#         num_aug_repeats=args.aug_repeats,  # 增强重复次数
#         num_aug_splits=num_aug_splits,  # 数据增强分割次数
#         interpolation=train_interpolation,  # 插值方法
#         mean=data_config['mean'],  # 标准化均值
#         std=data_config['std'],  # 标准化标准差
#         collate_fn=collate_fn,  # 自定义 collate 函数
#         distributed=args.distributed,  # 是否使用分布式训练
#         pin_memory=args.pin_mem,  # 是否使用固定内存
#         device=device,  # 设备类型
#         use_prefetcher=args.prefetcher,  # 是否使用预取器
#         use_multi_epochs_loader=args.use_multi_epochs_loader,  # 是否使用多轮次加载器
#         worker_seeding=args.worker_seeding,  # 设置 worker 的随机种子
#     )
#
#     # 如果有验证集划分，则创建验证数据加载器
#     loader_eval = None
#     if args.val_split:
#         eval_workers = args.workers
#         if args.distributed and ('tfds' in args.dataset or 'wds' in args.dataset):
#             # FIXME 当使用 TFDS 或 WDS 并进行分布式训练时，减少验证集的填充问题
#             eval_workers = min(2, args.workers)
#
#         # 创建验证数据加载器
#         loader_eval = create_loader(
#             dataset_eval,
#             input_size=data_config['input_size'],
#             batch_size=args.validation_batch_size or args.batch_size,
#             is_training=False,  # 标记为验证模式
#             interpolation=data_config['interpolation'],
#             mean=data_config['mean'],
#             std=data_config['std'],
#             num_workers=eval_workers,
#             distributed=args.distributed,
#             crop_pct=data_config['crop_pct'],  # 验证集裁剪比例
#             pin_memory=args.pin_mem,
#             device=device,
#             use_prefetcher=args.prefetcher,
#         )

    # 设置损失函数
    if args.jsd_loss:
        assert num_aug_splits > 1                                            # JSD 仅在设置了多个增强分割时有效
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing)
    elif mixup_active:
        # 在 mixup 增强的目标转换中已处理标签平滑，输出稀疏且柔和的目标
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(
                target_threshold=args.bce_target_thresh,
                sum_classes=args.bce_sum,
                pos_weight=args.bce_pos_weight,
            )                                                                   # 使用二元交叉熵作为损失函数
        else:
            train_loss_fn = SoftTargetCrossEntropy()                             # 使用软目标交叉熵
    elif args.smoothing:
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(
                smoothing=args.smoothing,
                target_threshold=args.bce_target_thresh,
                sum_classes=args.bce_sum,
                pos_weight=args.bce_pos_weight,
            )                                                                   # 使用二元交叉熵并进行标签平滑处理
        else:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)    # 使用标签平滑交叉熵
    else:
        train_loss_fn = nn.CrossEntropyLoss()                                   # 使用标准交叉熵损失
    train_loss_fn = train_loss_fn.to(device=device)                             # 将训练损失函数迁移到指定设备上
    # 设置验证损失函数为标准交叉熵
    validate_loss_fn = nn.CrossEntropyLoss().to(device=device)

    # 设置检查点保存器和评估指标跟踪
    eval_metric = args.eval_metric if loader_eval is not None else '损失'        # 评估指标，如果没有验证集则使用损失
    decreasing_metric = eval_metric == '损失'       # 指示指标是否递减
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None

    # 仅在主节点执行以下操作
    if utils.is_primary(args):
        if args.experiment:
            exp_name = args.experiment      # 如果指定实验名称则使用用户提供的名称
        else:
            # 否则生成一个默认实验名称，包含日期、时间、模型名称和输入尺寸
            exp_name = '-'.join([
                datetime.now().strftime("%Y%m%d-%H%M%S"),
                safe_model_name(args.model),
                str(data_config['input_size'][-1])
            ])
            # 设置输出目录路径
        output_dir = utils.get_outdir(args.output if args.output else './output/train', exp_name)

        # 创建检查点保存器
        saver = utils.CheckpointSaver(
            model=model,
            optimizer=optimizer,
            args=args,
            model_ema=model_ema,
            amp_scaler=loss_scaler,
            checkpoint_dir=output_dir,
            recovery_dir=output_dir,
            decreasing=decreasing_metric,
            max_history=args.checkpoint_hist
        )
        # 将训练参数保存到 args.yaml 文件中
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    # 仅在主进程中初始化 TensorBoard
    if utils.is_primary(args):
        writer = SummaryWriter(log_dir=output_dir)
        print(f"日志将打印在{output_dir}目录下")

    # if utils.is_primary(args) and args.log_wandb:
    #     if has_wandb:
    #         wandb.init(project=args.experiment, config=args)
    #     else:
    #         _logger.warning(
    #             "You've requested to log metrics to wandb but package not found. "
    #             "Metrics not being logged to wandb, try `pip install wandb`")


    # setup learning rate schedule and starting epoch
    # 设置学习率调度器和起始轮数
    # 计算每轮的更新次数
    updates_per_epoch = (len(loader_train) + args.grad_accum_steps - 1) // args.grad_accum_steps
    # 创建学习率调度器
    lr_scheduler, num_epochs = create_scheduler_v2(
        optimizer,
        **scheduler_kwargs(args, decreasing_metric=decreasing_metric),
        updates_per_epoch=updates_per_epoch,
    )
    # 初始化起始轮数为 0
    start_epoch = 0

    if args.initial_checkpoint:
        print("已指定检查点模型路径")
        if os.path.isfile(args.initial_checkpoint):
            print(f"将恢复至 {args.initial_checkpoint}")
            checkpoint = torch.load(args.initial_checkpoint, map_location='cpu')

            # 恢复模型权重
            model.load_state_dict(checkpoint['state_dict'])
            print("模型权重已恢复")

            # 初始化新的优化器
            if hasattr(args, 'optimizer_type') and args.optimizer_type == 'adam':
                optimizer = optim.Adam(model.parameters(), lr=0.001)
            elif hasattr(args, 'optimizer_type') and args.optimizer_type == 'sgd':
                optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            else:
                if hasattr(args, 'optimizer_type'):
                    raise ValueError(f"未定义的优化器类型: {args.optimizer_type}")
                else:
                    raise ValueError("未指定优化器类型，请在参数中定义 'optimizer_type'")

            # 尝试加载优化器状态
            if 'optimizer' in checkpoint:
                checkpoint_optimizer_type = checkpoint.get('optimizer_type', 'sgd')  # 假设旧检查点未保存类型，默认为 sgd
                if checkpoint_optimizer_type == args.optimizer_type:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    print("已恢复优化器状态")
                else:
                    print(
                        f"优化器类型不同，跳过恢复优化器状态（检查点: {checkpoint_optimizer_type}, 当前: {args.optimizer_type}）")
            else:
                print("未找到优化器状态")

            # # 加载学习率调度器状态
            # if 'lr_scheduler' in checkpoint and lr_scheduler is not None:
            #     if checkpoint_optimizer_type == args.optimizer_type:
            #         lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            #         print("已恢复学习率调度器")
            #     else:
            #         print("优化器类型不同，跳过恢复学习率调度器")
            # else:
            #     print("未找到学习率调度器状态")

            # # 加载损失缩放器状态
            # if 'loss_scaler' in checkpoint and loss_scaler is not None:
            #     if checkpoint_optimizer_type == args.optimizer_type:
            #         loss_scaler.load_state_dict(checkpoint['loss_scaler'])
            #         print("已恢复损失缩放器")
            #     else:
            #         print("优化器类型不同，跳过恢复损失缩放器")
            # else:
            #     print("未找到损失缩放器状态")

            # 恢复 epoch 计数器
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch']
                print(f"恢复到 epoch {start_epoch} 开始训练")
            else:
                start_epoch = 0
                print("检查点中没有找到 epoch 信息，从第 0 epoch 开始训练")

            print(f"已加载至：'{args.initial_checkpoint}'")
        else:
            print(f"已指定的检查点路径 {args.initial_checkpoint} 未找到相应文件")
            exit(1)  # 文件找不到则退出
    else:
        print("未指定检查点模型路径")


    if args.start_epoch is not None:
        # 如果显式指定了 start_epoch，则优先使用该轮数
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        # 如果没有指定 start_epoch，但存在恢复轮数，则使用恢复的轮数
        start_epoch = resume_epoch
    # 如果学习率调度器存在，并且起始轮数大于 0，则更新学习率
    if lr_scheduler is not None and start_epoch > 0:
        if args.sched_on_updates:
            # 基于更新次数调整学习率
            lr_scheduler.step_update(start_epoch * updates_per_epoch)
        else:
            # 基于轮数调整学习率
            lr_scheduler.step(start_epoch)

    # 仅在主节点记录学习率调度信息
    # if utils.is_primary(args):
    #     _logger.info(
    #         f'计划的训练轮数:  {num_epochs}。学习率在每个{"轮次" if lr_scheduler.t_in_epochs else "更新"}时调整。')

    results = []


    try:
        for epoch in range(start_epoch, num_epochs):
            # 如果数据集具有 set_epoch 方法，则设置当前轮次
            if hasattr(dataset_train, 'set_epoch'):
                dataset_train.set_epoch(epoch)
            elif args.distributed and hasattr(loader_train.sampler, 'set_epoch'):
                # 如果是分布式训练且数据采样器具有 set_epoch 方法，则设置采样器的当前轮次
                loader_train.sampler.set_epoch(epoch)
            # 训练一个轮次
            train_metrics = train_one_epoch(
                epoch,
                model,
                loader_train,
                optimizer,
                train_loss_fn,
                args,
                lr_scheduler=lr_scheduler,
                saver=saver,
                output_dir=output_dir,
                amp_autocast=amp_autocast,
                loss_scaler=loss_scaler,
                model_ema=model_ema,
                mixup_fn=mixup_fn,
                num_updates_total=num_epochs * updates_per_epoch,
            )

            # 记录训练损失和学习率到 TensorBoard
            writer.add_scalar('训练损失', train_metrics['损失'], epoch)
            for i, param_group in enumerate(optimizer.param_groups):
                writer.add_scalar(f'组学习率{i}', param_group['lr'], epoch)

            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if utils.is_primary(args):
                    _logger.info("Distributing BatchNorm running means and vars")
                utils.distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if utils.is_primary(args):
                    _logger.info("Distributing BatchNorm running means and vars")
                utils.distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

            if loader_eval is not None:
                eval_metrics = validate(
                    model,
                    loader_eval,
                    validate_loss_fn,
                    args,
                    device=device,
                    amp_autocast=amp_autocast,
                )

                # 记录验证损失和准确率到 TensorBoard
                writer.add_scalar('损失/验证', eval_metrics['损失'], epoch)
                writer.add_scalar('准确率/验证_@1', eval_metrics['准确率@1'], epoch)
                writer.add_scalar('准确率/验证_@5', eval_metrics['准确率@5'], epoch)

                if model_ema is not None and not args.model_ema_force_cpu:
                    if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                        utils.distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')

                    ema_eval_metrics = validate(
                        model_ema,
                        loader_eval,
                        validate_loss_fn,
                        args,
                        device=device,
                        amp_autocast=amp_autocast,
                        log_suffix=' (EMA)',
                    )
                    eval_metrics = ema_eval_metrics
            else:
                eval_metrics = None

            if output_dir is not None:
                lrs = [param_group['lr'] for param_group in optimizer.param_groups]
                utils.update_summary(
                    epoch,
                    train_metrics,
                    eval_metrics,
                    filename=os.path.join(output_dir, 'summary.csv'),
                    lr=sum(lrs) / len(lrs),
                    write_header=best_metric is None,
                    log_wandb=args.log_wandb and has_wandb,
                )

            if eval_metrics is not None:
                latest_metric = eval_metrics[eval_metric]
            else:
                latest_metric = train_metrics[eval_metric]

            if saver is not None:
                # save proper checkpoint with eval metric
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=latest_metric)

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, latest_metric)

            results.append({
                '轮次': epoch,
                '训练': train_metrics,
                '验证': eval_metrics,
            })

    except KeyboardInterrupt:
        pass

    # 训练结束后关闭 TensorBoard SummaryWriter
    writer.close()


    results = {'全部结果如下': results}
    if best_metric is not None:
        results['最佳'] = results['全部结果如下'][best_epoch - start_epoch]
        _logger.info('*** 最佳模型性能: {0} (训练出最佳结果的轮次 {1})'.format(best_metric, best_epoch))
    print(f'--训练结果：\n{json.dumps(results, indent=4, ensure_ascii=False)}')


def train_one_epoch(
        epoch,
        model,
        loader,
        optimizer,
        loss_fn,
        args,
        device=torch.device('cuda'),
        lr_scheduler=None,
        saver=None,
        output_dir=None,
        amp_autocast=suppress,
        loss_scaler=None,
        model_ema=None,
        mixup_fn=None,
        num_updates_total=None,
):
    # 如果指定了关闭 Mixup 的轮数，并且当前轮次达到或超过该轮数
    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            # 如果使用了数据预取器并且加载器启用了 Mixup，则关闭 Mixup
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            # 如果使用了 mixup_fn 数据增强函数，则将其禁用
            mixup_fn.mixup_enabled = False

    # 检查优化器是否是二阶优化器
    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    # 检查模型是否支持 no_sync 方法（用于分布式训练）
    has_no_sync = hasattr(model, "no_sync")
    # 初始化计时器和损失记录工具
    update_time_m = utils.AverageMeter()        # 用于记录每次模型更新的时间
    data_time_m = utils.AverageMeter()          # 用于记录数据加载时间
    losses_m = utils.AverageMeter()             # 用于记录损失值

    # 设置模型为训练模式
    model.train()

    # 梯度累积设置
    accum_steps = args.grad_accum_steps                                     # 每次累积的步数
    last_accum_steps = len(loader) % accum_steps                            # 最后一个累积周期中剩余的步数
    updates_per_epoch = (len(loader) + accum_steps - 1) // accum_steps      # 每个轮次中的更新次数
    num_updates = epoch * updates_per_epoch                                 # 到当前轮次为止的总更新次数
    last_batch_idx = len(loader) - 1                                        # 最后一个批次的索引
    last_batch_idx_to_accum = len(loader) - last_accum_steps                # 最后一次需要累积的批次索引

    # 初始化计时器和优化器
    data_start_time = update_start_time = time.time()                       # 记录数据加载和更新的开始时间
    optimizer.zero_grad()                                                   # 清零梯度
    update_sample_count = 0                                                 # 初始化样本计数器

    # 遍历每个批次
    for batch_idx, (input, target) in enumerate(loader):
        # 判断当前批次是否为最后一个批次
        last_batch = batch_idx == last_batch_idx
        # 判断是否需要进行参数更新（最后一个批次或达到了累积步数）
        need_update = last_batch or (batch_idx + 1) % accum_steps == 0
        # 当前累积更新的索引
        update_idx = batch_idx // accum_steps
        # 如果到达了最后一个需要累积的批次，设置累积步数为剩余的批次数
        if batch_idx >= last_batch_idx_to_accum:
            accum_steps = last_accum_steps

        # 如果未使用数据预取器，则手动将数据移动到设备上
        if not args.prefetcher:
            input, target = input.to(device), target.to(device)
            # 如果定义了 Mixup 函数，则进行 Mixup 数据增强
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)
        # 如果设置了 channels_last，则将输入的数据格式调整为 channels_last
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        # multiply by accum steps to get equivalent for full update
        # 乘以 accum_steps 以获得等效于完整更新的时间
        data_time_m.update(accum_steps * (time.time() - data_start_time))

        def _forward():
            with amp_autocast():
                output = model(input)
                loss = loss_fn(output, target)
            if accum_steps > 1:
                loss /= accum_steps
            return loss

        def _backward(_loss):
            if loss_scaler is not None:
                loss_scaler(
                    _loss,
                    optimizer,
                    clip_grad=args.clip_grad,
                    clip_mode=args.clip_mode,
                    parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
                    create_graph=second_order,
                    need_update=need_update,
                )
            else:
                _loss.backward(create_graph=second_order)
                if need_update:
                    if args.clip_grad is not None:
                        utils.dispatch_clip_grad(
                            model_parameters(model, exclude_head='agc' in args.clip_mode),
                            value=args.clip_grad,
                            mode=args.clip_mode,
                        )
                    optimizer.step()

        if has_no_sync and not need_update:
            with model.no_sync():
                loss = _forward()
                _backward(loss)
        else:
            loss = _forward()
            _backward(loss)

        if not args.distributed:
            losses_m.update(loss.item() * accum_steps, input.size(0))
        update_sample_count += input.size(0)

        if not need_update:
            data_start_time = time.time()
            continue

        num_updates += 1
        optimizer.zero_grad()
        if model_ema is not None:
            model_ema.update(model, step=num_updates)

        if args.synchronize_step and device.type == 'cuda':
            torch.cuda.synchronize()
        time_now = time.time()
        update_time_m.update(time.time() - update_start_time)
        update_start_time = time_now

        if update_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item() * accum_steps, input.size(0))
                update_sample_count *= args.world_size

            if utils.is_primary(args):
                _logger.info(
                    f'训练: {epoch} [{update_idx:>4d}/{updates_per_epoch}]  '
                    f'({100. * (update_idx + 1) / updates_per_epoch:>3.0f}%)  '
                    f'损失值: {losses_m.val:#.3g} ({losses_m.avg:#.3g})  '
                    f'时间: {update_time_m.val:.3f}秒, {update_sample_count / update_time_m.val:>7.2f}/秒  '
                    f'({update_time_m.avg:.3f}秒, {update_sample_count / update_time_m.avg:>7.2f}/秒)  '
                    f'学习率: {lr:.3e}  '
                    f'数据加载时间: {data_time_m.val:.3f}秒 ({data_time_m.avg:.3f}秒)'
                )

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                        padding=0,
                        normalize=True
                    )

        if saver is not None and args.recovery_interval and (
                (update_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=update_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)
            # print('学习率已更新')

        update_sample_count = 0
        data_start_time = time.time()
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('损失', losses_m.avg)])


def validate(
        model,
        loader,
        loss_fn,
        args,
        device=torch.device('cuda'),
        amp_autocast=suppress,
        log_suffix=''
):
    batch_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.to(device)
                target = target.to(device)
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output = model(input)
                if isinstance(output, (tuple, list)):
                    output = output[0]

                # augmentation reduction
                reduce_factor = args.tta
                if reduce_factor > 1:
                    output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                    target = target[0:target.size(0):reduce_factor]

                loss = loss_fn(output, target)
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                acc1 = utils.reduce_tensor(acc1, args.world_size)
                acc5 = utils.reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            if device.type == 'cuda':
                torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if utils.is_primary(args) and (last_batch or batch_idx % args.log_interval == 0):
                log_name = '测试' + log_suffix
                _logger.info(
                    f'{log_name}: [{batch_idx:>4d}/{last_idx}]  '
                    f'时间: {batch_time_m.val:.3f} ({batch_time_m.avg:.3f})  '
                    f'损失: {losses_m.val:>7.3f} ({losses_m.avg:>6.3f})  '
                    f'准确率@1: {top1_m.val:>7.3f} ({top1_m.avg:>7.3f})  '
                    f'准确率@5: {top5_m.val:>7.3f} ({top5_m.avg:>7.3f})'
                )

            metrics = OrderedDict([('损失', losses_m.avg), ('准确率@1', top1_m.avg), ('准确率@5', top5_m.avg)])

    return metrics


if __name__ == '__main__':
    args, args_text = _parse_args()
    main()
