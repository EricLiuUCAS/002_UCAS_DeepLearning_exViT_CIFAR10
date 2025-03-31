import os
import argparse
import datetime
import time
import torch

from src.utils.utils import Solver


def update_args(args):
    # 统计可用GPU数量并选择指定的GPU
    num_gpus = torch.cuda.device_count()
    print(f"可用GPU数量: {num_gpus}")
    if num_gpus > 0:
        print(f"使用GPU编号: {args.gpu_id}")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    else:
        print("没有可用GPU，使用CPU训练")

    # 设置模型保存和输出目录
    args.model_path = os.path.join(args.model_path, args.dataset)
    args.output_path = os.path.join(args.output_path, args.dataset)
    os.makedirs(args.model_path, exist_ok=True)
    os.makedirs(args.output_path, exist_ok=True)
    args.n_patches = (args.img_size // args.patch_size) ** 2
    args.is_cuda = torch.cuda.is_available()
    return args


def main(args):
    solver = Solver(args)

    # 如果指定了 resume_path，则加载断点并接着训练
    start_epoch = 0
    if args.resume_path is not None and os.path.exists(args.resume_path):
        checkpoint = torch.load(args.resume_path, map_location='cpu')
        solver.model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        solver.train_loss = checkpoint.get('train_loss', [])
        solver.test_loss = checkpoint.get('test_loss', [])
        solver.train_acc = checkpoint.get('train_acc', [])
        solver.test_acc = checkpoint.get('test_acc', [])
        solver.test_precision = checkpoint.get('test_precision', [])
        solver.test_recall = checkpoint.get('test_recall', [])
        solver.test_f1 = checkpoint.get('test_f1', [])
        print(f"恢复训练，起始 epoch 为 {start_epoch}")

    # 打印数据集信息
    print(f"训练集样本数：{solver.train_dataset_size}，训练 batch 数量：{len(solver.train_loader)}，"
          f"batch_size * batches = {solver.args.batch_size * len(solver.train_loader)}")
    print(f"测试集样本数：{solver.test_dataset_size}，测试 batch 数量：{len(solver.test_loader)}，"
          f"batch_size * batches = {solver.args.batch_size * len(solver.test_loader)}")

    start_time = time.time()
    solver.train(start_epoch=start_epoch)
    solver.plot_graphs()
    solver.test(is_train=False)
    end_time = time.time()
    duration = end_time - start_time

    print("----------网络结构----------")
    print(solver.model)
    total_params = sum(p.numel() for p in solver.model.parameters() if p.requires_grad)
    print(f"模型参数总量：{total_params}")
    print(f"训练总用时：{datetime.timedelta(seconds=int(duration))}")

    # 在所有保存的checkpoint中选择最佳模型并另存为 best.pth
    solver.select_best_model()


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 ViT Training')
    # 数据参数
    parser.add_argument('--dataset', type=str, default='cifar10', help='使用的数据集')
    parser.add_argument('--patch_size', type=int, default=4, help='patch 大小')
    parser.add_argument('--img_size', type=int, default=32, help='图像尺寸')
    parser.add_argument('--n_channels', type=int, default=3, help='输入图像通道数')
    parser.add_argument('--data_path', type=str, default='/data1/nliu/2025_homework/Pro_002_ViT_CIFAR10/data/cifar-10-batches-py',
                        help='数据集根目录')
    # ViT 参数
    parser.add_argument('--use_torch_transformer_layers', type=bool, default=False,
                        help='是否使用 torch 内置 Transformer 层')
    parser.add_argument('--embed_dim', type=int, default=256, help='embedding 维度')
    parser.add_argument('--n_attention_heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--forward_mul', type=int, default=4, help='前向全连接层的维度倍数')
    parser.add_argument('--n_layers', type=int, default=8, help='Transformer Encoder 层数')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout 概率')
    # 模型保存与输出路径
    parser.add_argument('--model_path', type=str, default='./weight_pt', help='模型保存路径')
    parser.add_argument('--output_path', type=str, default='./output', help='输出结果保存路径')
    # 训练模式：resume_path 指定断点续训的检查点路径，load_pretrained 加载预训练模型
    parser.add_argument('--resume_path', type=str, default=None, help='断点续训的检查点路径')
    parser.add_argument('--load_pretrained', type=bool, default=False, help='是否加载预训练模型')
    # 训练参数
    parser.add_argument('--epochs', type=int, default=50, help='训练 epoch 数')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='warmup epoch 数')
    parser.add_argument('--batch_size', type=int, default=256, help='batch 大小')
    parser.add_argument('--n_classes', type=int, default=10, help='类别数')
    parser.add_argument('--workers', type=int, default=8, help='数据加载时的 worker 数量')
    parser.add_argument('--lr', type=float, default=5e-4, help='学习率')
    # GPU 选择
    parser.add_argument('--gpu_id', type=int, default=0, help='指定使用的GPU编号')
    # 早停设置：当连续若干epoch无提升时提前停止训练
    parser.add_argument('--early_stop_patience', type=int, default=5, help='提前停止的耐心值')
    return parser.parse_args()


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    print("训练开始时间：" + start_time.strftime('%Y-%m-%d %H:%M:%S'))
    args = parse_args()
    args = update_args(args)
    main(args)
    end_time = datetime.datetime.now()
    print("训练结束时间：" + end_time.strftime('%Y-%m-%d %H:%M:%S'))
    print("总训练时长：" + str(end_time - start_time))
