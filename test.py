import os
import argparse
import time
import datetime
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 导入数据加载函数和模型定义
from src.datasets.cifar10 import get_loader
from src.models.vit import VisionTransformer, VisionTransformer_pytorch


def test_model(model, test_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Testing", leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    avg_loss = total_loss / len(test_loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    return avg_loss, acc, prec, rec, f1, cm


def main(args):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载数据
    # get_loader 返回 (train_loader, test_loader, len(train_dataset), len(test_dataset))
    _, test_loader, _, test_dataset_size = get_loader(args)
    num_batches = len(test_loader)
    print(f"测试集样本数: {test_dataset_size}")
    print(f"测试集 batch 数: {num_batches}")
    print(f"batch_size * batches = {args.batch_size * num_batches}")

    # 根据参数选择模型结构（必须与训练时配置一致）
    if args.use_torch_transformer_layers:
        model = VisionTransformer_pytorch(
            n_channels=args.n_channels,
            embed_dim=args.embed_dim,
            n_layers=args.n_layers,
            n_attention_heads=args.n_attention_heads,
            forward_mul=args.forward_mul,
            image_size=args.img_size,
            patch_size=args.patch_size,
            n_classes=args.n_classes,
            dropout=args.dropout,
        )
    else:
        model = VisionTransformer(
            n_channels=args.n_channels,
            embed_dim=args.embed_dim,
            n_layers=args.n_layers,
            n_attention_heads=args.n_attention_heads,
            forward_mul=args.forward_mul,
            image_size=args.img_size,
            patch_size=args.patch_size,
            n_classes=args.n_classes,
            dropout=args.dropout,
        )
    model = model.to(device)

    # 加载 best.pth 模型权重
    checkpoint_path = args.checkpoint_path
    if os.path.exists(checkpoint_path):
        print(f"加载最佳模型权重: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            print("Missing keys:", missing_keys)
            print("Unexpected keys:", unexpected_keys)
        else:
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
            print("Missing keys:", missing_keys)
            print("Unexpected keys:", unexpected_keys)
    else:
        print(f"Checkpoint 文件不存在: {checkpoint_path}")
        return

    # 测试模型
    start_time = time.time()
    avg_loss, acc, prec, rec, f1, cm = test_model(model, test_loader, device)
    end_time = time.time()
    duration = end_time - start_time

    # 输出测试结果
    print("\n测试结果:")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Accuracy: {acc:.2%}")
    print(f"Precision: {prec:.2%}")
    print(f"Recall: {rec:.2%}")
    print(f"F1 Score: {f1:.2%}")
    print("Confusion Matrix:")
    print(cm)
    print(f"测试耗时: {datetime.timedelta(seconds=int(duration))}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test the best model on CIFAR-10 test set")
    # 数据集及图像参数
    parser.add_argument('--dataset', type=str, default='cifar10', help='使用的数据集')
    parser.add_argument('--patch_size', type=int, default=4, help='patch 大小')
    parser.add_argument('--img_size', type=int, default=32, help='图像尺寸')
    parser.add_argument('--n_channels', type=int, default=3, help='输入图像通道数')
    parser.add_argument('--data_path', type=str,
                        default='/data1/nliu/2025_homework/Pro_002_ViT_CIFAR10/data/cifar-10-batches-py',
                        help='数据集根目录')
    # ViT 参数（请与训练时保持一致）
    parser.add_argument('--use_torch_transformer_layers', type=bool, default=False,
                        help='是否使用 torch 内置 Transformer 层')
    parser.add_argument('--embed_dim', type=int, default=256, help='embedding 维度')
    parser.add_argument('--n_attention_heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--forward_mul', type=int, default=4, help='前向全连接层的维度倍数')
    parser.add_argument('--n_layers', type=int, default=8, help='Transformer Encoder 层数')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout 概率')
    parser.add_argument('--n_classes', type=int, default=10, help='类别数')
    parser.add_argument('--batch_size', type=int, default=256, help='batch 大小')
    parser.add_argument('--workers', type=int, default=8, help='数据加载时的 worker 数量')
    parser.add_argument('--lr', type=float, default=5e-4, help='学习率')
    # GPU 选择
    parser.add_argument('--gpu_id', type=int, default=0, help='指定使用的GPU编号')
    # 模型权重路径（best.pth）
    parser.add_argument('--checkpoint_path', type=str,
                        default='/data1/nliu/2025_homework/Pro_002_ViT_CIFAR10/weight_pt/cifar10/best.pth',
                        help='最佳模型权重路径')
    args = parser.parse_args()
    main(args)
