import os
import time
import torch
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

from src.datasets.cifar10 import get_loader
from src.models.vit import VisionTransformer_pytorch, VisionTransformer

class Solver:
    def __init__(self, args):
        self.args = args
        self.train_loader, self.test_loader, self.train_dataset_size, self.test_dataset_size = get_loader(args)
        if self.args.use_torch_transformer_layers:
            self.model = VisionTransformer_pytorch(
                n_channels=self.args.n_channels,
                embed_dim=self.args.embed_dim,
                n_layers=self.args.n_layers,
                n_attention_heads=self.args.n_attention_heads,
                forward_mul=self.args.forward_mul,
                image_size=self.args.img_size,
                patch_size=self.args.patch_size,
                n_classes=self.args.n_classes,
                dropout=self.args.dropout,
            )
        else:
            self.model = VisionTransformer(
                n_channels=self.args.n_channels,
                embed_dim=self.args.embed_dim,
                n_layers=self.args.n_layers,
                n_attention_heads=self.args.n_attention_heads,
                forward_mul=self.args.forward_mul,
                image_size=self.args.img_size,
                patch_size=self.args.patch_size,
                n_classes=self.args.n_classes,
                dropout=self.args.dropout,
            )
        if self.args.is_cuda:
            self.model = self.model.cuda()
            print("----------模型结构----------")
            print(self.model)
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"可训练参数数量：{n_parameters}")
        # 加载预训练模型（仅在 resume_path 为空时加载）
        if self.args.load_pretrained and self.args.resume_path is None:
            pretrained_path = os.path.join(self.args.model_path, 'ViT_model.pth')
            if os.path.exists(pretrained_path):
                print("加载预训练模型")
                self.model.load_state_dict(torch.load(pretrained_path))
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_loss = []
        self.test_loss = []
        self.train_acc = []
        self.test_acc = []
        self.test_precision = []
        self.test_recall = []
        self.test_f1 = []
        # 保存每个 epoch 的 checkpoint 信息
        self.checkpoints = []

    def test_dataset(self, loader):
        self.model.eval()
        all_labels = []
        all_logits = []
        with torch.no_grad():
            for x, y in tqdm(loader, desc="测试中", leave=False):
                if self.args.is_cuda:
                    x = x.cuda()
                logits = self.model(x)
                all_labels.append(y)
                all_logits.append(logits.cpu())
        all_labels = torch.cat(all_labels)
        all_logits = torch.cat(all_logits)
        all_pred = all_logits.max(1)[1]
        loss = self.loss_fn(all_logits, all_labels).item()
        acc = accuracy_score(all_labels, all_pred)
        prec = precision_score(all_labels, all_pred, average='macro', zero_division=0)
        rec = recall_score(all_labels, all_pred, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_pred, average='macro', zero_division=0)
        cm = confusion_matrix(all_labels, all_pred, labels=range(self.args.n_classes))
        return loss, acc, prec, rec, f1, cm

    def test(self, is_train=True):
        if is_train:
            loss, acc, prec, rec, f1, cm = self.test_dataset(self.train_loader)
            phase = "训练集"
        else:
            loss, acc, prec, rec, f1, cm = self.test_dataset(self.test_loader)
            phase = "测试集"
        print(f"{phase} - Loss: {loss:.4f}  Accuracy: {acc:.2%}  Precision: {prec:.2%}  Recall: {rec:.2%}  F1: {f1:.2%}")
        return loss, acc, prec, rec, f1

    def train(self, start_epoch=0):
        iters_per_epoch = len(self.train_loader)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=1e-3)
        warmup_scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda epoch: float(epoch + 1) / self.args.warmup_epochs if epoch < self.args.warmup_epochs else 1.0
        )
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.args.epochs - self.args.warmup_epochs,
            eta_min=1e-5
        )
        best_acc = 0.0
        early_stop_counter = 0
        for epoch in range(start_epoch, self.args.epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_acc = 0.0
            pbar = tqdm(self.train_loader, desc=f"Epoch [{epoch+1}/{self.args.epochs}]", leave=False)
            for x, y in pbar:
                if self.args.is_cuda:
                    x, y = x.cuda(), y.cuda()
                optimizer.zero_grad()
                logits = self.model(x)
                loss = self.loss_fn(logits, y)
                loss.backward()
                optimizer.step()
                pred = logits.max(1)[1]
                batch_acc = (pred == y).float().mean().item()
                epoch_loss += loss.item()
                epoch_acc += batch_acc
                pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_acc:.2%}")
            epoch_loss /= iters_per_epoch
            epoch_acc /= iters_per_epoch
            self.train_loss.append(epoch_loss)
            self.train_acc.append(epoch_acc)
            test_loss, test_acc, test_prec, test_rec, test_f1 = self.test(is_train=False)
            self.test_loss.append(test_loss)
            self.test_acc.append(test_acc)
            self.test_precision.append(test_prec)
            self.test_recall.append(test_rec)
            self.test_f1.append(test_f1)
            print(f"Epoch [{epoch+1}/{self.args.epochs}] Train Loss: {epoch_loss:.4f} Train Acc: {epoch_acc:.2%} | Test Loss: {test_loss:.4f} Test Acc: {test_acc:.2%}")
            # 保存当前 epoch 的 checkpoint 到文件 epoch_{epoch+1}.pth
            checkpoint_path = os.path.join(self.args.model_path, f'epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'train_loss': self.train_loss,
                'test_loss': self.test_loss,
                'train_acc': self.train_acc,
                'test_acc': self.test_acc,
                'test_precision': self.test_precision,
                'test_recall': self.test_recall,
                'test_f1': self.test_f1
            }, checkpoint_path)
            self.checkpoints.append({'epoch': epoch, 'test_acc': test_acc, 'checkpoint_path': checkpoint_path})
            # 更新学习率调度器
            if epoch < self.args.warmup_epochs:
                warmup_scheduler.step()
            else:
                cosine_scheduler.step()
            # 早停逻辑：若连续 early_stop_patience 个 epoch 无提升，则提前停止训练
            if test_acc > best_acc:
                best_acc = test_acc
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                print(f"无提升次数: {early_stop_counter}/{self.args.early_stop_patience}")
            if early_stop_counter >= self.args.early_stop_patience:
                print(f"达到耐心值 {self.args.early_stop_patience}，提前停止训练。")
                break

    def select_best_model(self):
        # 遍历所有保存的 checkpoint，选择测试准确率最高的
        if not self.checkpoints:
            print("没有保存的checkpoint，无法选择最佳模型")
            return
        best_checkpoint = max(self.checkpoints, key=lambda x: x['test_acc'])
        best_path = best_checkpoint['checkpoint_path']
        best_save_path = os.path.join(self.args.model_path, 'best.pth')
        torch.save(torch.load(best_path), best_save_path)
        print(f"最佳模型 (Epoch {best_checkpoint['epoch']+1}) 已保存至 {best_save_path}")

    def plot_graphs(self):
        plt.figure()
        plt.plot(self.train_loss, label='Train Loss', color='blue')
        plt.plot(self.test_loss, label='Test Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        loss_curve_path = os.path.join(self.args.output_path, 'loss_curve.png')
        plt.savefig(loss_curve_path, bbox_inches='tight')
        plt.close()
        print(f"Loss 曲线已保存至 {loss_curve_path}")

        plt.figure()
        plt.plot(self.train_acc, label='Train Acc', color='blue')
        plt.plot(self.test_acc, label='Test Acc', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        acc_curve_path = os.path.join(self.args.output_path, 'acc_curve.png')
        plt.savefig(acc_curve_path, bbox_inches='tight')
        plt.close()
        print(f"Accuracy 曲线已保存至 {acc_curve_path}")

        plt.figure()
        plt.plot(self.test_precision, label='Test Precision', color='green')
        plt.plot(self.test_recall, label='Test Recall', color='orange')
        plt.plot(self.test_f1, label='Test F1', color='purple')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        metrics_curve_path = os.path.join(self.args.output_path, 'metrics_curve.png')
        plt.savefig(metrics_curve_path, bbox_inches='tight')
        plt.close()
        print(f"评价指标曲线已保存至 {metrics_curve_path}")
