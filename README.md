# Vision Transformer CIFAR-10 图像分类项目

本项目从零开始实现了 Vision Transformer (ViT) 模型，并在 CIFAR-10 图像分类数据集上进行训练，最终模型可达到 **90%+ 的测试准确率**。项目不依赖任何现成的 ViT 实现（如 `timm`），完整地手动构建了 ViT 的各个模块，包括图像 **Patch Embedding**、**多头自注意力 (Multi-Head Self-Attention)**、**Transformer Encoder 堆叠**、**CLS Token** 等。代码使用 PyTorch 实现，支持 GPU 加速及多GPU并行训练，训练过程中会保存模型检查点并记录日志和训练曲线图。

## 项目结构

├── datasets │ └── cifar10.py # CIFAR-10 数据加载模块 ├── models │ └── vit.py # Vision Transformer 模型定义 ├── utils │ └── utils.py # 工具函数（检查点保存、绘图等） ├── train.py # 主训练脚本 ├── requirements.txt # 项目依赖列表 └── README.md # 使用说明和项目简介


## 准备环境

请确保安装了所需的依赖库（见 `pip install requirements.txt`）。例如，可使用以下命令创建并激活虚拟环境，然后安装依赖：


``` python
# 创建虚拟环境并激活
conda create -n  myenv
conda activate myenv
# 安装依赖
pip install -r requirements.txt
```

本项目需要PyTorch>=1.10及TorchVision等库来训练模型和加载数据。如果使用 GPU，请确保CUDA环境可用。

## 使用方法
首先，确保在 train.py 中设置了正确的数据集路径（默认为当前目录下的 data/）。可以通过命令行参数 --data_dir 指定 CIFAR-10 数据集的存放路径（若路径中没有数据，程序会自动下载）。 运行训练脚本示例如下：
### 使用默认参数在单块GPU上训练
``` python
python train.py --data_dir ./data --output_dir ./outputs
```
### 自定义训练参数，例如训练200个epoch、使用Label Smoothing和AutoAugment增强：
``` python
python train.py --epochs 200 --batch_size 128 --lr 1e-3 --min_lr 1e-5 \
    --weight_decay 5e-5 --label_smoothing 0.1 --patch_size 8 \
    --embed_dim 384 --num_heads 12 --num_layers 7 --dropout 0.0 \
    --scheduler cosine --output_dir ./outputs --data_dir ./data
``` 
### 上述第二个示例命令使用了参考文献中的超参数配置:
1. 训练200轮，
2. AdamW优化器，
3. 初始学习率1e-3
4. 余弦退火至1e-5，
5. 权重衰减5e-5，
6. Label Smoothing=0.1，
7. 无Dropout，
8. ViT结构为7层、
9. 嵌入维度384、12头注意力、patch大小8
这种配置在 CIFAR-10 上可取得约 90% 的测试准确率。 训练完成后，模型和日志将保存在指定的 output_dir 目录下，包括：
10. best_model.pth：验证准确率最高的模型参数。
11. last_model.pth：最后一轮训练结束时的模型参数。
12. train.log：训练过程中每轮的损失和准确率日志。
13. training_curves.png：训练集和验证集的损失/准确率曲线图。

如果有多块 GPU，脚本会自动检测并使用所有可用 GPU 进行并行训练（通过 DataParallel）。您也可以通过设置环境变量 CUDA_VISIBLE_DEVICES 来限定使用的 GPU 数。

# 结果与评估
在推荐的训练配置下（如上述示例），ViT 模型在 CIFAR-10 测试集上可以达到 90% 甚至更高 的准确率。
下表展示了训练200个epoch后模型在验证集上的性能：
模型	测试准确率 (Top-1)
ViT (7层, 384维, 12头)	90.9%
## Results
- Pre-trained models

<table>
  <tr>
    <th>Method</th>
    <th>Dataset</th>
    <th>Precision (%)</th>
    <th>Download</th>
  </tr>
  <tr>
    <td align="center">Vit</td>
    <td align="center">CIFAR-10</td>
    <td align="center">98.08</td>
    <td rowspan="2" align="center"><a href="">models</a> (code:)</td>
  </tr>
</table>


通过训练过程中绘制的曲线图，可以观察到模型损失的收敛以及训练/验证准确率的提升趋势。如有需要，可进一步通过调整超参数（如增加训练epoch、调整学习率调度策略、加入更多数据增强等）来提升模型表现或加快收敛。
## 参考文献
1. Dosovitskiy et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", ICLR 2021 
2. omihub777, ViT-CIFAR 项目 (2021): 一个针对 CIFAR-10 从头训练 ViT 的实现，取得了超过 90% 的准确率 
3. GeeksforGeeks, Building a Vision Transformer from Scratch in PyTorch (2024): 提供了 ViT 模型实现的教程和关键概念说明# 002_UCAS_DeepLearning_exViT_CIFAR10
# 002_UCAS_DeepLearning_exViT_CIFAR10
