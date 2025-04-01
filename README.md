# Vision Transformer CIFAR-10 图像分类项目

![outline](./asset/vit_figure.png)

本项目从零开始实现了 Vision Transformer (ViT) 模型，并在 CIFAR-10 图像分类数据集上进行训练，最终模型可达到 **80%+ 的测试准确率**。项目不依赖任何现成的 ViT 实现（如 `timm`），完整地手动构建了 ViT 的各个模块，包括图像 **Patch Embedding**、**多头自注意力 (Multi-Head Self-Attention)**、**Transformer Encoder 堆叠**、**CLS Token** 等。代码使用 PyTorch 实现，支持 GPU 加速及多GPU并行训练，训练过程中会保存模型检查点并记录日志和训练曲线图。

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
如果有多块 GPU，脚本会自动检测并使用所有可用 GPU 进行并行训练（通过 DataParallel）。您也可以通过设置环境变量 CUDA_VISIBLE_DEVICES 来限定使用的 GPU 数。

# 结果与评估
Precision: 83.76%  Recall: 83.85%  F1 Score: 83.75%
## 参考文献
1. Dosovitskiy et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", ICLR 2021 
2. omihub777, ViT-CIFAR 项目 (2021) 
3. GeeksforGeeks, Building a Vision Transformer from Scratch in PyTorch (2024)

