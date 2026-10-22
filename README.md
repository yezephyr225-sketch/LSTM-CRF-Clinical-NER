<div align="center">

# 基于 LSTM-CRF 模型的电子病历临床命名实体识别

**Clinical Named Entity Recognition in Electronic Medical Records Based on LSTM-CRF**

[![Python](https://img.shields.io/badge/Python-3.6+-blue.svg)](https://www.python.org/)
[![Keras](https://img.shields.io/badge/Keras-2.x-orange.svg)](https://keras.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![F1 Score](https://img.shields.io/badge/F1-93.4%25-brightgreen.svg)]()

针对中文电子病历（EMR）的命名实体识别（NER）方法，结合 **双向 LSTM 的序列建模能力** 与 **CRF 的标签转移优化**，
无需复杂特征工程即可从非结构化临床文本中抽取疾病、症状、身体部位、检查、治疗等医学实体。

**F1 值达到 93.4%** · 韶关学院

</div>

---

## 📽️ 项目演示 Demo

> 模型推理演示：输入中文电子病历文本，输出BIO格式实体标注结果。

[[Uploading 婕旂ず瑙嗛~1.zip…]()


▶️点击查看项目演示视频](这里粘贴GitHub自动生成你的视频链接)

> 演示说明：输入病历文本，模型自动识别身体部位、症状、检查、疾病、治疗五类医学实体，输出BIO标注序列。

---

## 📌 项目简介

电子病历（EMR）以数字化形式取代纸质病历，确保医疗信息的完整性与实时共享，显著提升临床效率。本项目面向中文电子病历
命名实体识别（Chinese-CNER），从非结构化文本中提取医学实体（疾病、症状、身体部位、检查、治疗等），为智能诊疗系统提供结构化数据支撑。

相比传统方法（HMM、标准 CRF），**LSTM-CRF** 无需复杂特征工程，通过 **LSTM 捕捉长距离上下文依赖 + CRF 建模标签转移关系**，
弥补了单独使用 LSTM 无法处理标签转移的局限，在不增加数据开销的情况下显著提升识别性能。

📄 **论文**：基于 LSTM-CRF 模型的电子病历临床命名实体识别（DOI: 10.69979/3029-2808.25.05.045）

---

## ✨ 核心亮点

- 🧠 **LSTM-CRF 联合建模**：双向 LSTM 序列编码 + CRF 标签转移优化
- 🏷️ **BIO 三段标注**：B（起始）/ I（内部）/ O（外部），并针对医学术语误切分问题做了实体保留预处理
- 🔁 **一致性检查（Consistency Check）**：校验预测标签类别与训练集标签命名的一致性，错误标签自动修正
- ➖ **负反馈机制**：将序列建模输出标签信息负反馈给模型，减少梯度爆炸/消失导致的不收敛
- 🐳 **一键 Docker 部署**：提供开箱即用的 Docker 镜像，免去环境配置烦恼
- 🔒 **隐私友好**：训练数据来自真实医院电子病历，涉及患者个人信息，故数据集不公开

---

## 🏗️ 系统架构
输入句子：曾在外院行右膝关节 DR 检查
│
▼
┌──────────────────────────────┐
│ 分段层 (Segment Layer) │ 对句子分词，并按 BIO 标记
│ 曾在 | 外院 | 行 | 右膝 | 关节 | DR | 检查
└──────────────────────────────┘
│
▼
┌──────────────────────────────┐
│ 输入嵌入层 (Embedding Layer) │ 字符嵌入 + 词嵌入（Word2Vec / GloVe 预训练）
└──────────────────────────────┘
│
▼
┌──────────────────────────────┐
│ LSTM-CRF 层 │ 双向 LSTM 编码 + Dense + CRF 序列标注
└──────────────────────────────┘
│
▼
┌──────────────────────────────┐
│ 一致性检查层 (Check Layer) │ 校验标签类别一致性，输出标注结果
└──────────────────────────────┘
│
▼
输出: o o o B-check B-check B-check o

### 1. 分段层（Segment Layer）
中文词语边界难以界定，医学术语（如"上腹部"）常被误切分。本项目**先从训练集提取所有临床命名实体并保留其在原句中的形式**，
再在此基础分词，最终使用 **BIO 标注**：词组起始为 `B`，内部为 `I`，其他为 `O`。
> 训练文本格式：每个词占一行，每句后留空行；每行至少两列，第一列为单词，最后一列为实体标签。

### 2. 输入嵌入层（Embedding Layer）
包含 **字符嵌入** 与 **词嵌入** 两部分，使用 Word2Vec 与 GloVe 预训练初始化查找表。
- 嵌入维度：`50`
- 窗口大小：`4`
- 迭代次数：`100`

### 3. LSTM-CRF 层
双向 LSTM 对词左右上下文联合编码，有效捕捉上下文语义，再经 Dense 层后送入 CRF 进行联合序列标注。

### 4. 一致性检查层（Consistency Check Layer）
统计训练集中 1344 个不同命名标签对，若预测标签类别与训练集原始标签不一致，自动修正为原始标签。

---

## 🧬 实体类别

| 实体类别 | 标签前缀 | 说明 |
|---------|---------|------|
| TREATMENT | B/I | 治疗方式 |
| BODY | B/I | 身体部位 |
| SIGNS | B/I | 症状和体征 |
| CHECK | B/I | 检查项目 |
| DISEASE | B/I | 疾病名称 |

标注示例：
```python
self.classdict = {
    'TREATMENT-B': 1, 'TREATMENT-I': 2,
    'BODY-B': 3, 'BODY-I': 4,
    'SIGNS-B': 5, 'SIGNS-I': 6,
    'CHECK-B': 7, 'CHECK-I': 8,
    'DISEASE-B': 9, 'DISEASE-I': 10,
    'O': 0
}

📊 实验结果
不同模型性能对比（同一数据集，无需预处理与特征工程）
模型结构	预处理策略	F1 值	增益
CRF	无	85.7%	—
LSTM	无	89.2%	+3.5%
LSTM-CRF	无	91.4%	+2.2%
LSTM-CRF	实体保留 + 一致性检查	93.4%	+2.0%
预处理策略增益（相同模型下）
模型结构	变体	F1 值
LSTM-CRF + 预处理	pretrain	91.6%
LSTM-CRF + 预处理	pretrain + dropout	92.3%
LSTM-CRF + 预处理	pretrain + dropout + char	93.0%
LSTM-CRF + 预处理	pretrain + dropout + char + CC	93.5%
结论：预训练输入嵌入提升约 +3.37，dropout 提升约 +1.43，char 嵌入提升约 +0.56，一致性检查层提升约 +2.26。
在多种分词工具（jieba /nlpir/stanford /hanlp）下，hanlp + LSTM-CRF 表现最佳，作为基准模型。
层 (Layer)	输出形状	参数量
embedding	(None, 150, 300)	527,788
bidirectional (LSTM)	(None, 150, 256)	439,296
dropout	(None, 150, 256)	0
bidirectional_1 (LSTM)	(None, 150, 128)	164,352
dropout_1	(None, 150, 128)	0
time_distributed (Dense)	(None, 150, 11)	1,419
crf (CRF)	(None, 150, 11)	275
Total	—	1,133,042
Trainable	—	605,342模型网络结构

🚀 快速开始
环境要求
系统：Ubuntu 20.04 (LTS) / Windows
Python 3.6+
Keras 2.x / TensorFlow 2.x

方式一：Docker（推荐）
docker pull dockerpush2878724536/keras:init
docker run -it --rm -v $(pwd):/workspace dockerpush2878724536/keras:init bash

方式二：本地安装
git clone https://github.com/yezephyr225-sketch/LSTM-CRF-Clinical-NER.git
cd LSTM-CRF

pip install -r requirements.txt   # 依赖：keras, tensorflow, keras-contrib, gensim, jieba 等
训练
python lstm_train.py
预测
python lstm_predict.py
运行后在 enter a sentence: 提示符后输入中文病历文本，即可输出实体标注结果：
输入: 曾在外院行右膝关节DR检查
输出: o o o B-check B-check B-check o
推理所需的三个文件（后缀对上即可）：
model0.bin
model0.h5
vocab.txt
📁 项目结构
LSTM-CRF/
├── lstm_train.py          # 模型训练脚本
├── lstm_predict.py        # 模型推理/预测脚本
├── transfer_data.py       # 数据转换脚本（原始标注 → 模型格式）
├── data/                  # 训练数据目录
├── data_origin/           # 原始数据目录
├── model/
│   ├── model0.bin         # 训练好的模型权重
│   ├── model0.h5          # Keras 模型文件
│   ├── vocab.txt          # 词表文件
│   └── readme.txt
└── README.md

🔧 核心代码片段
模型搭建（关键结构）：
# 嵌入层 + 双向 LSTM 编码
model.add(embedding_layer)
model.add(Bidirectional(LSTM(128, return_sequences=True)))

# 降维 + 特征提取
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(self.NUM_CLASSES)))

# CRF 序列标注层
crf_layer = CRF(self.NUM_CLASSES, sparse_target=True)
model.add(crf_layer)
model.compile('adam', loss=crf_layer.loss_function, metrics=[crf_layer.accuracy])
model.summary()

📚 引用
如果你在研究中使用了本项目，请引用：
@article{ye2025lstmcrf,
  title={基于 LSTM-CRF 模型的电子病历临床命名实体识别},
  author={叶振鹏 and 蔡凯文 and 张家豪},
  journal={医学研究},
  year={2025},
  doi={10.69979/3029-2808.25.05.045}
}
📄 许可
本项目代码遵循开源协议发布，原始电子病历数据因涉及患者隐私不予公开。
作者：叶振鹏・蔡凯文・张家豪（韶关学院）
研究方向：深度学习

---

### 配套 .gitignore（另存为 `.gitignore` 文件）
```gitignore
model/*.bin
model/*.h5
*.bin
*.h5
__pycache__/
*.pyc
*.pkl
.vscode/
.idea/
