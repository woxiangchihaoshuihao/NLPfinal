# Fraud-ADV-Detection

## 项目简介
- 基于中文 BERT 的交易欺诈文本分类，并包含基础对抗样本生成脚本。

## 安装
- 安装依赖：`pip install -r requirements.txt`

## 训练
- 运行：`python train.py --model_name bert-base-chinese --epochs 3 --batch_size 8`
- 训练数据路径：`data/train.json`

## 评估
- 运行：`python evaluate.py --model_dir models/out --file data/test.json`

## 对抗攻击（TextFooler-中文）
- 运行：`python attack/textfooler_cn.py --model_dir models/out --text "可疑转账，可能是欺诈"`

## 数据格式
- JSON 数组，元素包含 `text` 与 `label` 字段。

## 目录结构
- `data/` 训练与测试数据
- `models/` 模型与分类器模块
- `attack/` 词重要性估计与同义替换攻击
- `train.py` 训练脚本
- `evaluate.py` 评估脚本
