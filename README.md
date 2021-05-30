# PR-project

## 文件说明

本项目所有相关代码都在`codes`文件夹中，各模块及文件说明如下：

- `text`文件夹中为仅使用文本信息进行检索的相关方法代码
  - `ext_features.py`：抽取文本特征并存储
  - `run_bm25.py`：BM25模型训练及测试
  - `run_pca.py`：PCA模型训练及测试
  - `run_bert.py：BERT模型训练及测试`
  - `run_ensemble.py`：Ensemble方法实现及测试
  - `visualization.ipynb：可视化相关代码`
- `image`文件夹中为仅使用图像信息进行检索的相关方法代码
- `multimodal`文件夹中为使用混合信息进行检索的相关代码
  - `run_pca.py`：PCA模型训练及测试
  - `run_ensemble.py`：Ensemble方法实现及测试
- `utils`文件夹中为项目所使用的功能函数代码
  - `metrics.py`：评价指标函数相关代码
  - `split_data.py`：数据集划分相关代码
  - `text_utils.py`：文本处理的一些基础功能函数，如去停用词、词干提取等等。
  - `pairwise_generate.py`：生成深度模型训练所需的训练数据

## 环境准备

硬件要求：Ubuntu 18.04.5 LTS (GNU/Linux 4.15.0-143-generic x86_64)

软件要求：Anaconda3环境

依次执行以下命令完成环境配置：

```shell
conda create -n shopee python==3.8
conda activate shopee
pip install -r requirements.txt
```

## 数据准备

本项目所用数据为课程提供的清华云盘[数据](https://cloud.tsinghua.edu.cn/f/5c7ba8c55e04478d86d9/)，请下载后将数据文件放置在本项目代码根目录`codes`下。

下载完成后，依次执行

```shell
unzip -d ./data shopee-product-matching.zip
cd data
# 数据预处理
python preprocess.py
# 数据集划分
python split_data.py
# BERT模型训练数据生成
python pairwise_generate.py
```

即完成数据准备，可以进入模型训练及测试环节。

## 模型训练及测试

### 基于文本信息的检索模型

#### 模型训练及保存

```shell
cd text/
```

1. BM25模型

   ```python
   python run_bm25.py \
   		--data_dir ../data \
   		--save_dir ../result/text \
   		--threshold 18 \
   		--include_self \
       --do_train
   ```

2. PCA模型

   ```python
   # 提取特征
   wget .... # 下载Glove预训练模型
   python ext_features.py \
   		--model_name glove \   # 提取特征
   
   python run_pca.py \
   		--data_dir ../data \
     	--save_dir ../result/text \
       --n_components 15 \
       --threshold 0.75 \
       --whiten \
       --do_train 
   ```

3. BERT模型

   ```python
   python run_bert.py \
   		--data_dir ../data \
     	--model_name indo \
       --model_save_path ../result/text \
       --train_batch_size 123 \
       --num_epochs 8 \
       --do_train
   ```

#### 模型加载及测试

1. BM25模型
2. PCA模型
3. BERT模型

#### 最优模型结果复现

文本最优单一模型是基于BERT的深度语义匹配模型，因训练时间较长，为方便助教快速复现结果，我们提供使用训练完毕的模型文件：

```shell
# 最优模型文件下载
cd ../result/text/
wget ...
unzip -d 
```
使用最优模型在测试集上进行测试：

```python
# 特征提取
cd .././text
python ext_features.py \
		--model_name bert
```

1. 目标包含自身

   ```python
   python run_bert.py \
   		--model_dir ../result/text \
     	--threshold 0.7 \
     	--do_eval
   ```

   预期结果：

   ```shell
   F1:
   ```

2. 目标不包含自身

   ```python
   python run_bert.py \
   		--model_dir ../result/text \
   ```

   预期结果：

   ```shell
   F1: 
   ```

   

### 基于图像信息的检索模型

### 基于混合信息的检索模型

1. PCA模型
2. kNN模型
3. Ensemble方法



