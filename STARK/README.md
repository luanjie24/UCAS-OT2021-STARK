# UCAS-OT2021-STARK
《视频处理与分析》个人作业
使用STARK模型进行目标跟踪

## 作业说明
单目标跟踪 主要根据CVPR2021的《Learning Spatio-Temporal Transformer for Visual Tracking》提出的STARK模型进行复现

参考文章：https://arxiv.org/abs/2103.17154
参考源码：https://github.com/researchmm/Stark
基于pytorch，所需环境参考源码的Install the environment部分

数据集相关资源：
http://got-10k.aitestunion.com
https://github.com/got-10k/toolkit
https://github.com/got-10k/siamfc


## 作业文件结构
   ```
├── checkpoints                                     # 模型存储文件夹
│
├── data                                            # 数据集文件夹
│   │
│   └── got10k 
│       │
│   	├── test
│       │
│   	├── train
│       │
│   	└── val 
│
├── experiments                                    	# 用于存储模型的参数设置
│   │
│   └── stark_s
│
├── initialize                                      # 用于初始化文件存储位置
│   
├── logs                                         	# 用于存储训练日志
│   
├── test                                         	# 由于存储测试结果的文件夹
│   
├── lib                                             # 核心代码文件夹
│   │
│   ├── config 										# 配置定义
│   │                                     
│   ├── models 										# 模型定义
│   │                             
│   ├── test                                		# 测试配置文件
│   │
│   ├── train                               	    # 训练配置文件
│   │
│   └── utils                         				# 一些工具          
│
├── train.py										# 训练入口文件
│
├── test.py											# 测试入口文件
│
└── README.md


   ```

## 注意事项
把got10k的数据集放到data文件夹
需要先初始化路径，运行initialize\create_default_local_file.py会自动生成相关路径。生成的路径会存储在lib\train\admin\local.py中，所以也可以直接对local.py进行编辑。
运行train.py即可训练，训练完成后运行test.py，测试结果会自动添加到test文件夹中。

## 改进方向
主要还是个人硬件算力有限，训练一轮要好久(也是因为快到截止日期有点赶...)，所以可能需要先优化实验环境再谈改进。
本实验只对空间模型进行了训练，因此第一个改进方向是尝试空间+时间的模型。
同样因为算力有限，在训练的时候，其实并没有过多的调整参数进行对比，因此对于模型参数和超参数的调整进行对比也是未来的一个方向。
进一步思考算力问题，其实运算量大也是由于transformer本身的结构（我认为主要是注意力机制）造成的，因此优化STARK模型中的Transformer结构是最根本的改进方向。
可以尝试稀疏空间的定位来优化注意力机制，也可以直接设计预训练任务进行预训练后再微调训练。

