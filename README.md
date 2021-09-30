# 项目目录结构

    -- checkpoints # 存放模型的地方
    -- data        # 定义各种用于训练测试的dataset
    -- eval.py     # 测试代码
    -- loss.py     # 定义各种loss
    -- metrics.py  # 定义各种约定俗成的评估指标
    -- model       # 定义各种实验中的模型
    -- options.py  # 定义各种实验参数，以命令行形式传入
    -- scripts     # 各种训练，测试脚本
    -- train.py    # 训练代码
    -- utils       # 各种工具代码
    -- README.md   # 介绍本repos

# 数据集的构建
    
    如何去获取数据集，通过其他论文中使用的数据集。以及一些数据增强的手段。
    数据集的完备性：数据集应该包括中英文的数据，还应该包括含有各种不同信噪比的噪声的数据。最终测试集也应该包括对这几项的验证，以此来保证模型的泛化性。
    针对不同的模型，其数据集的构建也是不一样的，这一点需要进行考虑。
    对于收集来的数据集，需要做不同的处理，使得之后呈现的一致的东西。
    这就需要考虑如何去存放数据集以及标签。
    将标签统一写成一帧一帧的形式，对于每一条数据，保存成一个文件。
    对于每一条数据，提取出特征，同样保存为一个文件。放在feats文件夹下。
    或者考虑
    -- 训练集
    -- 测试集
    -- 验证集

# 模型的选择
    
    模型的选择应该尽量简单，所使用的参数应该尽可能的少，因为到调度机上的算力有限，需要在模型尽量精简的情况下达到一个较高的准确率。
    目前尽量考虑DNN、LSTM以及CNN模型。
    在训练LSTM以及CNN模型的时候，数据集的构建也应该考虑以下，因为LSTM针对的是序列数据，那么一次应该输入一条完整的语音数据；而CNN则是一个二维的输入。这样数据就不能以帧为目的了。

# 特征的选取
    
    在这里考虑Mel频率倒谱系数(MFCC)。维数需要进一步考虑。目前考虑14维。

# 训练
    
# 损失函数

# 测试

# 验证

    

