

#### torch

**torch.randperm**：

给定参数`n`，返回一个从`0` 到`n -1` 的随机整数排列。



`transforms.ToTensor()`

将尺寸为 (H x W x C) 且数据位于[0, 255]的PIL图片转换为数据类型为`np.uint8`的NumPy数组

或者直接接受数据类型为`np.uint8`的NumPy数组

然后转换为尺寸为(C x H x W)且数据类型为`torch.float32`且位于[0.0, 1.0]的`Tensor`。

**Channel在前方便卷积计算**

#### torchvision

.dataset：



transforms.Normalize(mean, std) 的计算公式：
 `input[channel] = (input[channel] - mean[channel]) / std[channel]`



Normalize() 函数的作用是将数据转换为标准正太分布，使模型更容易收敛。