前言：

- 碍于个人的笔电显卡性能有限
  - 难以实现原实验中300epoch + 64BatchSize的参数条件
  - 运行速度不足以执行完论文中的实验
  - （我办法也不多，多少给点分就行qaq）
- 设计了一个简单的实验，通过对比开启关闭MoEx的训练结果，来验证MoEx抽象方法的有效性
- 同时在多种数据集上实验，证明MoEx抽象方法的普适性





**参数设定：**

- Epoch：5
- Batch_Size：10
- λ = 0.9



CIFAR100:关闭MoEx（moex_prob=0）

![Top(0.9,0)](MoEx工具验证.assets/Top(0.9,0).png)

CIFAR100:每组数据50%概率发生MoEx (moex_prob=0.5):

![Top(0.9 ,0.5)](MoEx工具验证.assets/Top(0.9 ,0.5).png)



CIFAR10:关闭MoEx：

![CIFAR10 TOP(0.9, 0)](MoEx工具验证.assets/CIFAR10 TOP(0.9, 0).png)

CIFAR10:每组数据50%概率发生MoEx (moex_prob=0.5):

![CIFAR10 Top(0.9,0.5)](MoEx工具验证.assets/CIFAR10 Top(0.9,0.5).png)



**验证结论：**

1. 对比训练结果的Top-err，发现使用MoEx之后，错误率有了一定的下降，证明MoEx增扩的图像对增加神经网络训练的鲁棒性有一定的效果。
2. 在CIFAR100与CIFAR10图像集上，MoEx方法均表现有效，证明MoEx增扩方法的普适性

