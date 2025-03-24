---
layout: post
toc: true
title: "MINI XCEPTION 模型学习"
categories: Note
tags: [AI, DeepLearning, CV, CNN]
author:
  - Vortez Wohl
---
# MINI XCEPTION 架构

From 《Real-time Convolutional Neural Networks for Emotion and Gender Classification》 by Oscar Arrigapoulos in 2017

 MINI XCEPTION 架构是一种轻量级的卷积神经网络架构，它受到了 XCEPTION 架构的启发，并结合了残差模块和深度可分离卷积。这种架构可以用于减少参数数量，降低计算成本，同时保持较高的准确率，适合在计算资源有限的终端进行实时视觉任务

## 特点

- 全卷积

    1. MINI XCEPTION 是一个全卷积神经网络架构，包含四个残差深度可分离卷积模块，每个卷积层后添加批量归一化层和 ReLU 函数

    2. 全卷积网络避免了过多的全连接层，减少模型中的权重参数数量，整个网络仅有不到 60,000 个参数，与传统 CNN 相比要减少了 80 余倍

- 高性能

    1. 其在基于 FER2013 数据集的情感识别任务上达到了 66% 的准确率，性能可观

- 轻量级

    1. 模型的权重最终只有 800 kB 不到，适合存储资源有限的终端设备

## 架构

1. Base Section：常规卷积层，批量归一化层并依靠 ReLU 激活

2. Module Section：4 个残差深度可分离卷积模块，每个模块后都紧随着批量归一化和 ReLU

3. Output Section：最后一个卷积层，用于将特征图转换成所需的输出维度

4. Global Average Pooling：使用通道均值对通道进行池化，将特征图降到一维

5. Softmax：用于分类模型的预测输出

```python
# tensorflow 实现

def create_mini_XCEPTION(input_shape, num_classes, l2_regularization=0.01) -> Model:
    regularization = l2(l2_regularization)

    # base
    img_input = Input(input_shape)
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
               use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
               use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # module 1
    residual = Conv2D(16, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(16, (3, 3), padding='same',
                        pointwise_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(16, (3, 3), padding='same',
                        pointwise_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 2
    residual = Conv2D(32, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(32, (3, 3), padding='same',
                        pointwise_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(32, (3, 3), padding='same',
                        pointwise_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 3
    residual = Conv2D(64, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(64, (3, 3), padding='same',
                        pointwise_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(64, (3, 3), padding='same',
                        pointwise_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 4
    residual = Conv2D(128, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(128, (3, 3), padding='same',
                        pointwise_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(128, (3, 3), padding='same',
                        pointwise_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    x = Conv2D(num_classes, (3, 3),
               # kernel_regularizer=regularization,
               padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax', name='predictions')(x)

    model = Model(img_input, output)
    return model
```

## 实验

基于 FER2013 对 MINI XCEPTION 进行训练，并完成人脸情绪识别任务

![alt text](/images/Mini-XCEPTION是什么/image-31.png)
