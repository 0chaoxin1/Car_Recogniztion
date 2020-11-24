# -*- coding: utf-8 -*-

from keras.optimizers import SGD
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Flatten, Activation, add
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K
from sklearn.metrics import log_loss
from custom_layers.scale_layer import Scale
import sys
import numpy as np
from keras.datasets import cifar10
from keras import backend as K
from keras.utils import np_utils

# nb_train_samples = 3000 # 3000 training samples
# nb_valid_samples = 100 # 100 validation samples
# num_classes = 10

# sys.setrecursionlimit(3000)手工设置递归调用深度
sys.setrecursionlimit(3000)
def identity_block(input_tensor, kernel_size, filters, stage, block):
    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    # 卷积名
    conv_name_base = 'res' + str(stage) + block + '_branch'
    # 偏置
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    # 尺度
    scale_name_base = 'scale' + str(stage) + block + '_branch'
    # 步长（1,1）
    x = Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    # x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size),
                      name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    # x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    # x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    x = add([x, input_tensor], name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x

# y = F(x) + W*x
def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):

    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), strides=strides,
                      name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    # x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size),
                      name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    # x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    # x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    shortcut = Conv2D(nb_filter3, (1, 1), strides=strides,
                             name=conv_name_base + '1', use_bias=False)(input_tensor)
    shortcut = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '1')(shortcut)
    # shortcut = Scale(axis=bn_axis, name=scale_name_base + '1')(shortcut)

    x = add([x, shortcut], name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x

def cnn_model(img_rows, img_cols, color_type=1, num_classes=None):
    eps = 1.1e-5

    # 处理尺寸不同的后端
    global bn_axis
    if K.image_data_format() == 'channels_first':
      bn_axis = 3
      img_input = Input(shape=(img_rows, img_cols, color_type), name='data')
    else:
      bn_axis = 1
      img_input = Input(shape=(color_type, img_rows, img_cols), name='data')

    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)

    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name='bn_conv1')(x)
    # x = Scale(axis=bn_axis, name='scale_conv1')(x)
    x = Activation('relu', name='conv1_relu')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    # 特征维度较高，高层语义特征不够清晰
    # 八层[128, 128, 512]共24层
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    # 七层
    for i in range(1,8):
      x = identity_block(x, 3, [128, 128, 512], stage=3, block='b'+str(i))
    # 36x3=108层
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    for i in range(1,36):
      x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b'+str(i))

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x_fc = AveragePooling2D((7, 7), name='avg_pool')(x)
    x_fc = Flatten()(x_fc)
    # Dense(units:输出维度，activation=激活函数)全连接层（对上一层的神经元进行全部连接，实现特征的非线性组合）
    x_fc = Dense(num_classes, activation='softmax', name='fc1000')(x_fc)

    model = Model(img_input, x_fc)

    if K.image_data_format() == 'channels_first':
      # 使用预先训练过的权重进行Theano后端
      weights_path = 'models/resnet152_weights_th.h5'
    else:
      # 在Tensorflow后端使用预先训练的权重
      weights_path = 'models/resnet152_weights_tf.h5'

    model.load_weights(weights_path, by_name=True)

    # 截断并替换softmax层以进行传输学习
    # 不能使用model.layers.pop（），因为model不是Sequential（）类型
    # 下面的方法有效，因为预训练的权重存储在图层中但不存储在模型中
    x_newfc = AveragePooling2D((7, 7), name='avg_pool')(x)
    x_newfc = Flatten()(x_newfc)
    x_newfc = Dense(num_classes, activation='softmax', name='fc8')(x_newfc)

    model = Model(img_input, x_newfc)

    # 学习率为0.001  decay：每次更新时学习率衰减量
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    # 'categorical_crossentropy'：交叉熵损失函数
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model