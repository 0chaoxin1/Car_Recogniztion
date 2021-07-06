
from keras.optimizers import SGD
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Flatten, Dropout, Activation, add, GaussianNoise
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K
from sklearn.metrics import log_loss
from scale_layer import Scale
import sys
import numpy as np
from keras.datasets import cifar10
from keras import backend as K
import c_loss02 as cl
from keras.utils import np_utils
# from GroupNorm import GroupNormalization
# import mxnet as mx
# import gluoncv

# nb_train_samples = 3000 # 3000 training samples
# nb_valid_samples = 100 # 100 validation samples
# num_classes = 10

# sys.setrecursionlimit(3000)手工设置递归调用深度
sys.setrecursionlimit(3000)
# def IC(input, p):
#     eps = 1.1e-5
#     x = BatchNormalization(epsilon=eps, axis=bn_axis)(input)
#     x = Scale(axis=bn_axis)(x)
#     x = Dropout(p)(x)
#     return x
# p = 0.01
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
    # x = IC(input_tensor, p)
    x = Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a', use_bias=False)(input_tensor)
    # x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    # x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)
    # x = IC(x, p)


    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    # x = IC(x, p)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size),
               name=conv_name_base + '2b', use_bias=False)(x)
    # x = GroupNormalization()(x)
    # x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    # x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)
    # x = IC(x, p)

    # x = IC(x, p)
    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
    # x = GroupNormalization()(x)
    # x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    # x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    x = add([x, input_tensor], name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    # x = IC(x, p)
    return x


def identity_block_tanh(input_tensor, kernel_size, filters, stage, block):
    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    # 卷积名
    conv_name_base = 'res' + str(stage) + block + '_branch'
    # 偏置
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    # 尺度
    scale_name_base = 'scale' + str(stage) + block + '_branch'
    # 步长（1,1）
    # x = IC(input_tensor, p)
    x = Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a', use_bias=False)(input_tensor)
    # x = GroupNormalization()(x)
    # x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    # x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)
    # x = IC(x, p)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    # x = IC(x, p)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size),
               name=conv_name_base + '2b', use_bias=False)(x)
    # x = GroupNormalization()(x)
    # x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    # x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)
    # x = IC(x, p)

    # x = IC(x, p)
    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
    # x = GroupNormalization()(x)
    # x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    # x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    x = add([x, input_tensor], name='res' + str(stage) + block)
    x = Activation('tanh', name='res' + str(stage) + block + '_relu')(x)
    # x = IC(x, p)
    return x


# y = F(x) + W*x
def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    # x = IC(input_tensor, p)
    x = Conv2D(nb_filter1, (1, 1), strides=strides,
               name=conv_name_base + '2a', use_bias=False)(input_tensor)
    # x = GroupNormalization()(x)
    # x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    # x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)
    # x = IC(x, p)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    # x = IC(x, p)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size),
               name=conv_name_base + '2b', use_bias=False)(x)
    # x = GroupNormalization()(x)
    # x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    # x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)
    # x = IC(x, p)

    # x = IC(x, p)
    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
    # x = GroupNormalization()(x)
    # x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    # x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    # shortcut = IC(input_tensor, p)
    shortcut = Conv2D(nb_filter3, (1, 1), strides=strides,
                      name=conv_name_base + '1', use_bias=False)(input_tensor)
    # shortcut = GroupNormalization()(shortcut)
    # shortcut = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '1')(shortcut)
    # shortcut = Scale(axis=bn_axis, name=scale_name_base + '1')(shortcut)

    x = add([x, shortcut], name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    # x = IC(x, p)
    return x


def conv_block_D(input_tensor, kernel_size, filters, stage, block):
    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    # x = IC(input_tensor, p)
    x = Conv2D(nb_filter1, (1, 1),
               name=conv_name_base + '2a', use_bias=False)(input_tensor)
    # x = GroupNormalization()(x)
    # x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    # x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)
    # x = IC(x, p)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    # x = IC(x, p)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size), strides=(2, 2),
               name=conv_name_base + '2b', use_bias=False)(x)
    # x = GroupNormalization()(x)
    # x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    # x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)
    # x = IC(x, p)

    # x = IC(x, p)
    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
    # x = GroupNormalization()(x)
    # x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    # x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)


    shortcut = AveragePooling2D((2, 2), strides=(2, 2), padding='same',
                                    name='AvgPloo' + str(stage))(input_tensor)
    # shortcut = IC(shortcut, p)
    shortcut = Conv2D(nb_filter3, (1, 1),
                      name=conv_name_base + '1', use_bias=False)(shortcut)
    # shortcut = GroupNormalization()(shortcut)
    # shortcut = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '1')(shortcut)
    # shortcut = Scale(axis=bn_axis, name=scale_name_base + '1')(shortcut)

    x = add([x, shortcut], name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    # x = IC(x, p)
    return x


def cnn_model02(img_rows, img_cols, color_type=1, num_classes=None):
    eps = 1.1e-5

    # 处理尺寸不同的后端
    global bn_axis
    # ## th : if image_dim_ordering = channels_first”数据组织为（3,128,128,128），
    # ## tf : ...=“channels_last”数据组织为（128,128,128,3）
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
        img_input = Input(shape=(img_rows, img_cols, color_type), name='data')
    else:
        bn_axis = 1
        img_input = Input(shape=(color_type, img_rows, img_cols), name='data')

    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)

    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=False)(x)
    # x = GroupNormalization()(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name='bn_conv1')(x)
    x = Scale(axis=bn_axis, name='scale_conv1')(x)
    x = Activation('relu', name='conv1_relu')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block_D(x, 3, [128, 128, 512], stage=3, block='a')
    for i in range(1, 8):
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='b' + str(i))

    x = conv_block_D(x, 3, [256, 256, 1024], stage=4, block='a')
    for i in range(1, 36):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b' + str(i))

    x = conv_block_D(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block_tanh(x, 3, [512, 512, 2048], stage=5, block='c')

    x_fc = AveragePooling2D((7, 7), name='avg_pool')(x)
    x_fc = Flatten()(x_fc)
    x_fc = Dropout(0.1)(x_fc)
    # Dense(units:输出维度，activation=激活函数)全连接层（对上一层的神经元进行全部连接，实现特征的非线性组合）
    x_fc = Dense(1000, activation='softmax', name='fc1000')(x_fc)

    model = Model(img_input, x_fc)
    # net = get_model('ResNet50_v1d', pretrained='117a384e')
    # you may modify it to switch to another model. The name is case-insensitive
    # model_name = 'ResNet152_v1d'
    # # download and load the pre-trained model
    # net = gluoncv.model_zoo.get_model(model_name, pretrained='cddbc86f')
    # net_params = net.collect_params()
    if K.image_data_format() == 'channels_first':
        # 使用预先训练过的权重进行Theano后端
        # weights_path = 'ResNet152_v1d.h5'
        weights_path = 'models/resnet152_weights_tf.h5'
    else:
        # 在Tensorflow后端使用预先训练的权重
        # weights_path = 'ResNet152_v1d.h5'
        weights_path = 'models/resnet152_weights_tf.h5'
    model.load_weights(weights_path, by_name=True)

    # 截断并替换softmax层以进行传输学习
    # 不能使用model.layers.pop（），因为model不是Sequential（）类型
    # 下面的方法有效，因为预训练的权重存储在图层中但不存储在模型中
    x_newfc = AveragePooling2D((7, 7), name='avg_pool')(x)
    x_newfc = Flatten()(x_newfc)
    x_newfc = Dropout(0.1)(x_newfc)
    x_newfc = Dense(num_classes, activation=None, name='fc8')(x_newfc)

    model = Model(img_input, x_newfc)

    # 学习率改为0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss=cl.loss, metrics=[cl.categorical_accuracy])

    return model


