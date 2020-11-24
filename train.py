import keras
import os
from ResNet152BDcl import cnn_model02
from ResNet152BDICcl import cnn_model01
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau, TensorBoard
from keras.layers import Dense, Dropout, Input, GaussianNoise
import keras.backend as K
from keras.callbacks import LearningRateScheduler
import c_loss as cl
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
from pylab import *
from matplotlib.font_manager import FontProperties
# mpl.rcParams['font.sans-serif'] = ['SimHei']
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

img_width, img_height = 224, 224
num_channels = 3
train_data = 'data/train_all'
valid_data = 'data/test_196'
num_classes = 196
num_train_samples = 8144
num_valid_samples = 8041
verbose = 1
batch_size = 4
num_epochs = 150
patience = 50

if __name__ == '__main__':

    # def training_hist_one(hist,list):
    #     acc = hist.history['categorical_accuracy']
    #     loss = hist.history['loss']
    #
    #     # make a figure
    #     fig = plt.figure(figsize=(8, 4))  # igsize:指定figure的宽和高，单位为英寸
    #     # subplot loss
    #     ax1 = fig.add_subplot(121)  # 画布分割成1行2列，图像画在从左到右从上到下的第1块
    #     # new_ticks = np.linspace(1, 5, 5) # x轴的刻度显示在1到5之间，一共5个刻度
    #     # plt.xticks(new_ticks)
    #     ax1.plot(acc, label='accuracy')
    #     ax1.set_xlabel('epoch')
    #     ax1.set_ylabel('accuracy')
    #     # ax1.set_title('模型准确率')
    #
    #     # ax1.legend()
    #     # subplot acc
    #     ax2 = fig.add_subplot(122)
    #     ax2.plot(loss, label='loss')
    #     ax2.set_xlabel('epoch')
    #     ax2.set_ylabel('loss')
    #     # ax2.set_title('模型损失')
    #     # ax2.legend()
    #     plt.tight_layout()#tight_layout会自动调整子图参数，使之填充整个图像区域
    #     plt.savefig(list+'one.jpg')
    #     plt.show()
    #
    #
    # # 折线函数
    # def training_hist_two(hist_1,hist_2):
    #     acc_1 = hist_1.history['categorical_accuracy']
    #     loss_1 = hist_1.history['loss']
    #     acc_2 = hist_2.history['categorical_accuracy']
    #     loss_2 = hist_2.history['loss']
    #
    #     # make a figure
    #     fig = plt.figure(figsize=(8, 4))  # igsize:指定figure的宽和高，单位为英寸
    #     # subplot loss
    #     ax1 = fig.add_subplot(121)  # 画布分割成1行2列，图像画在从左到右从上到下的第1块
    #     # new_ticks = np.linspace(1, 5, 5) # x轴的刻度显示在1到5之间，一共5个刻度
    #     # plt.xticks(new_ticks)
    #     ax1.plot(acc_1, label='含IC')
    #     ax1.plot(acc_2, label='无IC')
    #     ax1.set_xlabel('迭代次数')
    #     ax1.set_ylabel('准确率')
    #     ax1.set_title('模型准确率')
    #
    #     ax1.legend()
    #     # subplot acc
    #     ax2 = fig.add_subplot(122)
    #     ax2.plot(loss_1, label='含IC')
    #     ax2.plot(loss_2, label='无IC')
    #     ax2.set_xlabel('迭代次数')
    #     ax2.set_ylabel('损失')
    #     ax2.set_title('模型损失')
    #     ax2.legend()
    #     plt.tight_layout()#tight_layout会自动调整子图参数，使之填充整个图像区域
    #     plt.savefig('two.jpg')
    #     plt.show()


    # 构建分类器模型
    model01 = cnn_model01(img_height, img_width, num_channels, num_classes)
    # model02 = cnn_model02(img_height, img_width, num_channels, num_classes)
    # 准备数据
    # keras图片生成器ImageDataGenerator 用以生成一个batch的图像数据
    # ImageDataGenerator()
    # rotation_range：整数，数据提升时图片随机转动的角度
    # width_shift_range：浮点数，图片宽度的某个比例，数据提升时图片水平偏移的幅度
    # height_shift_range：浮点数，图片高度的某个比例，数据提升时图片竖直偏移的幅度
    # zoom_range：浮点数或形如[lower,upper]的列表，随机缩放的幅度，若为浮点数，则相当于[lower,upper] = [1 - zoom_range, 1+zoom_range]
    # horizontal_flip：布尔值，进行随机水平翻转

    train_data_gen = ImageDataGenerator(rotation_range=20.,
                                        width_shift_range=0.1,
                                        height_shift_range=0.1,
                                        zoom_range=0.2,
                                        rescale=1. / 255,
                                        horizontal_flip = True)
    valid_data_gen = ImageDataGenerator( rescale=1. / 255)
    # 回调模型
    # keras.callbacks.TensorBoard()可视化工具
    # log_dir: 用来保存被 TensorBoard 分析的日志文件的文件名。
    # histogram_freq: 对于模型中各个层计算激活值和模型权重直方图的频率（训练轮数中）。
    # 如果设置成 0 ，直方图不会被计算。对于直方图可视化的验证数据（或分离数据）一定要明确的指出。
    # write_graph: 是否在 TensorBoard 中可视化图像。 如果 write_graph 被设置为 True，日志文件会变得非常大。
    # write_images: 是否在 TensorBoard 中将模型权重以图片可视化。
    tensor_board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    log_file_path = 'logs/training01.log'
    # log_file_path = 'logs/training02.log'
    csv_logger01 = CSVLogger(log_file_path, append=False)
    # log_file_path = 'logs/training02.log'
    # csv_logger02 = CSVLogger(log_file_path, append=False)
    # EarlyStopping()
    # monitor: 监控的数据接口，有’acc’,’val_acc’,’loss’,’val_loss’等等。
    # patience：能够容忍多少个epoch内都没有improvement。
    early_stop = EarlyStopping('categorical_accuracy', patience=patience)
    # keras.callbacks.ReduceLROnPlateau('val_acc', factor=0.1, patience=10,
    #                                    verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
    # 当评价指标不在提升时，减少学习率
    # monitor：被监测的量
    # factor：每次减少学习率的因子，学习率将以lr = lr*factor的形式被减少
    # patience：当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
    # mode：‘auto’，‘min’，‘max’之一，在min模式下，如果检测值触发学习率减少。在max模式下，当检测值不再上升则触发学习率减少。
    # epsilon：阈值，用来确定是否进入检测值的“平原区”
    # cooldown：学习率减少后，会经过cooldown个epoch才重新进行正常操作
    # min_lr：学习率的下限
    # reduce_lr = ReduceLROnPlateau('accuracy', factor=0.5, patience=int(patience / 10), verbose=1)
    trained_models_path = 'models/'
    # 模型保存为 路径 + epoch-验证精度.hdf5
    model_names = trained_models_path + '{epoch:02d}-{categorical_accuracy:.05f}.hdf5'
    # 阶跃型学习率下降
    # Lr = 0.001
    # def scheduler(epoch):
    #     if epoch <= 30 and epoch != 0:
    #         K.set_value(model.optimizer.lr, lr)
    #         print("lr changed to {}".format(lr ))
    #     elif epoch <=80 and epoch >30:
    #         K.set_value(model.optimizer.lr, lr * 0.1)
    #     elif epoch > 80 and epoch<= 120:
    #         model.lr.set_value(Lr*0.01)
    #     else:
    #         model.lr.set_value(Lr*0.001)
    #
    #     return model.lr.get_value()
    #
    # change_lr = LearningRateScheduler(scheduler)

    # def scheduler(epoch):
    #     # 每隔100个epoch，学习率减小为原来的1/10
    #     if epoch  >= 40 and epoch <= 100:
    #         Lr = model.optimizer.lr
    #         model.optimizer.lr.set_value(Lr*0.1)
    #         print("lr changed to {}".format(model.optimizer.lr))
    #     return model.optimizer.lr.get_value()


    # def scheduler(epoch):
    #     # 每隔100个epoch，学习率减小为原来的1/10
    #     if epoch  == 30 and epoch != 0:
    #         lr = K.get_value(model.optimizer.lr)
    #         K.set_value(model.optimizer.lr, lr * 0.5)
    #         print("lr changed to {}".format(lr * 0.5))
    #     elif epoch == 60:
    #         lr = K.get_value(model.optimizer.lr)
    #         K.set_value(model.optimizer.lr, lr * 0.5)
    #         print("lr changed to {}".format(lr * 0.5))
    #     elif epoch == 100:
    #         lr = K.get_value(model.optimizer.lr)
    #         K.set_value(model.optimizer.lr, lr * 0.5)
    #         print("lr changed to {}".format(lr * 0.5))
    #     return  K.get_value(model.optimizer.lr)
    #
    # reduce_lr = LearningRateScheduler(scheduler)

    def scheduler01(epoch):
        # 每隔100个epoch，学习率减小为原来的1/10
        if epoch  == 40 and epoch != 0:
            lr = K.get_value(model01.optimizer.lr)
            K.set_value(model01.optimizer.lr, lr * 0.1)
            print("lr changed to {}".format(lr * 0.1))
        if epoch  == 100 and epoch != 0:
            lr = K.get_value(model01.optimizer.lr)
            K.set_value(model01.optimizer.lr, lr * 0.1)
            print("lr changed to {}".format(lr * 0.1))
        return  K.get_value(model01.optimizer.lr)


    reduce_lr01 = LearningRateScheduler(scheduler01)

    # def scheduler02(epoch):
    #     # 每隔100个epoch，学习率减小为原来的1/10
    #     if epoch  == 40 and epoch != 0:
    #         lr = K.get_value(model02.optimizer.lr)
    #         K.set_value(model02.optimizer.lr, lr * 0.1)
    #         print("lr changed to {}".format(lr * 0.1))
    #     if epoch  == 100 and epoch != 0:
    #         lr = K.get_value(model02.optimizer.lr)
    #         K.set_value(model02.optimizer.lr, lr * 0.1)
    #         print("lr changed to {}".format(lr * 0.1))
    #     return  K.get_value(model02.optimizer.lr)
    #
    # reduce_lr02 = LearningRateScheduler(scheduler02)



    # ModelCheckpoint(filepath,monitor='val_loss',verbose=0,save_best_only=False,
    #                      save_weights_only=False, mode='auto', period=1)
    # filename：字符串，保存模型的路径
    # monitor：需要监视的值
    # verbose：信息展示模式，0或1(checkpoint的保存信息，类似Epoch 00001: saving model to ...)
    # save_best_only：当设置为True时，监测值有改进时才会保存当前的模型
    # mode：‘auto’，‘min’，‘max’之一，在save_best_only=True时决定性能最佳模型的评判准则，
    # 例如，当监测值为val_acc时，模式应为max，当监测值为val_loss时，模式应为min。在auto模式下，评价准则由被监测值的名字自动推断。
    # save_weights_only：若设置为True，则只保存模型权重，否则将保存整个模型（包括模型结构，配置信息等）
    # period：CheckPoint之间的间隔的epoch数
    model_checkpoint = ModelCheckpoint(model_names, monitor='categorical_accuracy', verbose=1, save_best_only=False)
    # 实时loo和accury
    # log_dir: 保存日志的目录的路径由TensorBoard解析的文件。
    # histogram_freq: 计算激活的频率(单位为epoch)以及模型各层的权重直方图。如果设为0，直方图不需要计算。
    #                 验证数据(或分割)必须是指定用于直方图可视化。
    # write_graph: 是否在TensorBoard中可视化图形。日志文件可能会变得非常大
    # write_images:是否写模型权重可视化为在TensorBoard形象。
    # keras.callbacks.TensorBoard(log_dir='./Graph',
    #                             histogram_freq=0,
    #                             write_graph=True,
    #                             write_images=True)
    # tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph',
    #                                          batch_size=batch_size,
    #                                          histogram_freq=0,
    #                                          write_graph=True,
    #                                          write_images=True)

    # callbacks = [tensor_board, model_checkpoint, csv_logger, early_stop, reduce_lr, tbCallBack]
    callbacks01 = [ csv_logger01, reduce_lr01]
    # callbacks02 = [csv_logger02, reduce_lr02]
    # 训练测试发生器
    # image.ImageDataGenerator.flow_from_directory()实现从文件夹中提取图片和进行简单归一化处理
    # categorical"会返回2D的one-hot编码标签,

    train_generator = train_data_gen.flow_from_directory(train_data, (img_width, img_height), batch_size=batch_size,
                                                         class_mode='categorical', shuffle=True)
    # valid_generator = valid_data_gen.flow_from_directory(valid_data, (img_width, img_height), batch_size=batch_size,
    #                                                      class_mode='categorical', shuffle=False)
    # 微调模型
    # keras中的fit_generator是keras用来为训练模型生成批次数据的工具

    # fit_generator(self, generator, steps_per_epoch, epochs=1, verbose=1, callbacks=None, validation_data=None,
    #           validation_steps=None, class_weight=None, max_q_size=10, workers=1, pickle_safe=False, initial_epoch=0)
    # generator：生成器函数，生成器的输出应该为：一个形如（inputs，targets）的tuple，一个形如（inputs, targets,sample_weight）的tuple。
    # steps_per_epoch：整数，当生成器返回steps_per_epoch次数据时计一个epoch结束，执行下一个epoch
    # verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
    # class_weight：规定类别权重的字典，将类别映射为权重，常用于处理样本不均衡问题。
    # callbacks=None,#list，list中的元素为keras.callbacks.Callback对象，在训练过程中会调用list中的回调函数
    hist01 = model01.fit_generator(
        train_generator,
        steps_per_epoch=num_train_samples / batch_size,
        # validation_data=valid_generator,
        # validation_steps=num_valid_samples,
        epochs=num_epochs,
        callbacks=callbacks01,
        verbose=verbose)
    print("ResNet152BDICcl训练结束！")
    # training_hist_one(hist01,'ResNet152BDICcl')

    # model02 = cnn_model02(img_height, img_width, num_channels, num_classes)
    # log_file_path = 'logs/training02.log'
    # csv_logger02 = CSVLogger(log_file_path, append=False)
    # def scheduler02(epoch):
    #     # 每隔100个epoch，学习率减小为原来的1/10
    #     if epoch  == 40 and epoch != 0:
    #         lr = K.get_value(model02.optimizer.lr)
    #         K.set_value(model02.optimizer.lr, lr * 0.1)
    #         print("lr changed to {}".format(lr * 0.1))
    #     if epoch  == 100 and epoch != 0:
    #         lr = K.get_value(model02.optimizer.lr)
    #         K.set_value(model02.optimizer.lr, lr * 0.1)
    #         print("lr changed to {}".format(lr * 0.1))
    #     return  K.get_value(model02.optimizer.lr)
    #
    # reduce_lr02 = LearningRateScheduler(scheduler02)
    # callbacks02 = [csv_logger02, reduce_lr02]
    # hist02 = model02.fit_generator(
    #     train_generator,
    #     steps_per_epoch=num_train_samples / batch_size,
    #     # validation_data=valid_generator,
    #     # validation_steps=num_valid_samples,
    #     epochs=num_epochs,
    #     callbacks=callbacks02,
    #     verbose=verbose)
    # print("ResNet152BDcl训练结束！")
    # training_hist_one(hist02,'ResNet152BDcl')
    # #
    # training_hist_two(hist01,hist02)
    # loss, accuracy = model.evaluate_generator(valid_generator, steps=num_valid_samples)
    # print("Test_loss is :", loss)




