import os
import time
import cv2 as cv
import keras.backend as K
import numpy as np
from console_progressbar import ProgressBar
from get_model import get_model
import scipy.io
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    model = get_model()

    pb = ProgressBar(total=100, prefix='Predicting test data', suffix='', decimals=3, length=50, fill='=')
    num_samples = 8041
    true_samples = 0

    cars_meta = scipy.io.loadmat('get/cars_test_annos_withlabels')
    annotations = cars_meta['annotations']

    start = time.time()
    out = open('./test_output/result.txt', 'a')
    y_test = []
    y_pred = []
    for i in range(num_samples):
        filename = os.path.join('data/test', '%05d.jpg' % (i + 1))
        bgr_img = cv.imread(filename)
        rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
        rgb_img = np.expand_dims(rgb_img, 0)
        rgb_img = rgb_img/255.
        preds = model.predict(rgb_img)
        prob = np.max(preds)
        class_id = np.argmax(preds)
        true_class = annotations[0][i][4]

        out.write('{}\n'.format(str(class_id + 1)))
        # # 进度条
        # pb.print_progress_bar((i + 1) * 100 / num_samples)
        # 预测标签
        y_pred.append((int(np.argmax(preds))+1))
        # 真实标签
        y_test.append(int(annotations[0][i][4]))

        out.write('{}\n'.format(str(class_id + 1)))
        # 进度条
        pb.print_progress_bar((i + 1) * 100 / num_samples)


    print(accuracy_score(y_test, y_pred))
    end = time.time()
    seconds = end - start
    use_times = seconds / 60.0
    print('test time: %.3f' % use_times, '分钟')
    print(precision_score(y_test, y_pred, average='macro'))
    print(precision_score(y_test, y_pred, average=None))
    print(recall_score(y_test, y_pred, average='macro'))
    print(recall_score(y_test, y_pred, average=None))
    print(f1_score(y_test, y_pred, average='macro'))
    print(f1_score(y_test, y_pred, average=None))
    #     if class_id + 1 == true_class:
    #         true_samples = true_samples + 1
    #


    # print('test acc: %.3f' % (float(true_samples * 100 / num_samples)), '%')
    # # print('test acc: {}'.format(float(true_samples * 100 / num_samples)),'%')



    out.close()
    K.clear_session()
