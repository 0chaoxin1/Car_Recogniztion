# Car_Recogniztion
ResNet152, StandfordCar-196, Accuracy:94.7%

Abstract： Aiming at the problem of low recognition rate between the same car series in fine-grained models, in order to enhance the representational ability of the convolutional neural network, a ResNet model with integrated independent components (IC-ResNet) was proposed. Firstly, ResNet was optimized to reduce the loss of feature information by improving the lower sampling layer, and then the center loss function and Softmax loss function were combined to improve the class cohesion of the model. Then an IC layer is introduced in front of the convolution layer to obtain relatively independent neurons, enhance network independence, and improve the feature representation ability of the model, so as to achieve more accurate classification of fine-grained vehicle models. The experiment shows that the model recognition accuracy on the Stanford cars-196 data set is 94.7%, which achieves the optimal effect compared with other models and verifies the effectiveness of the recognition model of this model.

model framework:
![Image text]( https://github.com/0chaoxin1/Car_Recogniztion/blob/main/model_framework.png)

**train file:**
run train.py
**test file: **

1. get_model.py  
'''
from ResNet152BDICcl import cnn_model01

def get_model():
    model_weights_path = 'models/129-0.99853.hdf5'
    img_width, img_height = 224, 224
    num_channels = 3
    num_classes = 196
    model = cnn_model01(img_height, img_width, num_channels, num_classes)
    model.load_weights(model_weights_path, by_name=True)
    return model
'''
2. test.py
'''
from get_model import get_model

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
   
    out.close()
    K.clear_session()
'''

