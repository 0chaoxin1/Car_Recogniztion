# Car_Recogniztion
ResNet152, StandfordCar-196, Accuracy:94.7%

Abstract： Aiming at the problem of low recognition rate between the same car series in fine-grained models, in order to enhance the representational ability of the convolutional neural network, a ResNet model with integrated independent components (IC-ResNet) was proposed. Firstly, ResNet was optimized to reduce the loss of feature information by improving the lower sampling layer, and then the center loss function and Softmax loss function were combined to improve the class cohesion of the model. Then an IC layer is introduced in front of the convolution layer to obtain relatively independent neurons, enhance network independence, and improve the feature representation ability of the model, so as to achieve more accurate classification of fine-grained vehicle models. The experiment shows that the model recognition accuracy on the Stanford cars-196 data set is 94.7%, which achieves the optimal effect compared with other models and verifies the effectiveness of the recognition model of this model.

model framework:
![Image text]( https://github.com/0chaoxin1/Car_Recogniztion/blob/main/model_framework.png)

train file:  
run train.py  
test file:   
  
1. get_model.py    

from ResNet152BDICcl import cnn_model01  
def get_model():  
&nbsp;&nbsp;&nbsp;&nbsp;model_weights_path = 'models/129-0.99853.hdf5'  
&nbsp;&nbsp;&nbsp;&nbsp;img_width, img_height = 224, 224  
&nbsp;&nbsp;&nbsp;&nbsp;num_channels = 3  
&nbsp;&nbsp;&nbsp;&nbsp;num_classes = 196  
&nbsp;&nbsp;&nbsp;&nbsp;model = cnn_model01(img_height, img_width, num_channels, num_classes)  
&nbsp;&nbsp;&nbsp;&nbsp;model.load_weights(model_weights_path, by_name=True)  
&nbsp;&nbsp;&nbsp;&nbsp;return model  

2. test.py  

from get_model import get_model  

if __name__ == '__main__':  
&nbsp;&nbsp;&nbsp;&nbsp;model = get_model()
&nbsp;&nbsp;&nbsp;&nbsp;pb = ProgressBar(total=100, prefix='Predicting test data', suffix='', decimals=3, length=50, fill='=')  
&nbsp;&nbsp;&nbsp;&nbsp;num_samples = 8041  
&nbsp;&nbsp;&nbsp;&nbsp;true_samples = 0  
&nbsp;&nbsp;&nbsp;&nbsp;cars_meta = scipy.io.loadmat('get/cars_test_annos_withlabels')  
&nbsp;&nbsp;&nbsp;&nbsp;annotations = cars_meta['annotations']  
&nbsp;&nbsp;&nbsp;&nbsp;start = time.time()  
&nbsp;&nbsp;&nbsp;&nbsp;out = open('./test_output/result.txt', 'a')  
&nbsp;&nbsp;&nbsp;&nbsp;y_test = []  
&nbsp;&nbsp;&nbsp;&nbsp;y_pred = []  
&nbsp;&nbsp;&nbsp;&nbsp;for i in range(num_samples):  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;filename = os.path.join('data/test', '%05d.jpg' % (i + 1))  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;bgr_img = cv.imread(filename)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;rgb_img = np.expand_dims(rgb_img, 0)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;rgb_img = rgb_img/255.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;preds = model.predict(rgb_img)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;prob = np.max(preds)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;class_id = np.argmax(preds)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;true_class = annotations[0][i][4]  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;out.write('{}\n'.format(str(class_id + 1)))  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# # 进度条  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# pb.print_progress_bar((i + 1) * 100 / num_samples)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 预测标签  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;y_pred.append((int(np.argmax(preds))+1))  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 真实标签  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;y_test.append(int(annotations[0][i][4]))  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;out.write('{}\n'.format(str(class_id + 1)))  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# 进度条  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;pb.print_progress_bar((i + 1) * 100 / num_samples)  
&nbsp;&nbsp;&nbsp;&nbsp;print(accuracy_score(y_test, y_pred))  
&nbsp;&nbsp;&nbsp;&nbsp;end = time.time()  
&nbsp;&nbsp;&nbsp;&nbsp;seconds = end - start  
&nbsp;&nbsp;&nbsp;&nbsp;use_times = seconds / 60.0  
&nbsp;&nbsp;&nbsp;&nbsp;print('test time: %.3f' % use_times, '分钟')  
&nbsp;&nbsp;&nbsp;&nbsp;out.close()  
&nbsp;&nbsp;&nbsp;&nbsp;K.clear_session()  
