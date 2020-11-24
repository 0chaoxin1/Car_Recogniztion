import cv2 as cv
# from CNN import cnn_model
from ResNet152BDICcl import cnn_model01


def get_model():
    model_weights_path = 'models/129-0.99853.hdf5'
    img_width, img_height = 224, 224
    num_channels = 3
    num_classes = 196
    model = cnn_model01(img_height, img_width, num_channels, num_classes)
    model.load_weights(model_weights_path, by_name=True)
    return model

# 绘制
def draw_str(dst, target, s):
    x, y = target
    # cv.putText(）在图像上绘制文字
    # dst：图像   s：文字   (x + 1, y + 1)：坐标
    # cv.FONT_HERSHEY_PLAIN：字体    1.0：字体大小
    # (0, 0, 0)：颜色    thickness：字体粗细
    cv.putText(dst, s, (x + 1, y + 1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)
