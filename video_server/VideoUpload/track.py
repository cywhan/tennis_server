from collections import deque
import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from video_server.VideoUpload import TrackNet
import csv


def track(path):
    model_path = r'D:\Programming Files\PycharmProjects\tennis_server\video_server\VideoUpload\model'
    model = load_model(model_path)
    input_video_path = path

    save_weights_path = r'D:\Programming Files\PycharmProjects\tennis_server\video_server\VideoUpload\weights\origin_model.h5'
    n_classes = 256

    video = cv.VideoCapture(input_video_path)
    fps = int(video.get(cv.CAP_PROP_FPS))
    output_width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    output_height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    current_frame = 0
    width, height = 640, 360
    img, img1, img2 = None, None, None

    # load TrackNet model
    model_fn = TrackNet.TrackNet
    m = model_fn(n_classes, input_height=height, input_width=width)
    m.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    m.load_weights(save_weights_path)

    q = deque([None for _ in range(8)])

    # both first and second frames cant be predict, so we directly write the frames to output video
    video.set(1, current_frame)
    ret, img1 = video.read()
    current_frame += 1
    img1 = cv.resize(img1, (width, height))
    img1 = img1.astype(np.float32)

    video.set(1, current_frame)
    ret, img = video.read()
    current_frame += 1
    img = cv.resize(img, (width, height))
    img = img.astype(np.float32)

    # frame, x, y logging
    log = [(fps, output_height, output_width)]

    while True:
        img2, img1 = img1, img
        video.set(1, current_frame)
        ret, img_raw = video.read()
        if not ret:
            break

        output_img = img

        img = img_raw.copy()
        img = cv.resize(img, (width, height))
        img = img.astype(np.float32)

        # combine three images to  (width , height, rgb*3)
        X = np.concatenate((img, img1, img2), axis=2)
        # since the ordering of TrackNet  is 'channels_first', so we need to change the axis
        X = np.rollaxis(X, 2, 0)
        # predict heatmap
        pr = m.predict(np.array([X]))[0]

        # since TrackNet output is ( net_output_height*model_output_width , n_classes )
        # so we need to reshape image as ( net_output_height, model_output_width , n_classes(depth) )
        # .argmax( axis=2 ) => select the largest probability as class
        pr = pr.reshape((height, width, n_classes)).argmax(axis=2)

        # cv2 image must be numpy.uint8, convert numpy.int64 to numpy.uint8
        pr = pr.astype(np.uint8)

        # reshape the image size as original input image
        heatmap = cv.resize(pr, (output_width, output_height))

        # heatmap is converted into a binary image by threshold method.
        ret, heatmap = cv.threshold(heatmap, 127, 255, cv.THRESH_BINARY)

        # find the circle in image with 2<=radius<=7
        circles = cv.HoughCircles(heatmap, cv.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2,
                                  maxRadius=7)

        # In order to draw the circle in output_img, we need to used PIL library
        # Convert opencv image format to PIL image format
        PIL_image = cv.cvtColor(output_img, cv.COLOR_BGR2RGB)
        PIL_image = Image.fromarray(PIL_image.astype(np.uint8))

        # check if there have any tennis be detected
        if circles is not None:
            # if only one tennis be detected
            if len(circles) == 1:
                x = int(circles[0][0][0])
                y = int(circles[0][0][1])
                print(current_frame, x, y)

                # path model 로 실제 공인지 판단
                stride = 10
                size = 50

                if y - size < 0:
                    y_r1 = 0
                    y_r2 = y + size
                elif y > output_height - size:
                    y_r1 = y - size
                    y_r2 = output_height
                else:
                    y_r1 = y - size
                    y_r2 = y + size

                if x - size < 0:
                    x_r1 = 0
                    x_r2 = x + size
                elif x > output_width - size:
                    x_r1 = x - size
                    x_r2 = output_width
                else:
                    x_r1 = x - size
                    x_r2 = x + size

                # probability 결과 표시할 이미지
                prob_img = np.zeros((output_height, output_width, 2), dtype=np.float64)
                # channel 0 = sum of probability  /  channel 1 = count of patchs included

                # patch 의 시작점부터 범위 내의 patch 들에 대해 반복
                patch_sy = y_r1
                while patch_sy + size < y_r2:
                    patch_sx = x_r1
                    while patch_sx + size < x_r2:
                        crop_raw = img_raw[patch_sy:patch_sy + size, patch_sx:patch_sx + size]
                        crop = crop_raw.reshape((1, 50, 50, 3))
                        crop_rst = model.predict(crop)
                        crop_persent = crop_rst[0, 0] * 100

                        # result probability patch
                        prob_tmp = np.ones((50, 50, 2), dtype=np.float64)
                        if crop_persent >= 99.9:  # 99.9 % 인 부분은 probability 값을 그대로 표시
                            prob_tmp[:, :, 0] = prob_tmp[:, :, 0] * crop_persent
                        else:  # 99.9 % 가 안되는 부분은 probability 값을 1/10 로 줄여서 표시. -> 잘 검출된 부분이 극대화 되어 보임.
                            prob_tmp[:, :, 0] = prob_tmp[:, :, 0] * crop_persent / 10

                        prob_add = cv.add(prob_img[patch_sy:patch_sy + size, patch_sx:patch_sx + size, :], prob_tmp)
                        prob_img[patch_sy:patch_sy + size, patch_sx:patch_sx + size, :] = prob_add

                        patch_sx += stride
                    patch_sy += stride

                prob_value = prob_img[:, :, 0] / prob_img[:, :, 1]
                prob_value[np.isnan(prob_value)] = 0

                if prob_value.max() >= 99.9:  # 실제 공이면 (정확도가 99.9% 이상)
                    log.append((current_frame, x, y))
                    q.appendleft([x, y])
                    q.pop()
                else:
                    print("No actual ball")
                    q.appendleft(None)
                    q.pop()
            else:
                q.appendleft(None)
                q.pop()
        else:
            q.appendleft(None)
            q.pop()

        current_frame += 1

    video.release()
    csv_path = path.split('.')[0] + "_track_log.csv"
    f = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(f)
    for i in log:
        csv_writer.writerow(i)
    f.close()

    return csv_path
