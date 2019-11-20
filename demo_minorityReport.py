import argparse
import logging
import time

import cv2
import numpy as np
from collections import deque
import pyautogui

from conv2d_training import create_model, build_classes, build_droplist
from sample_build_training_data import extract_action

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

#
classes, classes_idx = build_classes()  # 3 classes
# drop_score_i = [a * 3 - 1 for a in range(1, 20)]
drop_score_i = build_droplist()
ACTION_ROWS, ACTION_WIDTH = 5, 57 - len(drop_score_i)  # 5, 57 # 38 #26=57-31

model = create_model(ACTION_ROWS, ACTION_WIDTH, len(classes))
model.load_weights('3actions_look_move_next_action_conv2d.hdf5')

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_v2_large',
                        help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')

    parser.add_argument('--tensorrt', type=str, default="False",
                        help='for tensorrt process.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=str2bool(args.tensorrt))

    logger.debug('cam read+')
    cam = cv2.VideoCapture(args.camera)
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    action_q = deque(maxlen=5)
    # action_text_q = deque(maxlen=5)
    action_text_prev = ''

    action_text = ''

    while True:
        ret_val, image = cam.read()

        # logger.debug('image process+')
        image = cv2.flip(image, 1)  # mirror image.

        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

        action = extract_action(humans)
        if len(action) > 0:
            action_q.append(action)

        # logger.debug('postprocess+', humans)
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        # logger.debug('show+')
        if len(action_q) >= 5 and len(action) > 0:
            data = np.array(list(action_q))

            data = np.delete(np.array(data), drop_score_i, axis=1)
            data = np.reshape(data, (ACTION_ROWS, ACTION_WIDTH, 1))

            # data = np.reshape(data, (ACTION_HEIGHT, ACTION_WIDTH, 1))

            # print('data shape:', data.shape)
            predi = np.argmax(model.predict(np.array([data])))
            action_text = classes_idx.get(predi)
            # print('predi:', predi, ' ',action_text)
        else:
            action_text = ''

        cv2.putText(image,
                    "%s FPS: %f" % (action_text, 1.0 / (time.time() - fps_time)),
                    (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)

        # how to handle continuous inference? 10 frames?
        if action_text=='Movingmouse' or action_text_prev != action_text:
            # print('actionL', action_text)
            action_text_prev = action_text
            if action_text == 'Righthandnext':
                print('Righthandnext')
                pyautogui.hotkey('ctrl', 'tab')
            elif action_text == 'Movingmouse':
                print('Movingmouse')
                # leftWrist = humans[0].body_parts.get(4)
                data = np.array(list(action_q))
                # print('hand height:', data[4, 4 * 2 + 1])
                # by experiment.
                neck_y = data[4, 1 * 3 + 1]
                # print('neck_y:',neck_y)
                if (data[4, 4 * 3 + 1]) < neck_y:
                    pyautogui.vscroll(1)
                else:
                    pyautogui.vscroll(-1)
            else:
                print('Looking')
                1 + 1  # looking case

        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break
        # logger.debug('finished+')

    cv2.destroyAllWindows()

'''
tf_pose.common import CocoPart
    Nose = 0
    Neck = 1
    RShoulder = 2
    RElbow = 3
    RWrist = 4
    LShoulder = 5
    LElbow = 6
    LWrist = 7
    RHip = 8
    RKnee = 9
    RAnkle = 10
    LHip = 11
    LKnee = 12
    LAnkle = 13
    REye = 14
    LEye = 15
    REar = 16
    LEar = 17
    Background = 18
'''
