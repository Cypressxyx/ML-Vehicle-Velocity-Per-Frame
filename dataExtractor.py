import os
import cv2
import numpy as np
from keras.utils import to_categorical

feature_params = dict(maxCorners=500,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

optical_flow_lk_params = dict(winSize=(10, 10),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# dense flow produces better results


def dense_optical_flow(frame_one, frame_two):

    # crop to only include roads
    frame_one = frame_one[240:360, 0:640]
    frame_two = frame_two[240:360, 0:640]

    hue_saturation_value = np.zeros_like(frame_one)
    hue_saturation_value[..., 1] = 255
    frame_one = cv2.cvtColor(frame_one, cv2.COLOR_BGR2GRAY)
    frame_two = cv2.cvtColor(frame_two, cv2.COLOR_BGR2GRAY)

    # equalize histogram of images, generates too much noise and produces bad results
    # frame_one = cv2.equalizeHist(frame_one)
    # frame_two = cv2.equalizeHist(frame_two)

    flow = cv2.calcOpticalFlowFarneback(frame_one, frame_two, None, 0.5, 5, 15, 3, 7, 1.5, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    hue_saturation_value[..., 0] = ang * 180 / np.pi / 2
    hue_saturation_value[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hue_saturation_value, cv2.COLOR_HSV2BGR)
    cv2.imshow('computer_vision', bgr)
    cv2.imshow('frame1', frame_one)
    cv2.imshow('frame2', frame_two)
    cv2.waitKey(30) & 0xff
    # angle is to miniscule, look into layer for better classification maybe but for now use magnitude
    return mag


def lucas_kanae_optical_flow(frame_one, frame_two):
    # Poor results given
    frame_two_copy = frame_two.copy()
    color = np.random.randint(0, 255, (100, 3))
    mask = np.zeros_like(frame_one)  # create mask using same size and set all values to 255
    corner_points = cv2.goodFeaturesToTrack(frame_one, mask=None, **feature_params)
    points_one, st, err = cv2.calcOpticalFlowPyrLK(frame_one, frame_two, corner_points, None, **optical_flow_lk_params)
    good_points = points_one[st == 1]
    good_original_points = corner_points[st == 1]

    # visual
    for i, (new, old) in enumerate(zip(good_points, good_original_points)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        frame_two_copy = cv2.circle(frame_two_copy, (a, b), 5, color[i].tolist(), -1)
    img = cv2.add(frame_two_copy, mask)
    cv2.imshow('frame', img)
    cv2.waitKey(30) & 0xff
    return points_one


def load_dataset(frame_dir, labels_location):
    print("Loading dataset")
    return load_frames(frame_dir), load_labels(labels_location)


def extract_frames(video_location, output_location):
    print("Extracting frames from video", video_location)
    video_capture = cv2.VideoCapture(video_location)
    success, image = video_capture.read()
    count = 0
    while success:
        success, image = video_capture.read()
        image_name = output_location + "frame_" + str(count) + ".jpg"
        cv2.imwrite(image_name, image)
        count += 1
    print("Done saving frames")

# Load frames in black and white image numpy array


def load_frames(frame_dir):
    test_loading = 0
    print("Loading frames from:", frame_dir)
    images = []
    file_list = os.listdir(frame_dir)
    file_list = sorted(file_list, key=lambda x: int((os.path.splitext(x))[0].split('_')[1]))
    for filename in file_list:
        if test_loading > 5000:
            break
        test_loading += 1
        file_location = frame_dir + filename
        images.append(cv2.imread(file_location, 1))
    return images


def load_labels(labels_location):
    if labels_location is None:
        return []
    print("Loading labels from:", labels_location)
    labels = open(labels_location, "r")
    train_labels = [float(label) for label in labels]
    return np.array(train_labels)
