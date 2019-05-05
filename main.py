import numpy as np
from keras import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Flatten
from dataExtractor import load_dataset
from dataExtractor import extract_frames
from dataExtractor import lucas_kanae_optical_flow, dense_optical_flow


# Generate the optical flow data using dense optical flow
# for each frame where flow = |diff[img_i - img_i+1]| -> u,v, u is mag v is angle, v is currently discarded
def generate_optical_flow_data(images):
    print("Generating optical flow data using Dense optical flow algorithm")
    data = []
    num_images = int(len(images) / 2)
    for i in range(num_images):
        print(i,"/", num_images)
        # data.append(lucas_kanae_optical_flow(images[i], images[i+1]))  gives poor results
        data.append(dense_optical_flow(images[i], images[i+1]))
    return data


def load_training_data():
    frame_dir_location = "data/train_frames/"
    labels_location = "data/train_labels/train.txt"
    return load_dataset(frame_dir_location, labels_location)


def generate_model():
    model = Sequential()

    #input and flatten
    model.add(Dense(20, kernel_initializer='normal', activation='relu', name="input", input_shape=(480,640)))
    model.add(Flatten()) # Flatten reuslts to 1-d vec

    # hidden layers
    model.add(Dense(10, kernel_initializer='normal', activation='relu', name="first_hidden"))
    model.add(Dense(5, kernel_initializer='normal', activation='relu', name="second_hidden"))

    #output layer
    model.add(Dense(1,kernel_initializer='normal', activation='linear', name="output_layer", input_dim=1))
    optimizer = Adam(lr=0.0001)
    model.compile(loss='mean_absolute_error', optimizer=optimizer, metrics=['mean_absolute_error'])
    model.summary()
    return model


def train_model(train_data, train_labels, model):
    #train_data_one = np.array([train_data[:20]])
    test = []
    train = []
    # train on all but last 2 frames seconds
    for i in range(10115):
        train.append(np.array(train_data[i]))
        test.append(train_labels[i])
    train = np.array(train)
    # test on next 30 seconds
    test = np.array(test)
    #ex = np.array([train_labels[0]])
    model.fit(train, test, batch_size=1, epochs=8, class_weight=None)
    predict_test = []
    predict_test.append(np.array(train_data[10119]))
    predict_test = np.array(predict_test)
    print(model.predict(predict_test))
    print("actual value is :", train_labels[10119])


def main():
    print("Starting program")
    # train_vid_location = "videos/train.mp4"
    # extract_frames(train_vid_location, frame_dir_location)
    train_data, train_labels = load_training_data()
    train_data = generate_optical_flow_data(train_data)
    model = generate_model()
    model = train_model(train_data, train_labels, model)

main()
