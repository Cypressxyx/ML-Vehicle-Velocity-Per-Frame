import numpy as np
from keras import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, BatchNormalization
from dataExtractor import load_dataset
from dataExtractor import extract_frames
from dataExtractor import lucas_kanae_optical_flow, dense_optical_flow
import matplotlib.pyplot as plt

# Generate the optical flow data using dense optical flow
# for each frame where flow = |diff[img_i - img_i+1]| -> u,v, u is mag v is angle, v is currently discarded


def generate_optical_flow_data(images):
    print("Generating optical flow data using Dense optical flow algorithm")
    data = []
    num_images = int(len(images) - 1)
    for i in range(num_images):
        print(i, "/", num_images)
        # data.append(lucas_kanae_optical_flow(images[i], images[i+1]))  gives poor results
        data.append(np.array(dense_optical_flow(images[i], images[i+1])))
    print("Done and returning")
    del images
    return np.array(data)


def load_training_data():
    frame_dir_location = "data/train_frames/"
    labels_location = "data/train_labels/train.txt"
    return load_dataset(frame_dir_location, labels_location)


def generate_model():
    print("Generating Model")
    model = Sequential()

    #input and flatten
    #model.add(Conv1D(128, kernel_size=2, strides=2, activation='relu',
     #                name="input_conv_2d", input_shape=(210, 640)))
    model.add(Dense(210, kernel_initializer='normal', activation='relu', name="input", input_shape=(120, 640)))
    #model.add(Dense(64, kernel_initializer='normal', activation='relu', name="s"))
    model.add(BatchNormalization())
    #model.add(MaxPooling1D(pool_size=2))

    model.add(Dropout(0.5))

    model.add(Flatten()) # Flatten reuslts to 1-d vec

    # hidden layers
    model.add(Dense(64, kernel_initializer='normal', activation='relu', name="first_hidden"))
    model.add(Dense(10, kernel_initializer='normal', activation='relu', name="second_hidden"))

    #output layer
    model.add(Dense(1,kernel_initializer='normal', activation='linear', name="output_layer", input_dim=1))
    optimizer = Adam(lr=0.0001)
    #model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])
    model.compile(loss='mean_absolute_error', optimizer=optimizer, metrics=['mean_absolute_error'])
    model.summary()
    return model


def analyze_predictions(predicted_values, actual_values):
    predicted_values = predicted_values.flatten()
    num_values = len(actual_values)
    x = [i for i in range(num_values)]
    print(predicted_values)
    print("actual values are:", actual_values)


    # calculate mean error
    diff = predicted_values - np.array(actual_values)
    mean_square = sum(diff * diff) / len(actual_values)
    print("Mean square is:", mean_square)
    mean_square_text = "Mean square is: " + str(mean_square)

    plt.text(1,0, mean_square_text , family="sans-serif")
    plt.scatter(x, predicted_values)
    plt.scatter(x, actual_values, color='red')
    plt.plot(predicted_values, color='yellow')
    plt.plot(actual_values, color='orange')
    plt.ylabel("Velocity")
    plt.xlabel("Frame")
    plt.show()


def train_model(train_data, train_labels, model):
    print("Training model")
    train_labels = train_labels[1:len(train_data) + 1]

    indices = np.arange(train_data.shape[0])
    np.random.shuffle(indices)
    train_data = train_data[indices]
    train_labels = train_labels[indices]
    model.fit(train_data, train_labels, batch_size=8, epochs=8, class_weight=None)

    # predicted_values = model.predict(train_data)
    # analyze_predictions(predicted_values, train_labels)
    return model


def predict_values(train_data, train_labels, model):
    print("Predicting Values")
    train_labels = train_labels[1:len(train_data) + 1]
    predicted_values = []
    value = np.array([train_data[0]])
    predicted_values.append(model.predict(value))

    for i in range(1,len(train_data)):
        value = np.array([train_data[i]])
        prediction = model.predict(value)
        predicted_value = prediction[0][0]
        previous_value  = predicted_values[i-1][0][0]
        # Check if prediction make sense, if it doesnt scale it down or up
        if predicted_value > previous_value + 20 or predicted_value < previous_value - 20:
            # normalize value since we cant go from i-1 to 20 less/more in one frame
            print("correcting")
            normalized_constant = predicted_value / (previous_value + predicted_value)
            prediction[0][0] = 20 + normalized_constant if predicted_value > previous_value else 20 - normalized_constant
        predicted_values.append(prediction)
    predicted_values = np.array(predicted_values)
    analyze_predictions(predicted_values, train_labels)


def main():
    print("Starting program")
    # train_vid_location = "videos/train.mp4"
    # extract_frames(train_vid_location, frame_dir_location)
    train_data, train_labels = load_training_data()
    train_data = generate_optical_flow_data(train_data)
    model = generate_model()
    model = train_model(train_data, train_labels, model)
    predict_values(train_data, train_labels, model)


main()
