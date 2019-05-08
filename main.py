import random
import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras.optimizers import Adam
from dataExtractor import load_dataset
from dataExtractor import extract_frames
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from dataExtractor import lucas_kanae_optical_flow, dense_optical_flow

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
    return data


def generate_data(data_dir, labels_loc):
    data_dir = "data/" + data_dir + "/"
    labels_loc = "data/" + labels_loc  if labels_loc is not None else None
    data, data_labels = load_dataset(data_dir, labels_loc)
    return generate_optical_flow_data(data), data_labels


def generate_model():
    print("Generating Model")
    model = Sequential()
    model.add(Dense(210, kernel_initializer='normal', activation='relu', name="input", input_shape=(120, 640)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Flatten())

    # hidden layers
    model.add(Dense(64, kernel_initializer='normal', activation='relu', name="first_hidden"))
    model.add(Dense(10, kernel_initializer='normal', activation='relu', name="second_hidden"))

    # output layer
    model.add(Dense(1,kernel_initializer='normal', activation='linear', name="output_layer", input_dim=1))
    optimizer = Adam(lr=0.0001)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])
    # model.compile(loss='mean_absolute_error', optimizer=optimizer, metrics=['mean_absolute_error'])
    model.summary()
    return model


def analyze_predictions_labels(predicted_values, actual_values):
    num_values = len(predicted_values)
    x = [i for i in range(num_values)]

    predicted_values = predicted_values.flatten()
    print(predicted_values)
    print("actual values are:", actual_values)

    # calculate mean squared error
    diff = predicted_values - np.array(actual_values)
    mean_square = sum(diff * diff) / len(actual_values)
    print("Mean square is:", mean_square)
    mean_square_text = "Mean square is: " + str(mean_square)

    plt.text(1, 0, mean_square_text , family="sans-serif")
    plt.scatter(x, predicted_values)
    plt.scatter(x, actual_values, color='red')
    plt.plot(predicted_values, color='yellow')
    plt.plot(actual_values, color='orange')
    plt.ylabel("Velocity")
    plt.xlabel("Frame")
    plt.show()


def generate_oversampled_data(data, labels):
    # find highest group, where group is i < k < i + 5
    num_sections = 50
    split_values = [[i, i+5] for i in range(0, num_sections, 5)]
    value_spread = [0 for _ in range(0,num_sections)]
    indices = [[] for _ in range(0, num_sections)]

    for i in range(0, len(labels)):
        label = labels[i]
        for j in range(0, len(split_values)):
            split_value = split_values[j]
            if split_value[0] <= label <= split_value[1]:
                value_spread[j] += 1
                indices[j].append(i)
                break

    plt.bar(range(len(value_spread)), value_spread)
    plt.show()
    plt.close()
    copy = value_spread.copy()
    value_spread = [value for value in value_spread if value != 0]  # remove zeros
    indices = [indices[i] for i in range(len(copy)) if copy[i] != 0]
    max_group = max(value_spread)  # Get highest group
    diff = [max_group - value for value in value_spread]

    # oversample each group using the number needed found in diff.
    for i in range(len(value_spread)):
        for j in range(diff[i]):
            value_to_copy = random.choice(indices[i])  # choose a random index to oversample
            data.append(data[value_to_copy])
            labels.append(labels[value_to_copy])

    # replot to make sure its right
    num_sections = 50
    split_values = [[i, i+5] for i in range(0, num_sections, 5)]
    value_spread = [0 for _ in range(0,num_sections)]
    indices = [[] for _ in range(0, num_sections)]

    for i in range(0, len(labels)):
        label = labels[i]
        for j in range(0, len(split_values)):
            split_value = split_values[j]
            if split_value[0] <= label <= split_value[1]:
                value_spread[j] += 1
                indices[j].append(i)
                break

    plt.bar(range(len(value_spread)), value_spread)
    plt.show()
    plt.close()
    return data, labels


def train_model(train_data, train_labels, model):
    print("Training model")
    # oversample groups in sections of 5
    train_labels = train_labels[1:len(train_data) + 1]
    generate_oversampled_data(train_data, train_labels)

    train_data = np.array(train_data)
    train_labels = np.array(train_labels)

    # randomize input
    indices = np.arange(train_data.shape[0])
    np.random.shuffle(indices)
    train_data = train_data[indices]
    train_labels = train_labels[indices]

    model.fit(train_data, train_labels, batch_size=8, epochs=8, class_weight=None)
    return model


def predict_values(train_data, model):
    print("Predicting Values")
    predicted_values = []
    train_data = np.array(train_data)
    value = np.array([train_data[0]])
    predicted_values.append(model.predict(value))

    for i in range(1, len(train_data)):
        value = np.array([train_data[i]])
        prediction = model.predict(value)
        predicted_value = prediction[0][0]
        previous_value = predicted_values[i-1][0][0]
        limit = 15
        # Check if prediction make sense, if it doesnt scale it down or up
        if predicted_value < 0:
            prediction[0][0] = 0
        elif predicted_value > previous_value + limit or predicted_value < previous_value - limit:
            # normalize value since we cant go from i-1 to 20 less/more in one frame
            print("correcting")
            normalized_constant = predicted_value / (previous_value + predicted_value)
            prediction[0][0] = limit + normalized_constant if predicted_value > previous_value else limit - normalized_constant
        predicted_values.append(prediction)
    predicted_values = np.array(predicted_values)
    return predicted_values


def generate_frames(vid_location, frame_dir_location):
    vid_location = "videos/" + vid_location
    frame_dir_location = "data/" + frame_dir_location + "/"
    extract_frames(vid_location, frame_dir_location)


def plot_and_save_predictions(predictions):
    print("Saving and plotting predictions")
    num_values = len(predictions)
    x = [i for i in range(num_values)]
    predictions = predictions.flatten()
    plt.scatter(x, predictions)
    plt.plot(predictions, color='yellow')
    plt.ylabel("Velocity")
    plt.xlabel("Frame")
    plt.show()

    file = open('data/test_labels.txt', 'w')
    for prediction in predictions:
        file.write(str(prediction) + "\n")
    file.close()


def main():
    # generate_frames("train.mp4", "train_frames")
    # generate_frames("test.mp4", "test_frames")
    print("Starting training")
    train_data, train_labels = generate_data("train_frames", "train_labels/train.txt")
    model = generate_model()
    model = train_model(train_data.copy(), train_labels.copy(), model)
    predicted_train_values = predict_values(train_data, model)
    analyze_predictions_labels(predicted_train_values, train_labels[1:len(train_data) + 1])

    print("Running testing video")
    del train_data
    del train_labels
    test_data = generate_data("test_frames", None)[0]
    predictions = predict_values(test_data, model)
    plot_and_save_predictions(predictions)


main()
