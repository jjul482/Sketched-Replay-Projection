import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import to_categorical
from ReplayBufferProjection import ReplayBufferProjection

accuracies = []
for runs in range(10):
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    features = 28*28

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    sorted_indices = sorted(range(len(y_train)), key=lambda k: y_train[k])

    x_train = x_train[sorted_indices]
    y_train = y_train[sorted_indices]
    print(y_train)
    print(y_train.shape)

    x_train = x_train.reshape(len(x_train), features)
    x_test = x_test.reshape(len(x_test), features)

    replay_buffer = ReplayBufferProjection(task_num=10, class_num=10, max_size=625, initial_features=features, k=1.5, random_state=runs+1, error=0.2)
    print(f"INITAL FEATURES = {features}, NEW SIZE = {replay_buffer.new_size}, ERROR = {replay_buffer.error}")
    model = Sequential([
        Input(shape = (replay_buffer.new_size)),
        Dense(256, activation='relu'),
        Dropout(0.2), 
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(10, activation='softmax') 
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def continual_learning_replay(new_x_train, new_y_train, epochs=1):
        replayed_examples_x, replayed_examples_y = replay_buffer.get_random_sample()
        if len(replayed_examples_x) > 0:
            combined_x_train = np.concatenate((replayed_examples_x, new_x_train), axis=0)
            combined_y_train = np.concatenate((replayed_examples_y, new_y_train), axis=0)
            indices = np.random.permutation(len(combined_x_train))
            combined_x_train = combined_x_train[indices]
            combined_y_train = combined_y_train[indices]
        else:
            combined_x_train = new_x_train
            combined_y_train = new_y_train
        
        for x,y in zip(new_x_train, new_y_train):
            replay_buffer.add_example(x,y)

        model.fit(combined_x_train, combined_y_train, epochs=epochs, batch_size=128, verbose=1)
        replay_buffer.update_R(new_y_train[0])

    x_train_initial = x_train[:5000]
    y_train_initial = y_train[:5000]
    x_train_initial = replay_buffer.transformer.fit_transform(x_train_initial)
    model.fit(x_train_initial, y_train_initial, epochs=1, batch_size=128, verbose=1)

    for i in range(10):
        new_x_train = x_train[5000 * (i + 1): 5000 * (i + 2)]
        new_y_train = y_train[5000 * (i + 1): 5000 * (i + 2)]
        new_x_train = replay_buffer.transformer.fit_transform(new_x_train)
        continual_learning_replay(new_x_train, new_y_train)

    x_test = replay_buffer.transformer.fit_transform(x_test)
    loss, accuracy = model.evaluate(x_test, y_test)
    accuracies.append(accuracy)
    print(f'Test accuracy: {accuracy * 100:.2f}%')