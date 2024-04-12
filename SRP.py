import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import to_categorical
from ReplayBufferProjection import ReplayBufferProjection

def power(x, y, p):
    res = 1
    x = x % p
    while (y > 0):
        if (y & 1):
            res = (res * x) % p
        y = y>>1 # y = y/2
        x = (x * x) % p
     
    return res

def miillerTest(d, n):
    a = 2 + random.randint(1, n - 4)
    x = power(a, d, n)
 
    if (x == 1 or x == n - 1):
        return True

    while (d != n - 1):
        x = (x * x) % n
        d *= 2
 
        if (x == 1):
            return False
        if (x == n - 1):
            return True
 
    # Return composite
    return False

def isPrime( n, k):
    if (n <= 1 or n == 4):
        return False
    if (n <= 3):
        return True
    d = n - 1
    while (d % 2 == 0):
        d //= 2
 
    for i in range(k):
        if (miillerTest(d, n) == False):
            return False
 
    return True

import math
import random
def get_random_hash_function():
    n = random.getrandbits(64)
    if n < 0: 
        n = -n 
    if n % 2 == 0:
        n = n + 1
    while not isPrime(n, 20):
        n = n + 1
    a = random.randint(2, n-1)
    b = random.randint(2, n-1)
    return (n, a, b)
# hash function fora number
def hashfun(hfun_rep, num):
    (p, a, b) = hfun_rep
    return (a * num + b) % p 

lamda = 0.1
delta = 0.05
w = int(math.e / lamda)
d = int(math.log(1/delta))
hash_functions = []
for i in range(d):
    hash_functions.append(get_random_hash_function())

accuracies = []
A = np.zeros((d,w))
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
            F = []
            for feature in x:
                feature = int(feature)
                F_row = []
                for k in range(d):
                    h = hashfun(hash_functions[k], feature)%w
                    if y==0:
                        A[k][h] += 1
                    F_row.append(A[k][h])
                F.append(min(F_row))
            grand_sum = np.sum(A)*len(x)
            prob = np.sum(F)/grand_sum
            if random.random() > prob:
                replay_buffer.add_example(x, y)

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