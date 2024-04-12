import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os
from PIL import Image

import random
imagenet_dir = "coil100" # enter dataset directory
features = 64*64*3
num_classes = 100
task_count = 100
subdirs = [x[0] for x in os.walk(imagenet_dir)]
subdirs.pop(0)
data_x = []
data_y = []
known_classes = {}
imgs = {}

print(len(subdirs))
class_choices = random.sample(range(1,100), num_classes)
count = 0

for subdir in [subdirs[i] for i in class_choices]:
    for filename in os.listdir(subdir):
        if subdir not in known_classes:
            known_classes[subdir] = len(known_classes)
        file = os.path.join(subdir, filename)
        img = Image.open(file)
        if img.mode != "RGB":
            continue
        img = img.resize((64, 64))
        data_x.append(np.asarray(img))
        data_y.append(count)
        imgs[known_classes[subdir]] = img
    count += 1

data_x = np.asarray(data_x)
data_y = np.asarray(data_y)
data_x = np.reshape(data_x, (len(data_x), features))
data_x = data_x.astype('float32')

test_x = data_x[:1000]
test_y = data_y[:1000]
data_x = data_x[1000:]
data_y = data_y[1000:]

def randomize_data(x, y):
    combined = list(zip(x, y))
    random.shuffle(combined)
    x, y = zip(*combined)
    return np.array(x), np.array(y)

tasks = []

class_list = [i for i in range(num_classes)]
for i in [j*2 for j in range(task_count)]:
    task_classes = class_list[i:i+2]
    task_indices = [idx for idx, label in enumerate(data_y) if label in task_classes]
    task_data_x = [data_x[idx] for idx in task_indices]
    task_data_y = [data_y[idx] for idx in task_indices]
    task_data_x, task_data_y = randomize_data(task_data_x, task_data_y)
    tasks.append((task_data_x, task_data_y))

# Example usage
for i, task in enumerate(tasks):
    task_x, task_y = task
    print(f"Task {i+1}:")
    print("Classes:", set(task_y))
    print("Number of samples:", len(task_y))

def create_cnn_model():
    model = models.Sequential([
        layers.Reshape((64, 64, 3), input_shape=(12288,)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

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
delta = 0.0001
w = int(math.e / lamda)
d = int(math.log(1/delta))
hash_functions = []
for i in range(d):
    hash_functions.append(get_random_hash_function())

from tensorflow import keras

num_examples = len(data_x)
loss_list = []
class_list = []
count = 0

# data_x, data_y = randomize_data(data_x, data_y)
replay = []
tasks_temp = []
#for i, task in enumerate(tasks):
#    task_x, task_y = task
#    task_x_train, task_x_test, task_y_train, task_y_test = train_test_split(task_x, task_y, test_size=0.10, random_state=42)
#    tasks_temp.append((task_x_train, task_y_train))
#    replay.append((task_x_test, task_y_test))

# Define a loss function and an optimizer
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# Train the model one example at a time
def train_one_example(model, x, y):
    with tf.GradientTape() as tape:
        # Forward pass
        logits = model(x[np.newaxis, ...], training=True)
        loss_value = loss_fn(y, logits)

    # Compute gradients
    gradients = tape.gradient(loss_value, model.trainable_variables)

    # Apply gradients using optimizer
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return (loss_value.numpy(), model)

from ReplayBuffer import ReplayBuffer

buffer = ReplayBuffer(100,100)
probs = []
A = np.zeros((d,w))
for k in range(1):
    keras.utils.set_random_seed(k)
    model = create_cnn_model()
    loss_it = []
    for i, task in enumerate(tasks):
        task_x, task_y = task
        print(f"Task {i+1}:")
        for x,y in zip(task_x, task_y):
            count += 1
            loss, model = train_one_example(model, x, y)
            for (buff_x, buff_y) in buffer.get_random_sample():
                loss, model = train_one_example(model, buff_x, buff_y)
            loss_it.append(loss)
            class_list.append(y)
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
                buffer.add_example(x, y)


model.evaluate(test_x, test_y)