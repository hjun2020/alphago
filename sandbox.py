from absl import app, flags
import tensorflow as tf
import numpy as np

# gpus = tf.config.list_physical_devices('GPU')
# print(gpus)
# for gpu in gpus:
#     print("Name:", gpu.name, "  Type:", gpu.device_type)


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))



strategy = tf.distribute.OneDeviceStrategy(device="/gPU:0")
with strategy.scope():
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)
    print(c)

tf.keras.layers.BatchNormalization(
    axis=-1,
    momentum=0.99,
    epsilon=0.001,
    center=True,
    scale=True,
    beta_initializer="zeros",
    gamma_initializer="ones",
    moving_mean_initializer="zeros",
    moving_variance_initializer="ones",
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None,
)


num = np.arange(4*4*4).reshape(4, 4, 4)
num = np.vsplit(num, 4)

print(num)
result = map(lambda x: np.squeeze(x), num)
print(list(result))
