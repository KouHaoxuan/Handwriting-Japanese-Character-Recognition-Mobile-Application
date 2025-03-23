import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np


def load(f):
    return np.load(f)['arr_0']


# 加载数据
train_images = load('./kmnist-master/k49-train-imgs.npz')
test_images = load('./kmnist-master/k49-test-imgs.npz')
train_labels = load('./kmnist-master/k49-train-labels.npz')
test_labels = load('./kmnist-master/k49-test-labels.npz')

# 数据预处理
train_images = train_images.reshape((-1, 28, 28, 1)) / 255.0
test_images = test_images.reshape((-1, 28, 28, 1)) / 255.0

# 数据增强
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1)
datagen.fit(train_images)

# 模型构建
# 调整后的模型结构（更平衡的正则化）
model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(28, 28, 1)),
    keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.3),  # 降低Dropout比率

    keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.3),

    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001)),  # 增加维度
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(49)
])

# 添加学习率衰减的优化器
optimizer = keras.optimizers.Adam(learning_rate=0.001, decay=1e-4)
model.compile(optimizer=optimizer, loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 减弱数据增强
datagen = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.05
)
# 早停法回调
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 训练模型（使用数据增强）
history = model.fit(datagen.flow(train_images, train_labels, batch_size=32),
                    epochs=250,
                    validation_data=(test_images, test_labels),
                    callbacks=[early_stop])

# save the model
model.save('kmnist_model_improved.h5')
print("kmnist_model_improved.h5 is saved")

# visualization
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Evolution')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Evolution')
plt.legend()
plt.show()

# evaluation
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"lOSS: {test_loss:.4f}")
print(f"ACCURACY: {test_acc:.4f}")