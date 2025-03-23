import tensorflow as tf
from tensorflow import keras

model_keras = keras.models.load_model('kmnist_model_improved.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model_keras)
tflite_float_model = converter.convert()

converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()

quantize_model_size = len(tflite_quantized_model) / 1024

f = open('kmnist_model_improved.tflite', 'wb')
f.write(tflite_quantized_model)
f.close()
