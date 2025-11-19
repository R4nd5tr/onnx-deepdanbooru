import tensorflow as tf
import onnx
import tf2onnx

# convert original DeepDanbooru model from TensorFlow to PyTorch

# package version that works:
# tensorflow==2.17.0
# onnx==1.17.0
# tf2onnx==1.16.1
# flatbuffers==25.9.23
# protobuf==3.20.3
# use these versions to avoid weird version conflict problems

model = tf.keras.models.load_model(r".\deepdanbooru-v3-20211112-sgd-e28-model\model-resnet_custom_v3.h5", compile=False)
input_signature = [tf.TensorSpec([None, 512, 512, 3], tf.float32, name='input_1')]
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)
onnx.save(onnx_model, "temp_model.onnx")
