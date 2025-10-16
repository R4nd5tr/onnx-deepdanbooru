import tensorflow as tf
import onnx
from onnx2torch import convert
import tf2onnx
import torch

# convert original DeepDanbooru model from TensorFlow to PyTorch

# package version that works:
# torch==2.9.0
# tensorflow==2.17.0
# onnx==1.17.0
# tf2onnx==1.16.1
# onnx2torch==1.5.15
# flatbuffers==25.9.23
# protobuf==3.20.3
# use these versions to avoid weird version conflict problems

model = tf.keras.models.load_model(r".\deepdanbooru-v3-20211112-sgd-e28-model\model-resnet_custom_v3.h5", compile=False)
input_signature = [tf.TensorSpec([None, 512, 512, 3], tf.float32, name='input_1')]
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)
onnx.save(onnx_model, "temp_model.onnx")
pytorch_model = convert(onnx_model)
torch.save(pytorch_model.state_dict(), "converted_model.pth")
print("Saved PyTorch state_dict: converted_model.pth")



# full package list:
# Package                      Version
# ---------------------------- ---------
# absl-py                      2.3.1
# astunparse                   1.6.3
# certifi                      2025.10.5
# charset-normalizer           3.4.4
# filelock                     3.20.0
# flatbuffers                  25.9.23 <-
# fsspec                       2025.9.0
# gast                         0.6.0
# google-pasta                 0.2.0
# grpcio                       1.75.1
# h5py                         3.15.0
# idna                         3.11
# Jinja2                       3.1.6
# keras                        3.11.3
# libclang                     18.1.1
# Markdown                     3.9
# markdown-it-py               4.0.0
# MarkupSafe                   3.0.3
# mdurl                        0.1.2
# ml-dtypes                    0.4.1
# mpmath                       1.3.0
# namex                        0.1.0
# networkx                     3.5
# numpy                        1.26.4
# onnx                         1.17.0 <-
# onnx2torch                   1.5.15 <-
# opt_einsum                   3.4.0
# optree                       0.17.0
# packaging                    25.0
# pillow                       12.0.0
# pip                          25.2
# protobuf                     3.20.3 <-
# Pygments                     2.19.2
# requests                     2.32.5
# rich                         14.2.0
# setuptools                   80.9.0
# six                          1.17.0
# sympy                        1.14.0
# tensorboard                  2.17.1
# tensorboard-data-server      0.7.2
# tensorflow                   2.17.0 <-
# tensorflow-intel             2.17.0
# tensorflow-io-gcs-filesystem 0.31.0
# termcolor                    3.1.0
# tf2onnx                      1.16.1 <-
# torch                        2.9.0 <-
# torchvision                  0.24.0
# typing_extensions            4.15.0
# urllib3                      2.5.0
# Werkzeug                     3.1.3
# wheel                        0.45.1
# wrapt                        1.17.3