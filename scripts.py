import tensorflow as tf
import os
import io
import numpy as np
import json
import onnx
from onnx import helper
from onnx import StringStringEntryProto

def h5_to_tflite(h5_model_path, tflite_model_path, quantize="none"):
    model = tf.keras.models.load_model(h5_model_path, compile=False)
    print(f"Input shape: {model.input_shape}, Output shape: {model.output_shape}")
    if quantize == "float16":
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
    else:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"TFLite Input details: {input_details}")
    print(f"TFLite Output details: {output_details}")
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)
    return tflite_model_path

def get_tf_model_info(model_path, output_path=None):
    if output_path is None:
        base = os.path.splitext(os.path.basename(model_path))[0]
        output_path = os.path.join(base, ".txt")

    with open(output_path, "w", encoding="utf-8") as f:
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
        except Exception as e:
            f.write("Failed to load model directly: " + str(e) + "\n")
            f.write("If the model uses custom layers/objects, pass a custom_objects dict to load_model.\n")
            raise

        f.write("\n=== Model.summary() ===\n")
        buf = io.StringIO()
        model.summary(print_fn=lambda s: buf.write(s + "\n"))
        f.write(buf.getvalue())

        f.write("\n=== Layers detail ===\n")
        for i, layer in enumerate(model.layers):
            name = layer.name
            cls = layer.__class__.__name__
            try:
                out_shape = layer.output_shape
            except Exception:
                out_shape = None
            params = layer.count_params()
            f.write(f"[{i}] {name} ({cls})  output_shape={out_shape}  params={params}\n")

        f.write("\n=== Total weights info ===\n")
        total = sum(w.size for w in model.get_weights())
        f.write(f"Total parameters (from get_weights arrays): {total}\n")

    return output_path

def test_h5_and_tflite_equivalence(h5_model_path, tflite_model_path):
    model = tf.keras.models.load_model(h5_model_path, compile=False)
    input_shape = model.input_shape
    test_input = np.random.random((1, *input_shape[1:])).astype(np.float32)


    h5_output = model.predict(test_input)

    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(output_details[0]['index'])

    if tf.reduce_all(tf.abs(h5_output - tflite_output) < 1e-5):
        print("Outputs are equivalent within tolerance.")
    else:
        print("Outputs differ!")

def tags_txt_to_json(tags_txt_path, tags_json_path):
    with open(tags_txt_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    with open(tags_json_path, "w", encoding="utf-8") as f:
        json.dump(lines, f, ensure_ascii=False, indent=2)

def add_metadata_to_onnx_model_file(onnx_model_path):
    model = onnx.load(onnx_model_path)
    metadata = {
        # === 原始模型信用 ===
        "original_model.name": "DeepDanbooru",
        "original_model.author": "Kichang Kim",
        "original_model.repository": "https://github.com/KichangKim/DeepDanbooru",
        "original_model.link": "https://github.com/KichangKim/DeepDanbooru/releases/tag/v3-20211112-sgd-e28",
        
        # === MIT 许可证信息 ===
        "license.type": "MIT",
        "license.url": "https://opensource.org/licenses/MIT",
        "license.terms": "This model is provided under MIT License. See original repository for full terms.",
        "license.attribution_required": "true",
        
        # === 模型技术规格 ===
        "model.type": "deep-learning",
        "model.task": "image-tagging",
        "model.domain": "computer-vision",
        "model.framework": "TensorFlow",  # 原始框架
        "model.input_shape": "1,512,512,3",
        "model.input_format": "RGB NHWC",
        "model.input_range": "0-1",
        "model.output_type": "tags-probabilities",
        "model.tags_count": "9176",  # 实际标签数量
        "model.tags_date": "2021/11/12 22:30:46",
        
        # === 你的处理信息 ===
        "name": "deepdanbooru-v3-20211112-sgd-e28-ONNX",
        "processed.by": "R4nd5tr(GitHub: https://github.com/R4nd5tr)",
        "processed.purpose": "Converted to ONNX format for deployment",
    }
    for key, value in metadata.items():        
        entry = StringStringEntryProto(key=key, value=value)
        model.metadata_props.append(entry)

    onnx.save(model, onnx_model_path)

def add_feature_vec_output(onnx_model_path):
    model = onnx.load(onnx_model_path)
    graph = model.graph

    # 找到最后的 ReLU 层输出
    relu_output_name = None
    for node in graph.node:
        if node.op_type == "Relu":  # 找到最后一个 ReLU 层
            relu_output_name = node.output[0]

    if relu_output_name is None:
        raise ValueError("No ReLU layer found in the model.")

    # 添加全局平均池化层
    pool_output_name = "feature_vec_output"
    pool_node = helper.make_node(
        "GlobalAveragePool",  # 使用全局平均池化
        inputs=[relu_output_name],
        outputs=[pool_output_name],
        name="GlobalAveragePool"
    )
    graph.node.append(pool_node)

    # 添加新的输出
    new_output = helper.make_tensor_value_info(
        pool_output_name,
        onnx.TensorProto.FLOAT,
        [None, 4096]  # 输出形状为 (None, 1, 1, 4096)，可以简化为 (None, 4096)
    )
    graph.output.append(new_output)

    # 保存修改后的模型
    onnx.save(model, onnx_model_path)
    print(f"Feature vector output added.")

if __name__ == "__main__":
    # test_h5_and_tflite_equivalence("deepdanbooru-v3-20211112-sgd-e28-model/model-resnet_custom_v3.h5", "model.tflite")
    # tags_txt_to_json("deepdanbooru-v3-20211112-sgd-e28-model/tags.txt", "deepdanbooru-v3-20211112-sgd-e28-model/tags.json")
    add_metadata_to_onnx_model_file("temp_model.onnx")
    # add_feature_vec_output("temp_model.onnx")
    pass