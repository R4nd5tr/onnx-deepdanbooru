import tensorflow as tf
import os
import io
import json
import onnx
from onnx import helper
from onnx import StringStringEntryProto
import pickle
import numpy as np

import onnx_deepdanbooru
import pca_itq

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
    """在 ONNX 模型中添加特征向量输出层。"""
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
    pool_output_name = "feature_vec_global_avg_pool"
    pool_node = helper.make_node(
        "GlobalAveragePool",  # 使用全局平均池化
        inputs=[relu_output_name],
        outputs=[pool_output_name],
        name="FeatureVectorGlobalAveragePool"
    )
    graph.node.append(pool_node)

    # 添加 Squeeze 层以去掉多余的维度
    squeeze_output_name = "feature_vec_output"
    axes_tensor_name = "squeeze_axes"
    axes_initializer = helper.make_tensor(
        axes_tensor_name,
        onnx.TensorProto.INT64,
        [2],
        np.array([2, 3], dtype=np.int64)
    )
    graph.initializer.append(axes_initializer)
    
    squeeze_node = helper.make_node(
        "Squeeze",
        inputs=[pool_output_name, axes_tensor_name],
        outputs=[squeeze_output_name],
        name="FeatureVectorSqueeze"
    )
    graph.node.append(squeeze_node)

    # 添加新的输出
    new_output = helper.make_tensor_value_info(
        squeeze_output_name,
        onnx.TensorProto.FLOAT,
        [None, 4096]
    )
    graph.output.append(new_output)

    # 保存修改后的模型
    onnx.save(model, onnx_model_path)
    print(f"Feature vector output added.")

def collect_feature_vectors(image_dirs_json_file, model_path, json_path):
    onnx_model = onnx_deepdanbooru.ONNXDeepDanbooruModel(model_path, json_path)
    print("ONNX model loaded.")

    with open(image_dirs_json_file, 'r', encoding='utf-8') as f:
        image_dirs = json.load(f)
    print(f"Image directories to process: {image_dirs}")

    image_files = []
    for dir_path in image_dirs:
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif')):
                    image_files.append(os.path.join(root, file))
    print(f"Found {len(image_files)} images.")

    feature_vectors = []
    for image_file in image_files:
        try:
            _, feature_vector = onnx_model.infer_image(image_file)
            feature_vectors.append(feature_vector)
        except Exception as e:
            print(f"Error processing {image_file}: {e}")

        print(f"Processed {len(feature_vectors)}/{len(image_files)} images.", end='\r')

    with open('feature_vectors.pkl', 'wb') as f:
        pickle.dump(np.vstack(feature_vectors), f)

def train_pca_itq_model(feature_vectors_file):
    with open(feature_vectors_file, 'rb') as f:
        feature_vectors = pickle.load(f)
    print(f"Loaded {feature_vectors.shape[0]} feature vectors of dimension {feature_vectors.shape[1]}.")

    pca_itq_model = pca_itq.PCAITQHasher()
    pca_itq_model.train(feature_vectors)
    pca_itq_model.save()

def add_pca_itq_hash_output(onnx_model_path, pca_itq_pkl_path):
    """将 PCA-ITQ 哈希编码器添加为 ONNX 模型的输出层。需要先添加特征向量输出层。"""
    onnx_model = onnx.load(onnx_model_path)
    pca_itq_model = pca_itq.PCAITQHasher(file_path=pca_itq_pkl_path)
    pca_itq_onnx_model = pca_itq_model.to_onnx()

    # 合并两个 ONNX 模型
    combined_model = onnx.compose.merge_models(onnx_model, pca_itq_onnx_model, io_map=[('feature_vec_output', 'input')])
    onnx.save(combined_model, onnx_model_path)

if __name__ == "__main__":
    # test_h5_and_tflite_equivalence("deepdanbooru-v3-20211112-sgd-e28-model/model-resnet_custom_v3.h5", "model.tflite")
    # tags_txt_to_json("deepdanbooru-v3-20211112-sgd-e28-model/tags.txt", "deepdanbooru-v3-20211112-sgd-e28-model/tags.json")
    # collect_feature_vectors("test_img_dirs.json", "./cpp_deploy/bin/msvc/Debug/model/defalt.onnx", "./cpp_deploy/bin/msvc/Debug/model/defalt.json")
    # train_pca_itq_model("feature_vectors_stacked.pkl")
    # add_metadata_to_onnx_model_file("converted.onnx")
    # add_feature_vec_output("converted.onnx")
    # add_pca_itq_hash_output("converted.onnx", "pca_itq_model.pkl")
    pass
