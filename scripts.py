import tensorflow as tf
import os
import io
import numpy as np
import json

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

if __name__ == "__main__":
    # test_h5_and_tflite_equivalence("deepdanbooru-v3-20211112-sgd-e28-model/model-resnet_custom_v3.h5", "model.tflite")
    tags_txt_to_json("deepdanbooru-v3-20211112-sgd-e28-model/tags.txt", "deepdanbooru-v3-20211112-sgd-e28-model/tags.json")