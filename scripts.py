import tensorflow as tf
import os
import io
import torch


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

