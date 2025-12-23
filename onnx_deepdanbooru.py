import onnxruntime as ort
import numpy as np
import cv2
import json

# the model run in python have negligible difference with the model run in cpp implementation.

class ONNXDeepDanbooruModel:
    def __init__(self, model_path, json_path):
        self.model_path = model_path
        self.json_path = json_path
        self.model_name = "deepdanbooru-v3-20211112-sgd-e28-ONNX"
        self.input_name = "input_1"
        self.output_name = "activation_172"
        self.hash_output_name = "hash_output"
        self.input_shape = (1, 512, 512, 3)
        self.tag_threshold = 0.5
        self.system_tag_start_index = 9173
        self.output_count = 9176

        # Load ONNX model
        self.session = ort.InferenceSession(str(self.model_path), providers=["CUDAExecutionProvider"])

        # Load tags
        self.tags = self.load_tags()

    def load_tags(self):
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        if data.get("name") != self.model_name:
            raise RuntimeError("Invalid model name in JSON")
        tags = data.get("tags", [])
        return tags

    def preprocess_image(self, image_path):
        # Load image in RGB format
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize and pad to 512x512
        target_height, target_width = 512, 512
        image_height, image_width = image.shape[:2]
        scale = min(target_width / image_width, target_height / image_height)
        tx = (target_width - image_width * scale) / 2.0
        ty = (target_height - image_height * scale) / 2.0
        affine_matrix = np.array([[scale, 0, tx], [0, scale, ty]], dtype=np.float32)
        transformed = cv2.warpAffine(image, affine_matrix, (target_width, target_height), 
                                    flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)

        # Normalize to [0, 1] and convert to float32
        transformed = transformed.astype(np.float32) / 255.0

        # Convert to NHWC format (1, 512, 512, 3)
        input_tensor = np.expand_dims(transformed, axis=0)
        return input_tensor

    def infer_image(self, image_path):
        # Preprocess the image
        input_tensor = self.preprocess_image(image_path)

        # Run inference
        outputs = self.session.run([self.output_name, self.hash_output_name], {self.input_name: input_tensor})
        tag_probabilities = outputs[0].flatten()
        feature_vector = outputs[1].flatten()

        return tag_probabilities, feature_vector

    def postprocess_results(self, tag_probabilities, feature_vector):
        # Postprocess results
        tag_indexes = [i for i, prob in enumerate(tag_probabilities[:self.system_tag_start_index]) if prob >= self.tag_threshold]
        restrict_type = ["General", "Questionable", "Explicit"][np.argmax(tag_probabilities[self.system_tag_start_index:])]
        tag_results = [(self.tags[i], tag_probabilities[i]) for i in tag_indexes]

        # Print results
        print("Restrict Type:", restrict_type)
        print("Tags:")
        for tag, prob in tag_results:
            print(f"({prob:.6f}) {tag}")
        print("Feature Vector:", feature_vector)

if __name__ == "__main__":
    model_path = "converted.onnx"
    json_path = "translated_defalt.json"
    image_path = "test_image.jpg"  # Replace with your test image path

    onnx_model = ONNXDeepDanbooruModel(model_path, json_path)
    tag_probabilities, feature_vector = onnx_model.infer_image(image_path)
    onnx_model.postprocess_results(tag_probabilities, feature_vector)
