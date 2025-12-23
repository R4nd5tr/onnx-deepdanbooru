import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import onnx
from onnx import helper
import pickle

class PCAITQHasher:
    def __init__(self, file_path="", output_bits=512, seed=0):
        if file_path:
            self.load(file_path)
            self.output_bits = self.pca.n_components_
            return
        self.output_bits = output_bits

        # initialize scaler and PCA
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=output_bits)

        # initialize ITQ
        np.random.seed(seed)
        self.rotation_matrix = np.random.randn(self.output_bits, self.output_bits)
        u, _, vt = np.linalg.svd(self.rotation_matrix, full_matrices=False)
        self.rotation_matrix = u @ vt

    def save(self, file_path="pca_itq_model.pkl"):
        with open(file_path, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'pca': self.pca,
                'rotation_matrix': self.rotation_matrix
            }, f)

    def load(self, file_path="pca_itq_model.pkl"):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            self.scaler = data['scaler']
            self.pca = data['pca']
            self.rotation_matrix = data['rotation_matrix']

    def train(self, feature_vectors, itq_iterations=50):
        # Fit PCA
        feature_vectors_scaled = self.scaler.fit_transform(feature_vectors)
        pca_transformed = self.pca.fit_transform(feature_vectors_scaled)
        
        # ITQ optimization
        for _ in range(itq_iterations):
            v = pca_transformed @ self.rotation_matrix
            b = np.sign(v)
            c = b.T @ pca_transformed
            u, _, vt = np.linalg.svd(c)
            self.rotation_matrix = vt.T @ u.T

    def encode(self, feature_vector):
        feature_vector = self.scaler.transform(feature_vector.reshape(1, -1))
        pca_transformed = self.pca.transform(feature_vector)
        v = pca_transformed @ self.rotation_matrix
        binary_code = (v >= 0).astype(np.uint8)
        return binary_code.flatten()

    def to_onnx(self, opset_version=13):
        input_dim = self.pca.n_features_in_
        output_dim = self.output_bits

        input_tensor = helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, [None, input_dim])
        output_tensor = helper.make_tensor_value_info('hash_output', onnx.TensorProto.BOOL, [None, output_dim])

        scale_mean = self.scaler.mean_
        scale_var = self.scaler.scale_
        pca_components = self.pca.components_.T # Transpose to match ONNX Gemm input
        rotation_matrix = self.rotation_matrix

        nodes = []

        # Standardization
        mean_node = helper.make_node(
            'Sub',
            inputs=['input', 'mean'],
            outputs=['centered'],
            name="StandardizationMean"
        )
        scale_node = helper.make_node(
            'Div',
            inputs=['centered', 'scale'],
            outputs=['scaled'],
            name="StandardizationScale"
        )
        nodes.extend([mean_node, scale_node])

        # PCA Transformation
        pca_node = helper.make_node(
            'Gemm',
            inputs=['scaled', 'pca_components'],
            outputs=['pca_transformed'],
            alpha=1.0,
            beta=0.0,
            name="PCATransformation"
        )
        nodes.append(pca_node)

        # ITQ Rotation
        itq_node = helper.make_node(
            'Gemm',
            inputs=['pca_transformed', 'rotation_matrix'],
            outputs=['itq_transformed'],
            alpha=1.0,
            beta=0.0,
            name="ITQRotation"
        )
        nodes.append(itq_node)

        # Binarization
        sign_node = helper.make_node(
            'Greater',
            inputs=['itq_transformed', 'zero'],
            outputs=['hash_output'],
            name="Binarization"
        )
        nodes.append(sign_node)

        # Create initializers
        initializers = [
            helper.make_tensor('mean', onnx.TensorProto.FLOAT, [input_dim], scale_mean.astype(np.float32)),
            helper.make_tensor('scale', onnx.TensorProto.FLOAT, [input_dim], scale_var.astype(np.float32)),
            helper.make_tensor('pca_components', onnx.TensorProto.FLOAT, [input_dim, output_dim], pca_components.astype(np.float32).flatten()),
            helper.make_tensor('rotation_matrix', onnx.TensorProto.FLOAT, [output_dim, output_dim], rotation_matrix.astype(np.float32).flatten()),
            helper.make_tensor('zero', onnx.TensorProto.FLOAT, [1], np.array([0], dtype=np.float32))
        ]
        graph = helper.make_graph(
            nodes,
            'PCA_ITQ_Graph',
            [input_tensor],
            [output_tensor],
            initializer=initializers
        )
        model = helper.make_model(graph)
        model.ir_version = 7
        model.opset_import[0].version = opset_version
        return model
