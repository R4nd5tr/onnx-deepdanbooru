# ONNX DeepDanbooru — Plugin-Based Automatic Image Tagging Module

Language: [中文](README.md) | [English](README-en.md)

Based on ONNX Runtime, deploys the [DeepDanbooru](https://github.com/KichangKim/DeepDanbooru) model in C++ for my [Waifu Gallery](https://github.com/R4nd5tr/waifu_gallery) illustration management application project, implementing automatic image tagging, image deduplication, and similarity search functionality.

## Features

- Input an image, output a tag set, rating level, and 512-bit feature hash
- Supports GPU (DirectML) inference, automatically falls back to CPU when unavailable
- Interface and implementation are separated, allowing the model/inference backend to be replaced

## Implementation Details

### Model Conversion, Modifying the Computation Graph

Original TensorFlow DeepDanbooru model → converted to ONNX format → inject feature output → merge hash subgraph:

1. **TF → ONNX**: Convert the base model, preserving the 9176-dimensional tag output
2. **Inject feature vector output**: Manually modify the computation graph, adding `GlobalAveragePool` + `Squeeze` nodes after the last ReLU, outputting a 4096-dimensional feature vector
3. **Train PCA-ITQ**: Collect image feature vectors, train the standardization + PCA + ITQ rotation matrices
4. **PCA-ITQ → ONNX**: Export the PCA-ITQ hashing pipeline as an ONNX subgraph
5. **Merge models**: Merge the PCA-ITQ subgraph with the feature output nodes to obtain a model that outputs both tags and feature hashes in a single inference pass

### C++ Deployment

1. **Implement preprocessing**: The original Python preprocessing code cannot be used on the C++ side; rewrite the image preprocessing logic using OpenCV to ensure consistency with the original model input
2. **C++ testing**: The C++ deployment model inference results are consistent with the Python side
3. **Postprocessing**: Implement postprocessing logic on the C++ side, including tag probability threshold filtering, rating level determination, and feature hash output
4. **DLL pluginization**: Compile the model inference module into a DLL, decoupling the model from the main program for easy replacement of the model or inference backend

### Plugin-Based Deployment Architecture

``` mermaid
flowchart TD
    A["Main Program (Waifu Gallery)<br/>AutoTaggerLoader<br/>Manages DLL lifecycle"]
    C["AutoTagger (pure virtual interface, no implementation)<br/>preprocess / predict / postprocess<br/>getTagSet / getModelName"]
    subgraph D["DefaltAutoTagger"]
        E["OpenCV preprocessing"]
        subgraph J["ONNX Runtime inference"]
            F["Image classification model"]
        end
        I["Postprocessing (output tags + rating level + feature hash)"]

        E -->|"Preprocessed image"| F
        F -->|"Image tag probability output"| I
        F -->|"PCA-ITQ feature hash output"| I
    end

    A -->|"Load dynamic library at runtime<br/>createAutoTagger()"| C
    C -->|"Model-specific inference logic implementation"| D
```

### Plugin Interface Design

- `AutoTagger` is a pure virtual base class, defining a three-stage interface of preprocessing / inference / postprocessing
- `extern "C"` exports the `createAutoTagger()` / `destroyAutoTagger()` factory functions
- Logs are injected via the `LogCallback` callback; the module does not depend on the main program's logging system
- `preprocess` and `predict` are separated, making it easy for the main program to implement a pipeline of multi-worker-thread preprocessing + single-thread inference

### Extension

To add a new model or inference backend, simply:
1. Inherit `AutoTagger` and implement the inference logic according to the specific model
2. Compile it into a DLL and place it together with the model file into `model/`
3. The main program dynamically loads the model and calls it at runtime

No modification of the main program is required.

## Development and Build

This project only implements the necessary functionality and standalone utility scripts; it does not have a complete process-based build and test system. Developers can extend it according to their own needs.

### Model Processing

The image tagging model must be downloaded separately from [DeepDanbooru](https://github.com/KichangKim/DeepDanbooru/releases).

#### Convert the Model

After installing the Python dependencies listed in the comments at the beginning of the conversion script `h5_to_onnx.py` in a Python virtual environment, run the script to convert the downloaded DeepDanbooru model to ONNX format

#### Modify the Model

In another Python environment, run the following command to install dependencies:

```powershell
pip install -r `requirements.txt`
```

The entry points for other functionality are located in `scripts.py`, which implements the necessary functions for modifying the model; call them as needed

### C++ Deployment

C++ deployment requires no Python environment, depending only on OpenCV and ONNX Runtime DirectML.

#### Environment Requirements

- Visual Studio 2022
- CMake 3.20 or higher
- C++17 compilation environment
- OpenCV 4.12
- ONNX Runtime DirectML 1.23.0

By default, the project loads third-party dependencies from the following directories:

```text
cpp_deploy/external/
├── include/
├── opencv/
└── microsoft.ml.onnxruntime.directml.1.23.0/
```

If the dependency directory locations are different, you need to modify `OPENCV_DIR`, `ONNXRUNTIME_DIR`, and `HEADER_LIB_DIR` in `CMakeLists.txt` accordingly.

#### Configure the Project

Run in the project root directory:

```powershell
cmake --preset msvc-debug
```

Configure the Release project:

```powershell
cmake --preset msvc-release
```

#### Build the Project

> [!IMPORTANT] 
> The DLL pluginized module and the main program interact through a C++ interface, and must use the same compiler and compilation options, otherwise runtime errors may occur.

Build the Debug version:

```powershell
cmake --build --preset msvc-debug-build
```

Build the Release version:

```powershell
cmake --build --preset msvc-release-build
```

The build artifacts are located at:

```text
cpp_deploy/bin/msvc/
├── Debug/
│   ├── autotagger_defalt.dll
└── Release/
    └── autotagger_defalt.dll
```

#### DLL Runtime Dependencies

When running `autotagger_defalt.dll`, ensure the following files are located in the same directory as the DLL, or have been added to the system `PATH`:

```text
onnxruntime.dll
onnxruntime_providers_shared.dll
opencv_world4120.dll       # Release
opencv_world4120d.dll      # Debug
```

DirectML inference also requires a usable DirectML/GPU environment on the system. If the GPU is unavailable, the module will automatically fall back to CPU inference.