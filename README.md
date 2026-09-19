# ONNX DeepDanbooru — 插件式图像自动标注模块

基于 ONNX Runtime 在 C++ 中部署 [DeepDanbooru](https://github.com/KichangKim/DeepDanbooru) 模型，用于我的 [Waifu Gallery](https://github.com/R4nd5tr/waifu_gallery) 插画管理应用项目，实现图像自动标签标注，图片查重和相似度搜索功能。

## 功能

- 输入图片，输出标签集合、限制等级、512-bit 特征哈希
- 支持 GPU（DirectML）推理，不可用时自动回退 CPU
- 接口与实现分离，可替换模型/推理后端

## 实现细节

### 模型转换、修改计算图

原始 TensorFlow DeepDanbooru 模型 → 转换为 ONNX 格式 → 注入特征输出 → 合并哈希子图：

1. **TF → ONNX**：转换基础模型，保留 9176 维标签输出
2. **注入特征向量输出**：手动修改计算图，在最后一个 ReLU 后添加 `GlobalAveragePool` + `Squeeze` 节点，输出 4096 维特征向量
3. **训练 PCA-ITQ**：采集图片特征向量，训练标准化 + PCA + ITQ 旋转矩阵
4. **PCA-ITQ → ONNX**：把 PCA-ITQ 哈希流程导出为 ONNX 子图
5. **合并模型**：将 PCA-ITQ 子图与特征输出节点合并，得到单次推理同时输出标签和特征哈希的模型

### C++ 部署

1. **实现预处理**：在 C++ 端无法使用原始的 python 预处理代码，使用 OpenCV 重写图像预处理逻辑，确保与原始模型输入一致
2. **C++ 测试**：C++ 部署模型推理结果与 Python 端一致
3. **后处理**：在 C++ 端实现标签概率阈值过滤、限制等级判定、特征哈希输出等后处理逻辑
4. **DLL 插件化**：将模型推理模块编译为 DLL，使得模型与主程序解耦，便于替换模型或推理后端

### 插件化部署架构

``` mermaid
flowchart TD
    A["主程序 (Waifu Gallery)<br/>AutoTaggerLoader<br/>管理 DLL 生命周期"]
    C["AutoTagger (纯虚接口, 无实现)<br/>preprocess / predict / postprocess<br/>getTagSet / getModelName"]
    subgraph D["DefaltAutoTagger"]
        E["OpenCV 预处理"]
        subgraph J["ONNX Runtime 推理"]
            F["图片分类模型"]
        end
        I["后处理(输出标签 + 限制等级 + 特征哈希)"]

        E -->|"预处理图像"| F
        F -->|"图片标签概率输出"| I
        F -->|"PCA-ITQ 特征哈希输出"| I
    end

    A -->|"Load dynamic library at runtime<br/>createAutoTagger()"| C
    C -->|"Model-specific inference logic implementation"| D
```

### 插件接口设计

- `AutoTagger` 为纯虚基类，定义预处理 / 推理 / 后处理三段式接口
- `extern "C"` 导出 `createAutoTagger()` / `destroyAutoTagger()` 工厂函数
- 日志通过 `LogCallback` 回调注入，模块不依赖主程序日志系统
- `preprocess` 与 `predict` 分离，便于主程序做多工作线程预处理 + 单线程推理的流水线

### 扩展

新增模型或推理后端只需：
1. 继承 `AutoTagger` 根据具体模型实现推理逻辑
2. 编译为 DLL 和模型文件一起放入 `model/`
3. 主程序运行时动态加载模型并调用

无需修改主程序。

## 开发与构建

本项目仅实现了必要的功能和单独的工具脚本，没有完整的流程化构建和测试体系，开发者可根据自身需求自行扩展。

### 模型处理

图像标注模型须从 [DeepDanbooru](https://github.com/KichangKim/DeepDanbooru/releases) 自行下载。

#### 转换模型

在一个 python 虚拟环境中安装转换脚本 `h5_to_onnx.py` 开头的注释中列出的 Python 依赖后，执行脚本将下载的 DeepDanbooru 模型转换为 ONNX 格式

#### 修改模型

在另一个 Python 环境中，执行以下命令安装依赖：

```powershell
pip install -r `requirements.txt`
```

其他功能的脚本入口位于 `scripts.py`，实现了修改模型的必要功能，按需调用

### C++ 部署

C++ 部署无需 Python 环境，仅依赖 OpenCV 和 ONNX Runtime DirectML。

#### 环境要求

- Visual Studio 2022
- CMake 3.20 或更高版本
- C++17 编译环境
- OpenCV 4.12
- ONNX Runtime DirectML 1.23.0

项目默认从以下目录加载第三方依赖：

```text
cpp_deploy/external/
├── include/
├── opencv/
└── microsoft.ml.onnxruntime.directml.1.23.0/
```

如果依赖目录位置不同，需要同步修改 `CMakeLists.txt` 中的 `OPENCV_DIR`、`ONNXRUNTIME_DIR` 和 `HEADER_LIB_DIR`。

#### 配置项目

在项目根目录执行：

```powershell
cmake --preset msvc-debug
```

配置 Release 工程：

```powershell
cmake --preset msvc-release
```

#### 编译项目

> [!IMPORTANT] 
> DLL 插件化模块和主程序通过 C++ 接口进行交互，必须使用相同的编译器和编译选项，否则可能出现运行时错误。

编译 Debug 版本：

```powershell
cmake --build --preset msvc-debug-build
```

编译 Release 版本：

```powershell
cmake --build --preset msvc-release-build
```

编译产物位于：

```text
cpp_deploy/bin/msvc/
├── Debug/
│   ├── autotagger_defalt.dll
└── Release/
    └── autotagger_defalt.dll
```

#### DLL 运行时依赖

运行 `autotagger_defalt.dll` 时，需要确保以下文件位于 DLL 所在目录，或已加入系统 `PATH`：

```text
onnxruntime.dll
onnxruntime_providers_shared.dll
opencv_world4120.dll       # Release
opencv_world4120d.dll      # Debug
```

DirectML 推理还需要系统具备可用的 DirectML/GPU 环境。若 GPU 不可用，模块会自动回退到 CPU 推理。