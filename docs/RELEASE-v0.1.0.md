<div align="center">

<img src="https://img.shields.io/badge/GreatNN-v0.1.0-2563eb?style=for-the-badge" alt="GreatNN v0.1.0" />
<img src="https://img.shields.io/badge/Windows-x64-2563eb?style=for-the-badge" alt="Windows x64" />
<img src="https://img.shields.io/badge/PyTorch-local-c2410c?style=for-the-badge&logo=pytorch&logoColor=white" alt="Local PyTorch runtime" />
<img src="https://img.shields.io/badge/License-MIT-147d64?style=for-the-badge" alt="MIT license" />

<br /><br />

<img src="../../raw/v0.1.0/docs/greatnn-cat.gif" width="108" height="108" alt="GreatNN" />

<h1>GreatNN</h1>
<h2><em>PyTorch 可视化工作区</em></h2>

<p>在画布上搭建网络，用真实样本核对张量形状，再用本地 Python 环境执行前向计算或训练。</p>

<table>
<tr>
<td align="center"><strong>网络画布</strong><br />命名端口和自建模块库</td>
<td align="center"><strong>模型检查</strong><br />形状、参数和 PyTorch 实现</td>
<td align="center"><strong>本地推理</strong><br />执行画布或向模型提问</td>
<td align="center"><strong>训练与微调</strong><br />自定义 LOSS 和 LoRA 适配器</td>
</tr>
</table>

<p><a href="#简体中文">简体中文</a> · <a href="#english">English</a></p>

</div>

---

## 简体中文

<div align="center">
  <img src="../../raw/v0.1.0/docs/workbench-zh.png" width="100%" alt="GreatNN Windows 中文界面，画布显示卷积网络，右侧显示端口检查器" />
</div>

### 使用场景

| 场景 | 从哪里开始 |
| --- | --- |
| 看懂一张网络图 | 载入 ResNet-18 或 VGG-16，顺着连线检查每一层的输入和输出形状。 |
| 试一个新的模块 | 导入 Python 类，或直接修改画布上的模块，先用一个验证批次检查结果。 |
| 微调本地模型 | 载入兼容的因果语言模型，选择 LoRA 目标层，训练后保存适配器。 |

### 快速开始

#### Windows

1. 在 GitHub Release 的 **Assets** 中下载 `GreatNN-0.1.0-win-x64.zip`，完整解压后运行 `GreatNN.exe`。
2. 点击「载入样例」，从「模型预设」选一个网络，或在空画布上手动添加输入和模块。
3. 要运行或训练时，点击右下角的 Python 状态栏，选择装有 PyTorch 的解释器。打开输入节点的「输入设置」，填写或导入脚本，点击「保存并检测」。
4. 打开「运行网络」查看输出，或打开「训练网络」设置 LOSS 和优化器。

便携包内含桌面程序。请按自己的设备安装 [PyTorch](https://pytorch.org/get-started/locally/)。要启用基于本地包的 Python 补全，请在所选解释器中安装 `jedi`。使用语言模型或图像、语音编码器时，再安装对应依赖：

```bash
python -m pip install jedi "transformers>=4.51,<5" "peft>=0.17,<0.19" "accelerate>=1,<2" Pillow
```

#### 从源码运行

使用 Node.js 22.12 或更新版本，在解压后的源码目录中执行：

```bash
npm ci
npm run dev
```

| 命令 | 用途 |
| --- | --- |
| `npm run dev` | 启动桌面开发版。 |
| `npm run build` | 检查 TypeScript 并构建应用资源。 |
| `npm run pack:win` | 打包 Windows 安装程序。 |
| `npm run lint` | 运行 ESLint。 |

### 功能

#### 网络画布与模型检查

- 拖动模块、连接端口和调整节点尺寸。预设可以横向或纵向排列，节点较多时用「适应画布」查看全图。
- 端口可以命名。生成代码使用画布里的名称，未命名的输入默认沿用上游变量名。
- 检查器显示推导出的形状和参数量。运行后可查看选中模块的实际参数名、形状和部分权重值。
- 底部「问题」面板列出未连接端口、形状不匹配等问题。右下角显示当前 Python 解释器，点击即可更换。

#### 内置模块与实现代码

| 类别 | 模块示例 |
| --- | --- |
| 输入输出 | `Input`、`Output` |
| 卷积和残差 | `Conv2d`、`ResidualBlock2d`、`ResNetBasicBlock`、`ResNetBottleneck` |
| 归一化、激活和池化 | `BatchNorm1d`、`BatchNorm2d`、`LayerNorm`、`ReLU`、`GELU`、`MaxPool2d`、`AdaptiveAvgPool2d` |
| 序列和注意力 | `Embedding`、`LSTM`、`GRU`、`SelfAttention`、`TransformerEncoder`、`TransformerDecoder` |
| 投影和张量操作 | `Linear`、`Flatten`、`Dropout`、`Add`、`Concat`、`PatchEmbedding`、`TokenPool` |
| 文本和媒体 | `Tokenizer`、`CausalLM`、`VisionEncoder`、`AudioEncoder` |
| 自建模块 | 自定义块、从 Python 文件导入的类，以及固化到自建库的模块 |

「查看实现代码」会展示所选模块类或内置层的 PyTorch 调用。「编辑实现」支持 Python 语法检查和补全。自定义块可以一起编辑 `__init__` 和 `forward`，也能保存到自建库，在另一张画布上复用。

#### 输入脚本与形状检测

只跑推理时，可以写 `build_inputs()`。需要训练时，写 `build_datasets()`，返回 `train`、`validation`、`test` 三个数据集，其中每项为 `(输入, 目标)`。多路输入使用字典时，键名和画布上的输入变量名一致。

例如画布连成「输入 → Linear → 输出」，把 `Linear` 的输出特征设为 1，下面的脚本会提供 4 个输入特征和 1 个回归目标：

```python
import torch
from torch.utils.data import TensorDataset


def build_datasets():
    rng = torch.Generator().manual_seed(42)
    features = torch.randn(120, 4, generator=rng)
    target = features.sum(dim=1, keepdim=True)
    return {
        "train": TensorDataset(features[:80], target[:80]),
        "validation": TensorDataset(features[80:100], target[80:100]),
        "test": TensorDataset(features[100:], target[100:]),
    }
```

点击「保存并检测」后，程序会用选中的 Python 环境读取样本。这个例子中，输入形状会同步为 `float32 [B, 4]`，下游 Linear 的输入特征也会改为 4。「运行画布」优先使用 `build_inputs()`，否则读取验证集的第一批数据。

#### 运行、训练与 LoRA 微调

| 模式 | 输入与输出 |
| --- | --- |
| **运行画布** | 使用输入节点里的脚本，展示网络输出。接入因果语言模型时，可以查看解码后的文字，并展开张量详情。 |
| **大模型问答** | 选择本地模型目录，直接输入问题。生成前可以加载训练过的 LoRA 适配器。 |
| **训练网络** | 训练集用于更新参数，验证集用于绘制 LOSS 曲线，训练结束后计算测试损失。可选设备、优化器、训练轮次、批大小和学习率。 |

冻结模块仍参与前向计算。自定义 LOSS 可以接收 `prediction`、`target`、`tensors` 和 `inputs`，例如通过 `tensors["logits"]` 引用名为 `logits` 的中间张量。关闭训练窗口后任务仍可继续，也可以重新打开窗口停止。

「Causal LM · LoRA」预设连接 token ID、注意力遮罩和本地模型。目标层可选自动匹配、Q/V、全部注意力投影、MLP 投影或全部线性层，秩和缩放也可以下拉设置。模型目录需要相容的权重和分词器。当前版本用小型 Qwen2 验证了微调与保存，尚未实测 7B 的资源占用和训练速度。

#### 导入与导出

| 格式 | 用途 |
| --- | --- |
| Graph JSON | 保存并重新打开可编辑的画布，保留端口名和脚本。 |
| PyTorch | 预览或复制生成的 `nn.Module`，也可从 Python 文件选一个类导入。导入后的类可能仍需原项目依赖。 |
| ONNX | 导入 `.onnx` 或 `.pb` 文件中的受支持算子，用于查看结构。 |

图上可使用分词器、图像编码器和语音编码器。图像与语音块输出特征，供下游模块继续处理。语音编码器读取采样率与处理器匹配的 16 位 PCM WAV。

### 界面布局

| 区域 | 内容 |
| --- | --- |
| 顶部工具栏 | 样例、预设、布局、导入导出、PyTorch 预览、运行、训练与语言切换。 |
| 左侧 | 搜索内置模块，或从自建库添加模块。 |
| 中间画布 | 接线、拖动、缩放和删除节点，或适应整张画布。 |
| 右侧检查器 | 端口名、连接来源、张量形状、实现代码和训练模式。 |
| 底部 | 「工作区」「问题」「终端」切换。蓝色状态栏显示图统计和当前 Python 环境。 |

### 模型预设

| 预设 | 内容 |
| --- | --- |
| Simple CNN | 只有一个输入节点的起始画布。 |
| ResNet-18 / ResNet-50 | 使用基础残差块或瓶颈块的网络布局。 |
| VGG-16 | 卷积、池化与分类头。 |
| LSTM Classifier | 用循环层做序列分类的示例。 |
| Transformer Seq2Seq | 独立输入的编码器与解码器。 |
| DINOv3-style ViT | 带 Patch Embedding 和 Token Pooling 的视觉 Transformer 布局。 |
| Causal LM · LoRA | 为 token ID、注意力遮罩和本地模型预接好端口。 |

预设提供网络结构。便携包不附带预训练权重和数据集。

### 项目结构

```text
GreatNN/
├── electron/           桌面菜单、Python 补全、推理与训练
├── public/             小黑猫和 Windows 图标
├── src/
│   ├── core/            图结构、导入、形状推导和 PyTorch 导出
│   ├── editor/          画布、检查器、代码编辑器、运行和训练面板
│   └── examples/        模型预设
└── docs/                截图和发布说明
```

### 后续工作

接下来要验证的是更大模型的显存与内存占用、更多 ONNX 算子的导入，以及各家多模态模型的输入格式。当前版本可以运行本地因果语言模型，也可以把独立的图像或语音编码器接入画布。

### 技术栈

桌面端使用 Electron，画布由 React 和 React Flow 构建。Python 编辑器使用 CodeMirror 和 Jedi，运行及训练由所选解释器中的 PyTorch 执行，语言模型和 LoRA 分别使用 Transformers 和 PEFT。

### 许可证

GreatNN 基于 [NasNeo 的 NNMind](https://github.com/NasNeo777/NNMind) 修改。源码与便携包中的 `LICENSE` 保留了 MIT 许可全文及原版权声明：Copyright (c) 2026 NasNeo。

---

## English

<div align="center">
  <img src="../../raw/v0.1.0/docs/workbench-en.png" width="100%" alt="GreatNN on Windows with a Conv2d network and the port inspector" />
</div>

### Why GreatNN?

| Use case | Start here |
| --- | --- |
| Trace a network | Load ResNet-18 or VGG-16 and follow the shape at each port. |
| Try a new block | Import a Python class or edit a layer on the canvas, then run a validation batch to check its output. |
| Adapt a local model | Load a compatible causal language model, choose which projections to adapt, and save the resulting LoRA adapter. |

### Quick Start

#### Windows

1. Under **Assets** on the GitHub release, download `GreatNN-0.1.0-win-x64.zip`. Extract the full folder and open `GreatNN.exe`.
2. Click **Load Sample**, choose a model preset, or add an **Input** node to the blank canvas.
3. For local execution, click the Python indicator at the bottom right and choose an interpreter with PyTorch. Open the Input node’s **Input Settings**, add your script, and click **Save and Detect**.
4. Open **Run Network** to inspect the output, or **Train Network** to choose a loss and optimizer.

The portable ZIP contains the desktop app. Install a [compatible build of PyTorch](https://pytorch.org/get-started/locally/) in the interpreter you intend to use. For package-aware Python completion, install `jedi` in that environment. For local language models and media encoders, install the packages you need:

```bash
python -m pip install jedi "transformers>=4.51,<5" "peft>=0.17,<0.19" "accelerate>=1,<2" Pillow
```

#### From source

Use Node.js 22.12 or later. From the extracted source directory:

```bash
npm ci
npm run dev
```

| Command | What it does |
| --- | --- |
| `npm run dev` | Opens the desktop workspace in development mode. |
| `npm run build` | Type-checks the project and builds the app assets. |
| `npm run pack:win` | Builds a Windows installer. |
| `npm run lint` | Runs ESLint. |

### Features

#### Graph canvas and model inspection

- Drag and connect layers, resize blocks, and rearrange the graph horizontally or vertically. Use **Fit Canvas** for a large preset.
- Give ports names before wiring the graph. Generated code uses those names, and an unnamed input follows its upstream variable by default.
- The inspector shows inferred shapes and parameter counts. After a run, it can show a selected module’s actual parameter shapes and a small sample of its weights.
- The **Problems** tab lists disconnected ports, shape mismatches, and other graph issues. The status bar shows the current Python interpreter.

#### Modules and code

| Category | Examples |
| --- | --- |
| Inputs and outputs | `Input`, `Output` |
| Convolution and residual blocks | `Conv2d`, `ResidualBlock2d`, `ResNetBasicBlock`, `ResNetBottleneck` |
| Normalization, activation and pooling | `BatchNorm1d`, `BatchNorm2d`, `LayerNorm`, `ReLU`, `GELU`, `MaxPool2d`, `AdaptiveAvgPool2d` |
| Sequences and attention | `Embedding`, `LSTM`, `GRU`, `SelfAttention`, `TransformerEncoder`, `TransformerDecoder` |
| Projections and graph operations | `Linear`, `Flatten`, `Dropout`, `Add`, `Concat`, `PatchEmbedding`, `TokenPool` |
| Text and media | `Tokenizer`, `CausalLM`, `VisionEncoder`, `AudioEncoder` |
| User-defined blocks | `CustomBlock`, imported Python classes, and blocks saved to the personal library |

**View Implementation** shows the selected class or the PyTorch calls used by a built-in layer. **Edit Implementation** opens a Python editor with syntax checks and completion. For a custom block, edit `__init__` and `forward` together, then save the block to reuse it on another graph.

#### Input scripts and shape detection

Define `build_inputs()` for inference-only input, or `build_datasets()` for training. A dataset returns `train`, `validation`, and `test`, and each item is an `(inputs, target)` pair. For a named input dictionary, use the names on the canvas.

For example, connect `Input → Linear → Output` and set `Linear` to one output feature. This script provides four input features and a single regression target:

```python
import torch
from torch.utils.data import TensorDataset


def build_datasets():
    rng = torch.Generator().manual_seed(42)
    features = torch.randn(120, 4, generator=rng)
    target = features.sum(dim=1, keepdim=True)
    return {
        "train": TensorDataset(features[:80], target[:80]),
        "validation": TensorDataset(features[80:100], target[80:100]),
        "test": TensorDataset(features[100:], target[100:]),
    }
```

**Save and Detect** reads a sample through the selected Python interpreter. In this example the input becomes `float32 [B, 4]` and the downstream Linear layer uses four input features. **Run Canvas** uses `build_inputs()` when present and otherwise uses the first validation batch.

#### Running, training and LoRA

| Mode | Inputs and results |
| --- | --- |
| **Run Canvas** | Uses the Input node’s script and displays graph output. For a causal language model, it can decode generated tokens to text and show tensor details separately. |
| **Ask a Model** | Takes a text prompt and a local model folder. You can load a LoRA adapter before generating a reply. |
| **Train Network** | Uses the training split for updates, the validation split for the loss curve, and the test split for final loss. Set the device, optimizer, epochs, batch size, and learning rate here. |

Frozen modules still participate in the forward pass. Custom loss functions can take `prediction`, `target`, `tensors`, and `inputs`; use a name from the graph, such as `tensors["logits"]`, to read an intermediate value. Training can continue with the dialog closed and can be stopped from the training window.

The **Causal LM · LoRA** preset connects token IDs and an attention mask to a local model. Choose automatic target matching, Q/V, all four attention projections, MLP projections, or all linear layers. Rank and alpha are available in the same inspector. Local model folders need compatible weights and tokenizer files. The release was smoke-tested with a small Qwen2 configuration. There are no 7B performance figures for this release.

#### Import and export

| Format | Use |
| --- | --- |
| Graph JSON | Save and reopen an editable graph, including named ports and scripts. |
| PyTorch | Preview and copy the generated `nn.Module`, or import a class from a Python source file. Imported classes can still depend on their original project. |
| ONNX | Import supported operators from `.onnx` or `.pb` files for inspection. |

The graph includes a tokenizer and separate vision and audio encoder blocks. Vision and audio blocks emit features for downstream nodes. For the audio encoder, use a 16-bit PCM WAV at the processor’s expected sample rate.

### UI Layout

| Region | Controls |
| --- | --- |
| Top toolbar | Sample, presets, layout, import/export, PyTorch preview, run, train, and language. |
| Left sidebar | Search built-in layers or open the personal block library. |
| Canvas | Connect, move, resize, and delete nodes. Zoom or fit the entire graph. |
| Right inspector | Named ports, upstream connections, shapes, implementation, and training mode. |
| Bottom panel | Workspace, Problems, and Terminal. The blue status bar shows graph totals and the active Python environment. |

### Built-in Model Presets

| Preset | What it contains |
| --- | --- |
| Simple CNN | Starter graph with a single Input node. |
| ResNet-18 / ResNet-50 | Residual network layouts with basic blocks or bottlenecks. |
| VGG-16 | A stack of convolution and pooling layers with a classifier head. |
| LSTM Classifier | An example of sequence classification with a recurrent layer. |
| Transformer Seq2Seq | Encoder and decoder paths with separate inputs. |
| DINOv3-style ViT | A vision-transformer layout with patch embedding and token pooling. |
| Causal LM · LoRA | A graph wired for token IDs, attention mask, and a local causal model. |

Presets describe network structure. The ZIP contains no pretrained weights or datasets.

### Project Structure

```text
GreatNN/
├── electron/           Desktop menus, Python completion, inference and training
├── public/             Mascot and Windows icons
├── src/
│   ├── core/            Graph types, import, shape inference and PyTorch export
│   ├── editor/          Canvas, inspector, code editor, run and training panels
│   └── examples/        Model presets
└── docs/                Screenshots and release notes
```

### Roadmap

The next items to investigate are larger-model memory use, coverage of more ONNX operators, and model-specific multimodal inputs. Current support covers local causal language models and separate media encoders that can be wired into a graph.

### Tech Stack

Electron hosts the desktop app, React and React Flow provide the editor, and CodeMirror and Jedi handle Python editing and completion. Local execution uses the selected Python interpreter with PyTorch. Transformers and PEFT are used for language models and LoRA.

### License

GreatNN is based on [NNMind by NasNeo](https://github.com/NasNeo777/NNMind). The source and portable package keep the original copyright notice and MIT license in `LICENSE`. Copyright (c) 2026 NasNeo.
