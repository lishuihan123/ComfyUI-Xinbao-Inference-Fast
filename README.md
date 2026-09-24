# ComfyUI Xinbao Inference Fast

`ComfyUI-Xinbao-Inference-Fast` 是独立发布的“心宝❤推理（极速版）”ComfyUI 节点。它不再与心宝通用节点组捆绑，安装后只注册一个可见节点。

## 心宝❤推理（极速版）

节点提供两套互不混用的本地推理能力：**Ternary Bonsai 2 27B** 负责通用文字扩写、图片与视频反推；Qwen 官方专项微调的 **Qwen Image 2.1 PE-T2I / PE-I2I** 负责文生图提示词扩写和图片编辑指令改写。

- 可以完全不输入图片，只根据“角色定位”和“用户指令”扩写提示词；
- 支持 1–10 张独立参考图片；
- 支持输入视频帧批次，并按设定数量均匀抽帧；
- 用户指令优先于角色定位，冲突时以用户指令为准；
- 支持中文或英文输出；
- 可控制最大图片边长、最大输出 token、温度、top_p、种子和上下文大小；
- 可选择推理后释放模型，或保持模型常驻以便连续运行。
- `Qwen PE 自动`：没有图片时使用 PE-T2I，连接图片时使用 PE-I2I；连接视频时自动保留 Bonsai 通用反推；
- PE 模式使用与专项权重配套的官方系统提示词，可指定画幅和透明 RGBA 背景。

“角色定位”只控制 Bonsai 通用模式。Qwen PE 模式固定使用与专项权重配套的官方角色和输出协议，避免自由角色提示破坏微调模型的行为。
选择任意 Qwen PE 模式时，节点会自动隐藏“角色定位”输入框；切回 Bonsai 通用模式后自动恢复，避免误解。
PE 模式固定采用官方生产参数，包括 T2I `16256`、I2I `24000` 的生成 token 预算；界面里的“最大输出 token 数”仅控制 Bonsai 通用模式。

## 安装节点

```powershell
cd ComfyUI\custom_nodes
git clone https://github.com/lishuihan123/ComfyUI-Xinbao-Inference-Fast.git
cd ComfyUI-Xinbao-Inference-Fast
powershell -ExecutionPolicy Bypass -File .\scripts\install_bonsai_windows.ps1
```

安装脚本会断点续传模型分卷，合并并校验主模型，同时从 Prism ML 官方 Release 安装匹配的 Windows CUDA 12.4 运行库。安装完成后重启 ComfyUI。

如需 Qwen Image 2.1 PE 模式，再运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_qwen_pe_windows.ps1
```

该脚本只在用户主动运行时联网，下载并校验官方 PE 权重的 Q4_K_M 转换版、I2I 视觉组件和固定版本的官方系统提示词。若 Bonsai 已有 llama.cpp 运行库会直接共用，不再重复占用约 1 GB；只有缺少共用运行库时才会下载兼容版本。

## 模型目录

模型和运行库与节点源码分离，统一安装到：

```text
ComfyUI/
├─ custom_nodes/
│  └─ ComfyUI-Xinbao-Inference-Fast/
└─ models/
   └─ LLM/
      └─ Bonsai2-27B/
         ├─ Ternary-Bonsai-2-27B-PTQ1_0.gguf
         ├─ Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf
         └─ runtime/（仅在无法共用 Bonsai 运行库时创建）
      └─ QwenImage2.1-PE/
         ├─ Qwen-Image-2.1-PE-T2I.Q4_K_M.gguf
         ├─ Qwen-Image-2.1-PE-I2I.Q4_K_M.gguf
         ├─ Qwen-Image-2.1-PE-I2I.mmproj-bf16.gguf
         ├─ system_prompt_t2i.txt
         ├─ system_prompt_edit.txt
         └─ runtime/
            ├─ llama-server.exe
            └─ 其余 DLL 与运行文件
```

也可以通过环境变量 `XINBAO_BONSAI_RUNTIME` 指定其他运行库目录。Qwen PE 默认共用该运行库，也可通过 `XINBAO_PE_RUNTIME` 单独指定。

## 相关项目

通用节点“心宝❤图片标准化”和“心宝❤构图”位于长期维护的 [`ComfyUI-Xinbao-Node-Group`](https://github.com/lishuihan123/ComfyUI-Xinbao-Node-Group)。两个仓库可以同时安装，不会重复注册节点。

## 模型来源与授权

- Bonsai 2 27B GGUF：[`prism-ml/Ternary-Bonsai-2-27B-gguf`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)，Apache-2.0；
- 本项目重新分发的模型文件未作修改，模型版权归原作者 Prism ML；
- Windows 推理运行库来自 Prism ML 官方 llama.cpp fork Release，MIT License。
- Qwen Image 2.1 PE-T2I / PE-I2I 原始权重及系统提示词来自 Qwen，适用 Qwen Research License；
- GGUF 文件由第三方对官方 PE 权重进行量化转换，安装脚本固定来源版本并校验 SHA256；
- Qwen PE 只负责改写提示词，不包含 Qwen Image 2.1 扩散出图模型。

完整说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
