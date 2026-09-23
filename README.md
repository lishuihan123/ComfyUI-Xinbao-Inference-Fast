# ComfyUI Xinbao Inference Fast

`ComfyUI-Xinbao-Inference-Fast` 是独立发布的“心宝❤推理（极速版）”ComfyUI 节点。它不再与心宝通用节点组捆绑，安装后只注册一个可见节点。

## 心宝❤推理（极速版）

本地调用 **Ternary Bonsai 2 27B**，把文字要求、参考图片或视频扩写/反推为可用于图像和视频生成的提示词。

- 可以完全不输入图片，只根据“角色定位”和“用户指令”扩写提示词；
- 支持 1–10 张独立参考图片；
- 支持输入视频帧批次，并按设定数量均匀抽帧；
- 用户指令优先于角色定位，冲突时以用户指令为准；
- 支持中文或英文输出；
- 可控制最大图片边长、最大输出 token、温度、top_p、种子和上下文大小；
- 可选择推理后释放模型，或保持模型常驻以便连续运行。

## 安装节点

```powershell
cd ComfyUI\custom_nodes
git clone https://github.com/lishuihan123/ComfyUI-Xinbao-Inference-Fast.git
cd ComfyUI-Xinbao-Inference-Fast
powershell -ExecutionPolicy Bypass -File .\scripts\install_bonsai_windows.ps1
```

安装脚本会断点续传模型分卷，合并并校验主模型，同时从 Prism ML 官方 Release 安装匹配的 Windows CUDA 12.4 运行库。安装完成后重启 ComfyUI。

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
         └─ runtime/
            ├─ llama-server.exe
            └─ 其余 DLL 与运行文件
```

也可以通过环境变量 `XINBAO_BONSAI_RUNTIME` 指定其他运行库目录。

## 相关项目

通用节点“心宝❤图片标准化”和“心宝❤构图”位于长期维护的 [`ComfyUI-Xinbao-Node-Group`](https://github.com/lishuihan123/ComfyUI-Xinbao-Node-Group)。两个仓库可以同时安装，不会重复注册节点。

## 模型来源与授权

- Bonsai 2 27B GGUF：[`prism-ml/Ternary-Bonsai-2-27B-gguf`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)，Apache-2.0；
- 本项目重新分发的模型文件未作修改，模型版权归原作者 Prism ML；
- Windows 推理运行库来自 Prism ML 官方 llama.cpp fork Release，MIT License。

完整说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
