from __future__ import annotations

import atexit
import base64
import io
import json
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import psutil
from PIL import Image

try:
    import folder_paths
except ImportError:  # allows lightweight import tests outside ComfyUI
    folder_paths = None


PLUGIN_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = (
    Path(folder_paths.models_dir) / "LLM" / "Bonsai2-27B"
    if folder_paths is not None
    else PLUGIN_DIR / "models"
)


def _resolve_runtime_dir() -> Path:
    """Locate llama.cpp outside the custom-node folder, with legacy fallback."""
    candidates = []
    configured = os.environ.get("XINBAO_BONSAI_RUNTIME")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        (
            DEFAULT_MODEL_DIR / "runtime",
            PLUGIN_DIR / "runtime",
        )
    )
    for candidate in candidates:
        if (candidate / "llama-server.exe").is_file():
            return candidate.resolve()
    # Point error messages at the preferred location when no runtime exists.
    return (DEFAULT_MODEL_DIR / "runtime").resolve()


RUNTIME_DIR = _resolve_runtime_dir()
SERVER_PORT = 8199

def _find_one(pattern: str) -> Path:
    found = sorted(DEFAULT_MODEL_DIR.glob(pattern))
    if not found:
        raise FileNotFoundError(
            f"未找到 {pattern}，请放到：{DEFAULT_MODEL_DIR}"
        )
    return found[0]


def _tensor_frame_to_pil(frame, max_side: int) -> Image.Image:
    """Convert a ComfyUI IMAGE frame [H,W,C] in 0..1 to resized RGB PIL."""
    if hasattr(frame, "detach"):
        frame = frame.detach().float().cpu().numpy()
    frame = np.asarray(frame)
    if frame.ndim == 4:
        frame = frame[0]
    if frame.ndim != 3:
        raise ValueError(f"图像维度应为 [H,W,C]，实际为 {frame.shape}")
    frame = np.clip(frame[..., :3], 0.0, 1.0)
    pil = Image.fromarray((frame * 255.0 + 0.5).astype(np.uint8), "RGB")
    if max(pil.size) > max_side:
        scale = max_side / max(pil.size)
        size = (max(1, round(pil.width * scale)), max(1, round(pil.height * scale)))
        pil = pil.resize(size, Image.Resampling.LANCZOS)
    return pil


def _as_frames(value) -> list:
    if value is None:
        return []
    if hasattr(value, "shape") and len(value.shape) == 4:
        return [value[i] for i in range(int(value.shape[0]))]
    return [value]


def _data_url(pil: Image.Image, quality: int) -> str:
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


class _ServerManager:
    def __init__(self):
        self.process: subprocess.Popen | None = None
        self.log_handle = None
        self.lock = threading.RLock()
        self.model_path: Path | None = None

    @staticmethod
    def _health(timeout=1.0) -> bool:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{SERVER_PORT}/health", timeout=timeout
            ) as response:
                return response.status == 200
        except Exception:
            return False

    def stop(self):
        with self.lock:
            proc = self.process
            self.process = None
            self.model_path = None
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
            # ComfyUI may have been force-restarted while llama-server was
            # running. In that case the child survives but this Python object
            # no longer owns its Popen handle. Only clean up a listener whose
            # executable is this node's bundled llama-server.
            bundled_server = (RUNTIME_DIR / "llama-server.exe").resolve()
            try:
                for connection in psutil.net_connections(kind="tcp"):
                    if (
                        connection.status != psutil.CONN_LISTEN
                        or not connection.laddr
                        or connection.laddr.port != SERVER_PORT
                        or not connection.pid
                    ):
                        continue
                    orphan = psutil.Process(connection.pid)
                    try:
                        executable = Path(orphan.exe()).resolve()
                    except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
                        continue
                    if str(executable).casefold() != str(bundled_server).casefold():
                        continue
                    orphan.terminate()
                    try:
                        orphan.wait(timeout=8)
                    except psutil.TimeoutExpired:
                        orphan.kill()
                        orphan.wait(timeout=5)
                    break
            except (psutil.AccessDenied, psutil.Error, OSError):
                pass
            if self.log_handle is not None:
                self.log_handle.close()
                self.log_handle = None

    def ensure(self, model: Path, mmproj: Path, context_size: int):
        with self.lock:
            if (
                self.process is not None
                and self.process.poll() is None
                and self.model_path == model
                and self._health()
            ):
                return
            self.stop()
            if self._health():
                raise RuntimeError(
                    f"端口 {SERVER_PORT} 已被其他程序占用，请关闭该程序后重试。"
                )

            binary = RUNTIME_DIR / "llama-server.exe"
            if not binary.exists():
                raise FileNotFoundError(f"缺少推理程序：{binary}")

            log_path = PLUGIN_DIR / "bonsai_server.log"
            self.log_handle = open(log_path, "w", encoding="utf-8", errors="replace")
            args = [
                str(binary),
                "-m", str(model),
                "--mmproj", str(mmproj),
                "--host", "127.0.0.1",
                "--port", str(SERVER_PORT),
                "-ngl", "99",
                "-fa", "on",
                "-c", str(context_size),
                "--jinja",
                "--reasoning", "off",
                "--reasoning-budget", "0",
                "--reasoning-format", "none",
                "--chat-template-kwargs", '{"enable_thinking":false}',
                "--image-max-tokens", "1024",
            ]
            env = os.environ.copy()
            env["PATH"] = str(RUNTIME_DIR) + os.pathsep + env.get("PATH", "")
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            self.process = subprocess.Popen(
                args,
                cwd=str(RUNTIME_DIR),
                env=env,
                stdout=self.log_handle,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )
            self.model_path = model

            deadline = time.time() + 240
            while time.time() < deadline:
                if self.process.poll() is not None:
                    self.log_handle.flush()
                    try:
                        tail = log_path.read_text(encoding="utf-8", errors="replace")[-5000:]
                    except Exception:
                        tail = ""
                    raise RuntimeError(f"Bonsai推理服务启动失败：\n{tail}")
                if self._health(timeout=2.0):
                    return
                time.sleep(1.0)
            self.stop()
            raise TimeoutError("Bonsai模型加载超过240秒，请查看 bonsai_server.log。")


SERVER = _ServerManager()
atexit.register(SERVER.stop)


def _release_comfy_models():
    try:
        import comfy.model_management as mm

        mm.unload_all_models()
        mm.soft_empty_cache()
    except Exception as exc:
        print(f"[Bonsai提示词] 释放ComfyUI显存时跳过：{exc}")


def _post_json(url: str, payload: dict, timeout: int) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Bonsai接口返回 HTTP {exc.code}：{detail}") from exc


def _clean_prompt(raw: str) -> str:
    """Remove reasoning/code fences and unwrap common JSON response formats."""
    prompt = (raw or "").strip()
    if "</think>" in prompt:
        prompt = prompt.split("</think>", 1)[1].strip()
    elif prompt.startswith("<think>"):
        prompt = prompt[len("<think>"):].strip()
    if prompt.startswith("```") and prompt.endswith("```"):
        lines = prompt.splitlines()
        if len(lines) >= 2:
            prompt = "\n".join(lines[1:-1]).strip()
    try:
        decoded = json.loads(prompt)
        if isinstance(decoded, dict):
            for key in ("rewritten_prompt", "prompt", "text", "content", "result"):
                value = decoded.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        elif isinstance(decoded, str):
            return decoded.strip()
    except (json.JSONDecodeError, TypeError):
        pass
    return prompt


def _language_matches(prompt: str, output_language: str) -> bool:
    cjk_count = len(re.findall(r"[\u3400-\u9fff]", prompt))
    latin_count = len(re.findall(r"[A-Za-z]", prompt))
    if output_language == "中文":
        return cjk_count >= 12 and cjk_count >= latin_count * 0.25
    return latin_count >= 20 and latin_count >= cjk_count * 2


class Bonsai2ReversePrompt:
    """Expand text-only instructions or reverse-engineer prompts from visual inputs."""

    @classmethod
    def INPUT_TYPES(cls):
        optional = {f"image_{i}": ("IMAGE",) for i in range(1, 11)}
        optional["video"] = ("IMAGE",)
        return {
            "required": {
                "role_positioning": (
                    "STRING",
                    {
                        "default": "你是专业的视觉分析师和AI图像、视频提示词工程师。",
                        "multiline": True,
                    },
                ),
                "user_instruction": (
                    "STRING",
                    {
                        "default": "将我的要求扩写成可直接用于生成模型的高质量提示词。如果连接了图片或视频，请综合分析素材并保留人物身份、服装、动作、环境、镜头、光线、构图和风格；如果没有连接素材，就完全依据文字要求进行创作和扩写。",
                        "multiline": True,
                    },
                ),
                "output_language": (["中文", "English"], {"default": "中文"}),
                "video_sample_frames": (
                    "INT", {"default": 8, "min": 1, "max": 32, "step": 1}
                ),
                "max_image_side": (
                    "INT", {"default": 1024, "min": 384, "max": 2048, "step": 64}
                ),
                "max_output_tokens": (
                    "INT", {"default": 1024, "min": 128, "max": 4096, "step": 64}
                ),
                "temperature": (
                    "FLOAT", {"default": 0.3, "min": 0.0, "max": 1.5, "step": 0.05}
                ),
                "top_p": (
                    "FLOAT", {"default": 0.85, "min": 0.05, "max": 1.0, "step": 0.01}
                ),
                "seed": (
                    "INT",
                    {
                        "default": 8888,
                        "min": 0,
                        "max": 0x7FFFFFFF,
                        "control_after_generate": True,
                    },
                ),
                "context_size": (
                    "INT", {"default": 32768, "min": 8192, "max": 131072, "step": 8192}
                ),
                "release_comfy_vram": ("BOOLEAN", {"default": True}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    OUTPUT_TOOLTIPS = ("可直接连接到正向提示词输入的反推结果",)
    FUNCTION = "generate"
    OUTPUT_NODE = True
    CATEGORY = "心宝❤推理（极速版）"
    DESCRIPTION = "使用Ternary Bonsai 2 27B进行纯文字提示词扩写，或从1-10张图片/视频抽帧反推提示词。"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return kwargs.get("seed", 8888)

    def generate(
        self,
        role_positioning,
        user_instruction,
        output_language,
        video_sample_frames,
        max_image_side,
        max_output_tokens,
        temperature,
        top_p,
        seed,
        context_size,
        release_comfy_vram,
        keep_model_loaded,
        **kwargs,
    ):
        model = _find_one("*PTQ1_0.gguf")
        mmproj = _find_one("*mmproj-Q8_0.gguf")

        media: list[tuple[str, Image.Image]] = []
        for index in range(1, 11):
            frames = _as_frames(kwargs.get(f"image_{index}"))
            if frames:
                media.append(
                    (f"参考图片 {index}", _tensor_frame_to_pil(frames[0], max_image_side))
                )

        video_frames = _as_frames(kwargs.get("video"))
        if video_frames:
            take = min(video_sample_frames, len(video_frames))
            indices = np.linspace(0, len(video_frames) - 1, take, dtype=int).tolist()
            for sequence, frame_index in enumerate(indices, 1):
                media.append(
                    (
                        f"视频时间顺序帧 {sequence}/{take}",
                        _tensor_frame_to_pil(video_frames[frame_index], max_image_side),
                    )
                )

        if release_comfy_vram:
            _release_comfy_models()
        SERVER.ensure(model, mmproj, context_size)

        language_rule = (
            "最终结果必须以中文汉字为主，不得输出英文提示词。"
            if output_language == "中文"
            else "The final result must be written in English, not Chinese."
        )
        task_mode = (
            "当前提供了视觉素材，请综合全部素材进行视觉反推和提示词扩写。"
            if media
            else "当前没有视觉素材，请完全依据用户的文字要求进行创作和提示词扩写。"
        )
        system_text = (
            "你要严格按照以下优先级执行：\n"
            "1. 用户指令是本次任务的最高优先级要求。\n"
            "2. 角色定位只提供专业能力、知识和风格背景，不是本次的输出指令。"
            "当角色定位与用户指令冲突时，必须忽略角色定位中的冲突内容并执行用户指令。\n"
            "3. " + language_rule + "即使角色定位中提到其他语言，也必须忽略。\n\n"
            "【角色定位，仅作能力背景】\n"
            + role_positioning.strip()
            + "\n\n【当前任务模式】\n"
            + task_mode
            + "\n只输出一段纯文本最终提示词；不要分析过程、标题、Markdown、JSON、字段名或代码块，"
            "不要提到‘参考图’或‘视频帧’。"
            + ("必须忠实综合全部素材；看不清的内容不要编造。" if media else "应补足主体、动作、场景、构图、镜头、光线、色彩、材质、氛围和风格等有助于生成的细节，但不得偏离用户核心要求。")
        )
        content = []
        for label, pil in media:
            content.append({"type": "text", "text": label})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _data_url(pil, quality=90)},
                }
            )
        content.append(
            {
                "type": "text",
                "text": (
                    user_instruction.strip()
                    + (
                        "\n请把全部观察结果整理成一段可直接复制使用的最终提示词。"
                        if media
                        else "\n请扩写成一段可直接复制到图像或视频生成模型中使用的最终提示词。"
                    )
                ),
            }
        )

        payload = {
            "model": model.name,
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": content},
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_output_tokens,
            "seed": seed,
            "reasoning_effort": "none",
            "chat_template_kwargs": {"enable_thinking": False},
            "stream": False,
        }
        try:
            result = _post_json(
                f"http://127.0.0.1:{SERVER_PORT}/v1/chat/completions",
                payload,
                timeout=1200,
            )
            prompt = _clean_prompt(result["choices"][0]["message"]["content"])
            if prompt and not _language_matches(prompt, output_language):
                correction_rule = (
                    "将下面内容完整改写为自然、专业的中文生成提示词。"
                    "不得保留英文句子；只输出一段纯文本，不要JSON、标题或解释。"
                    if output_language == "中文"
                    else "Rewrite the following as a natural, professional English generation prompt. Output one plain-text paragraph only, with no JSON, title, or explanation."
                )
                correction_payload = {
                    **payload,
                    "messages": [
                        {"role": "system", "content": correction_rule},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": min(temperature, 0.2),
                }
                correction = _post_json(
                    f"http://127.0.0.1:{SERVER_PORT}/v1/chat/completions",
                    correction_payload,
                    timeout=1200,
                )
                prompt = _clean_prompt(correction["choices"][0]["message"]["content"])
            if not prompt:
                raise RuntimeError(f"模型返回了空结果：{result}")
            return {"ui": {"text": (prompt,)}, "result": (prompt,)}
        finally:
            if not keep_model_loaded:
                SERVER.stop()


NODE_CLASS_MAPPINGS = {
    "Bonsai2ReversePrompt": Bonsai2ReversePrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Bonsai2ReversePrompt": "心宝❤推理（极速版）",
}
