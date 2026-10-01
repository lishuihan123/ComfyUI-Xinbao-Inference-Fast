from __future__ import annotations

import math
import random
from pathlib import Path

from .bonsai_nodes import (
    BONSAI_MODEL_DIR,
    BONSAI_RUNTIME_DIR,
    SERVER,
    _clean_prompt,
    _data_url,
    _find_one,
    _post_json,
    _release_comfy_models,
    _tensor_frame_to_pil,
)


H3_SKILL_ROOT = Path(__file__).resolve().parent / "h3_skills"
BASE_SKILL_ID = "h3-prompt-writing"
H3_MODES = {
    "文本生成视频 (T2VA)": "T2VA",
    "首帧生成视频 (I2VA)": "I2VA",
    "首尾帧生成视频 (FL2VA)": "FL2VA",
    "尾帧生成视频 (L2VA)": "L2VA",
    "全参考生成视频 (Ref2VA)": "Ref2VA",
}
ASPECT_RATIOS = {
    "1:1": (1, 1),
    "2:3": (2, 3),
    "3:2": (3, 2),
    "3:4": (3, 4),
    "4:3": (4, 3),
    "9:16": (9, 16),
    "16:9": (16, 9),
    "21:9": (21, 9),
}
H3_FPS = 24
H3_FRAME_REMAINDER = 5
H3_FRAME_INTERVAL = 17
H3_CANVAS_MULTIPLE = 32
DEFAULT_MEGAPIXELS = 0.4
SKILL_LANGUAGE = {
    "优先使用中文": "Chinese if available",
    "仅英文": "English",
    "仅中文": "Chinese",
    "中英文都使用": "Both",
}
REFERENCE_DEPTH = {
    "仅使用 Skill": "Skill only",
    "Skill 与参考资料": "Skill with references",
}
OUTPUT_FORMAT = {
    "仅输出 H3 提示词": "H3 prompt only",
    "输出 H3 提示词和简要说明": "H3 prompt with brief notes",
}
PROFESSIONAL_STORYBOARD_SKILL_ID = "professional-chinese-sales-storyboard"
H3_SYSTEM_BUDGET = 18000
BASE_SKILL_LIMIT = 2500
BASE_REFERENCE_LIMIT = 2500
SKILL_LIMIT = 7000
REFERENCE_LIMIT = 1500
SPECIALIZED_H3_SYSTEM_BUDGET = 7500
SPECIALIZED_BASE_SKILL_LIMIT = 1400
SPECIALIZED_BASE_REFERENCE_LIMIT = 1200
SPECIALIZED_SKILL_LIMIT = 3000
SPECIALIZED_REFERENCE_LIMIT = 400
SPECIALIZED_MAX_NEW_TOKENS = 768
MAX_CHAT_NEW_TOKENS = 4096
PROFESSIONAL_MIN_NEW_TOKENS = 2048
SKILL_NAMES = {
    "h3-prompt-writing": "H3 通用提示词编写",
    "3d-animation-short-generator": "3D 动画短片",
    "brand-promo-video-generator": "品牌宣传片",
    "co-op-game-intro-generator": "合作游戏开场",
    "handdrawn-live-video-generator": "手绘实拍视频",
    "minimalist-product-ad-generator": "极简产品广告",
    "mv-subtitle-skill-confirmed": "MV 字幕视频",
    "paper-collage-explainer-generator": "纸艺拼贴讲解",
    "papercraft-stop-motion-explainer": "纸艺定格动画",
    PROFESSIONAL_STORYBOARD_SKILL_ID: "中文带货视频",
}


def _skill_ids() -> list[str]:
    if not H3_SKILL_ROOT.exists():
        return [BASE_SKILL_ID]

    ids = [
        p.name
        for p in H3_SKILL_ROOT.iterdir()
        if p.is_dir() and (p / "SKILL.md").exists()
    ]
    ids.sort()
    if BASE_SKILL_ID in ids:
        ids.remove(BASE_SKILL_ID)
        ids.insert(0, BASE_SKILL_ID)
    return ids or [BASE_SKILL_ID]


def _skill_choices() -> list[str]:
    choices = [SKILL_NAMES.get(skill_id, skill_id) for skill_id in _skill_ids()]
    professional = SKILL_NAMES[PROFESSIONAL_STORYBOARD_SKILL_ID]
    if professional not in choices:
        choices.append(professional)
    return choices


def _normalize_choice(value: str, choices: dict[str, str]) -> str:
    return choices.get(value, value)


def _normalize_skill(value: str) -> str:
    for skill_id, label in SKILL_NAMES.items():
        if value == label:
            return skill_id
    return value


def _h3_frame_count(duration_seconds: float) -> int:
    frame_count = max(H3_FRAME_REMAINDER, round(duration_seconds * H3_FPS))
    return frame_count + (H3_FRAME_REMAINDER - frame_count % H3_FRAME_INTERVAL) % H3_FRAME_INTERVAL


def _h3_resolution(aspect_ratio: str, megapixels: float) -> tuple[int, int]:
    width_ratio, height_ratio = ASPECT_RATIOS[aspect_ratio]
    total_pixels = megapixels * 1024 * 1024
    scale = math.sqrt(total_pixels / (width_ratio * height_ratio))
    width = round(width_ratio * scale / H3_CANVAS_MULTIPLE) * H3_CANVAS_MULTIPLE
    height = round(height_ratio * scale / H3_CANVAS_MULTIPLE) * H3_CANVAS_MULTIPLE
    return width, height


def _normalize_megapixels(value) -> float:
    try:
        megapixels = float(value)
    except (TypeError, ValueError):
        return DEFAULT_MEGAPIXELS
    if not math.isfinite(megapixels):
        return DEFAULT_MEGAPIXELS
    return min(16.0, max(0.1, megapixels))


def _normalize_duration(value, fallback: float = 6.0) -> float:
    try:
        duration = float(value)
    except (TypeError, ValueError):
        duration = fallback
    if not math.isfinite(duration) or duration <= 0:
        duration = fallback
    return min(15.0, max(5.0, duration))


def _normalize_aspect_ratio(value: str) -> str:
    return value.split(" ", 1)[0]


def _aspect_ratio_from_dimensions(width: int, height: int) -> str:
    ratio = width / max(1, height)
    return min(
        ASPECT_RATIOS,
        key=lambda name: abs(ASPECT_RATIOS[name][0] / ASPECT_RATIOS[name][1] - ratio),
    )


def _read_text(path: Path, limit: int | None = None) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if limit is not None and len(text) > limit:
        return text[:limit].rstrip()
    return text


def _append_file(
    parts: list[str],
    path: Path,
    title: str,
    remaining_chars: list[int] | None = None,
    limit: int | None = None,
) -> None:
    if path.exists():
        if remaining_chars is not None and remaining_chars[0] <= 0:
            return
        text = _read_text(path, limit)
        if remaining_chars is not None and len(text) > remaining_chars[0]:
            text = text[:remaining_chars[0]].rstrip()
        if text:
            parts.append(f"## {title}\n\n{text}")
            if remaining_chars is not None:
                remaining_chars[0] -= len(text)


def _append_reference_files(
    parts: list[str],
    skill_dir: Path,
    skill_id: str,
    remaining_chars: list[int] | None = None,
    limit: int | None = None,
) -> None:
    ref_dir = skill_dir / "references"
    if not ref_dir.exists():
        return

    for path in sorted(ref_dir.glob("*.md")) + sorted(ref_dir.glob("*.txt")):
        _append_file(
            parts,
            path,
            f"{skill_id}/{path.relative_to(skill_dir).as_posix()}",
            remaining_chars,
            limit,
        )


def _append_skill_file(
    parts: list[str],
    skill_dir: Path,
    skill_id: str,
    skill_language: str,
    remaining_chars: list[int] | None = None,
    limit: int | None = None,
) -> None:
    cn_path = skill_dir / "SKILL.cn.md"
    en_path = skill_dir / "SKILL.md"

    if skill_language == "English":
        _append_file(parts, en_path, f"{skill_id}/SKILL.md", remaining_chars, limit)
    elif skill_language == "Chinese":
        _append_file(
            parts,
            cn_path if cn_path.exists() else en_path,
            f"{skill_id}/SKILL",
            remaining_chars,
            limit,
        )
    elif skill_language == "Both":
        _append_file(parts, cn_path, f"{skill_id}/SKILL.cn.md", remaining_chars, limit)
        _append_file(parts, en_path, f"{skill_id}/SKILL.md", remaining_chars, limit)
    else:
        _append_file(
            parts,
            cn_path if cn_path.exists() else en_path,
            f"{skill_id}/SKILL",
            remaining_chars,
            limit,
        )


def _append_output_language_policy(parts: list[str], skill_language: str) -> None:
    if skill_language in {"Chinese if available", "Chinese"}:
        parts.append(
            "Mandatory final output language: Simplified Chinese. The final answer is invalid if descriptive "
            "prompt prose is written in English. "
            "Keep the official English field names, section names, reference labels such as <Subject N>, "
            "<Picture N>, <Video N>, and <Audio N>, shot labels such as [Shot N], timing notation, XML-like "
            "tags, model names, and user-provided dialogue, lyrics, or visible text exactly as required by the "
            "official guide. Do not translate those structural tokens. If the official guide or examples are "
            "in English, translate their descriptive style and requirements into Simplified Chinese."
        )
    elif skill_language == "Both":
        parts.append(
            "Final output language policy: write the complete official H3 prompt in Simplified Chinese first, "
            "then provide an English version with the same structure. Keep official field names, section names, "
            "reference labels, shot labels, timing notation, XML-like tags, model names, and user-provided "
            "dialogue, lyrics, or visible text unchanged where the guide requires them."
        )
    else:
        parts.append(
            "Final output language policy: write all natural-language prompt prose in English. "
            "Keep the official field names, reference labels, shot labels, timing notation, XML-like tags, "
            "model names, and user-provided dialogue, lyrics, or visible text as required by the official guide."
        )


def _append_professional_storyboard_format(parts: list[str]) -> None:
    parts.append(
        "When the selected skill is Professional Chinese sales storyboard, convert the official H3 prompt requirements "
        "into a polished Chinese short-video prompt using exactly this bracket-section structure. Return only the "
        "prompt text, with no Markdown code fence, no explanation, and no notes.\n\n"
        "Required section order:\n"
        "【全局参数】\n"
        "State total duration, aspect ratio, generation mode, realism/style, target short-video purpose, subtitle/text "
        "policy, audio/dialogue policy, and shot count. Use the workflow duration and aspect ratio provided by the node. "
        "State explicitly that the whole video has no subtitles, captions, title text, stickers, floating text, "
        "corner labels, logos, or watermarks. State that the first 1 second and the final 1 second are silent for "
        "speech: no voiceover, dialogue, narration, or lip-sync speech may occur there. All spoken content must start "
        "after 1s and finish before total_duration - 1s.\n\n"
        "【人物与产品设定】\n"
        "Describe the main person, identity stability, wardrobe, environment, lighting, product/object appearance, "
        "material, color, scale, and consistency constraints. If there is no product, rename this section content to "
        "人物与核心主体设定 while keeping the heading unchanged.\n\n"
        "【口播文案】\n"
        "Write or preserve natural Simplified Chinese Mandarin voiceover when the request needs speech. Mention lip-sync "
        "requirements. Divide the voiceover into exact time ranges inside the allowed speech window, for example "
        "1s-3.5s: “...”, 3.5s-7s: “...”. "
        "Every spoken sentence must belong to one and only one time range, and the total spoken content must fit the "
        "available duration. Explicitly mark 0s-1s and the final 1s as 无口播. If the user did not request speech, write "
        "that there is no voiceover and preserve clean ambient sound instead.\n\n"
        "【分镜设计】\n"
        "Break the full duration into numbered shots such as 分镜1（约0s-3.5s）. Each shot must include subject action, "
        "camera movement, composition, transitions, texture details, background continuity, product/reference handling, "
        "the exact voiceover line for that time range, or explicitly state 无口播. The first 1 second and final 1 second "
        "must be labeled 无口播 even when they are inside a larger opening or closing shot. The time ranges must cover "
        "the full video continuously without gaps or overlaps. For 15-second product videos, prefer 4 shots unless the user "
        "explicitly asks otherwise.\n\n"
        "【转场要求】\n"
        "List natural cinematic transitions matched to the shot content, such as object wipe, reflection transition, "
        "hand/arm occlusion, hair wipe, foreground light wipe, liquid cover, or camera movement continuity. Forbid cheap "
        "jump cuts and spatially confusing transitions.\n\n"
        "【镜头与质感要求】\n"
        "Describe premium cinematography, macro detail, depth of field, real skin/object texture, stable motion, "
        "material highlights, foreground/midground/background layering, natural motion blur, and ad-level realism.\n\n"
        "【音效与音乐】\n"
        "Describe clean native audio, environmental sounds, object Foley, voice clarity, background music style, volume "
        "relationship, and emotional progression. Do not let music cover speech.\n\n"
        "【严格约束】\n"
        "List negative constraints in Chinese and make the no-text rule unconditional: absolutely no subtitles, captions, "
        "title text, stickers, floating text, product callouts, corner labels, logos, or watermarks anywhere in the video. "
        "Also forbid extra people, identity drift, hand errors, product deformation, cheap filters, overexposure, plastic "
        "skin, abrupt spatial jumps, and strong AI artifacts.\n\n"
        "Style requirements: make the Chinese prompt dense, professional, concrete, and production-ready. Do not be brief. "
        "Prefer 800-1500 Chinese characters for normal requests, and more if max_new_tokens allows it. Keep user-provided "
        "visible text, dialogue, brand names, and reference labels exactly when needed."
    )


def _build_h3_system(
    skill_id: str,
    h3_mode: str,
    skill_language: str,
    reference_depth: str,
    output_format: str,
) -> str:
    if not H3_SKILL_ROOT.exists():
        raise FileNotFoundError(
            f"MiniMax H3 skill files are missing: {H3_SKILL_ROOT}"
        )

    if skill_id in {BASE_SKILL_ID, PROFESSIONAL_STORYBOARD_SKILL_ID}:
        system_budget = H3_SYSTEM_BUDGET
        base_skill_limit = BASE_SKILL_LIMIT
        base_reference_limit = BASE_REFERENCE_LIMIT
        skill_limit = SKILL_LIMIT
        reference_limit = REFERENCE_LIMIT
    else:
        system_budget = SPECIALIZED_H3_SYSTEM_BUDGET
        base_skill_limit = SPECIALIZED_BASE_SKILL_LIMIT
        base_reference_limit = SPECIALIZED_BASE_REFERENCE_LIMIT
        skill_limit = SPECIALIZED_SKILL_LIMIT
        reference_limit = SPECIALIZED_REFERENCE_LIMIT

    remaining_chars = [system_budget]
    parts = [
        "You are a MiniMax H3 prompt optimizer running inside ComfyUI.",
        "Use the official MiniMax H3 skill files below as the authority.",
        "This node only writes prompts; do not browse, download, call tools, or ask follow-up questions.",
        "Infer missing details conservatively from the text and attached images.",
        "Preserve the exact MiniMax H3 field names, section order, reference labels, and timing notation required by the official guides.",
    ]
    if skill_id == PROFESSIONAL_STORYBOARD_SKILL_ID:
        parts.append(
            "Return only the final professional Chinese storyboard prompt, without commentary."
        )
        _append_professional_storyboard_format(parts)
    elif output_format == "H3 prompt only":
        parts.append("Return only the final optimized H3 prompt, without commentary.")
    else:
        parts.append("Return the final optimized H3 prompt first, then add a short notes section.")

    base_dir = H3_SKILL_ROOT / BASE_SKILL_ID
    _append_file(
        parts,
        base_dir / "SKILL.md",
        f"{BASE_SKILL_ID}/SKILL.md",
        remaining_chars,
        base_skill_limit,
    )

    if h3_mode == "Ref2VA":
        _append_file(
            parts,
            base_dir / "references" / "ref-en.txt",
            f"{BASE_SKILL_ID}/references/ref-en.txt",
            remaining_chars,
            base_reference_limit,
        )
    else:
        _append_file(
            parts,
            base_dir / "references" / "base-en.txt",
            f"{BASE_SKILL_ID}/references/base-en.txt",
            remaining_chars,
            base_reference_limit,
        )

    if skill_id not in {BASE_SKILL_ID, PROFESSIONAL_STORYBOARD_SKILL_ID}:
        skill_dir = H3_SKILL_ROOT / skill_id
        _append_skill_file(
            parts,
            skill_dir,
            skill_id,
            skill_language,
            remaining_chars,
            skill_limit,
        )
        if reference_depth == "Skill with references":
            _append_reference_files(
                parts,
                skill_dir,
                skill_id,
                remaining_chars,
                reference_limit,
            )

    _append_output_language_policy(parts, skill_language)
    return "\n\n---\n\n".join(parts)


def _build_h3_user(
    source_prompt: str,
    extra_requirements: str,
    h3_mode: str,
    duration_seconds: float,
    aspect_ratio: str,
    skill_id: str,
    output_format: str,
    skill_language: str,
) -> str:
    if skill_language in {"Chinese if available", "Chinese"}:
        language_policy = (
            "mandatory_output_language: Simplified Chinese. Write all scene descriptions, action descriptions, "
            "style descriptions, camera descriptions, retention analysis, summary, and notes in Simplified Chinese. "
            "Keep only official structural tokens such as subject_definitions, summary, retention_analysis, "
            "detailed_description, [Shot N], <Picture N>, <Subject N>, <d>, model names, and user-provided visible "
            "text/dialogue unchanged. Do not output English descriptive prose."
        )
    elif skill_language == "Both":
        language_policy = (
            "mandatory_output_language: Simplified Chinese first, then English with the same H3 structure."
        )
    else:
        language_policy = "mandatory_output_language: English."

    format_policy = ""
    if skill_id == PROFESSIONAL_STORYBOARD_SKILL_ID:
        format_policy = (
            "\nprofessional_output_contract: Use the exact Chinese bracket-section format: "
            "【全局参数】, 【人物与产品设定】, 【口播文案】, 【分镜设计】, 【转场要求】, "
            "【镜头与质感要求】, 【音效与音乐】, 【严格约束】. "
            "For every shot, write an explicit start-end time range and the exact voiceover line for that range. "
            "The first 1 second and the final 1 second must be speech-free and explicitly marked 无口播; all dialogue "
            "must start after 1s and finish before total_duration - 1s. "
            "State unconditionally that there are no subtitles or any other on-screen text. "
            "Write a dense production-ready prompt, not a short summary. Preserve exact user voiceover text when provided. "
            "Do not output Markdown, notes, analysis, TOML, JSON, or English descriptive prose."
        )

    return (
        "Optimize this request into a MiniMax H3 prompt.\n\n"
        f"h3_mode: {h3_mode}\n"
        f"target_duration_seconds: {duration_seconds:.2f}\n"
        f"duration_must_match_workflow_seconds: {duration_seconds:.2f}\n"
        "If source_prompt or examples mention another duration, ignore that duration and use "
        "target_duration_seconds exactly.\n"
        f"aspect_ratio: {aspect_ratio}\n"
        f"official_skill: {skill_id}\n"
        f"output_format: {output_format}\n"
        f"{language_policy}{format_policy}\n\n"
        "Attached images, if any, are H3 reference images in socket order: image_1, image_2, image_3, image_4.\n\n"
        f"source_prompt:\n{source_prompt.strip()}\n\n"
        f"extra_requirements:\n{extra_requirements.strip() if extra_requirements else 'None'}"
    )


class XinbaoH3PromptOptimizer:
    """MiniMax H3 prompt optimizer powered by the local Bonsai model."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "max_side": (
                    "INT",
                    {"default": 512, "min": 256, "max": 2048, "step": 64},
                ),
                "h3_skill": (_skill_choices(), {"default": SKILL_NAMES[BASE_SKILL_ID]}),
                "h3_mode": (list(H3_MODES), {"default": "文本生成视频 (T2VA)"}),
                "duration_seconds": (
                    "FLOAT",
                    {"default": 6.0, "min": 5.0, "max": 15.0, "step": 0.5},
                ),
                "aspect_ratio": (list(ASPECT_RATIOS), {"default": "16:9"}),
                "megapixels": (
                    "FLOAT",
                    {"default": DEFAULT_MEGAPIXELS, "min": 0.1, "max": 16.0, "step": 0.1},
                ),
                "skill_language": (list(SKILL_LANGUAGE), {"default": "优先使用中文"}),
                "reference_depth": (list(REFERENCE_DEPTH), {"default": "Skill 与参考资料"}),
                "output_format": (list(OUTPUT_FORMAT), {"default": "仅输出 H3 提示词"}),
                "source_prompt": (
                    "STRING",
                    {
                        "default": "一段电影感短视频提示词。",
                        "multiline": True,
                    },
                ),
                "extra_requirements": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.4, "min": 0.1, "max": 1.0, "step": 0.1},
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "max_new_tokens": (
                    "INT",
                    {"default": 2048, "min": 64, "max": 8192, "step": 64},
                ),
                "seed": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 0x7FFFFFFF,
                        "control_after_generate": True,
                    },
                ),
                "context_size": (
                    "INT", {"default": 32768, "min": 16384, "max": 131072, "step": 8192}
                ),
                "release_comfy_vram": ("BOOLEAN", {"default": True}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "workflow_width": ("INT",),
                "workflow_height": ("INT",),
                "workflow_duration": ("FLOAT",),
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("H3提示词",)
    FUNCTION = "forward"
    OUTPUT_NODE = True
    CATEGORY = "心宝❤推理（极速版）"
    DESCRIPTION = "使用本地 Ternary Bonsai 2 27B 和官方 H3 Skill 生成 MiniMax H3 专用提示词。"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        seed = kwargs.get("seed", -1)
        return float("nan") if seed is None or seed < 0 else seed

    def forward(self, **kwargs):
        skill_id = _normalize_skill(kwargs["h3_skill"])
        h3_mode = _normalize_choice(kwargs["h3_mode"], H3_MODES)
        skill_language = _normalize_choice(kwargs["skill_language"], SKILL_LANGUAGE)
        reference_depth = _normalize_choice(kwargs["reference_depth"], REFERENCE_DEPTH)
        output_format = _normalize_choice(kwargs["output_format"], OUTPUT_FORMAT)
        if output_format in {"Professional Chinese sales storyboard", "专业中文分镜带货格式", "中文带货视频"}:
            skill_id = PROFESSIONAL_STORYBOARD_SKILL_ID
            output_format = "H3 prompt only"
        width = int(kwargs.get("workflow_width") or 0)
        height = int(kwargs.get("workflow_height") or 0)
        if width < H3_CANVAS_MULTIPLE or height < H3_CANVAS_MULTIPLE:
            aspect_ratio = _normalize_aspect_ratio(kwargs["aspect_ratio"])
            width, height = _h3_resolution(
                aspect_ratio,
                _normalize_megapixels(kwargs["megapixels"]),
            )
        else:
            aspect_ratio = _aspect_ratio_from_dimensions(width, height)

        duration_seconds = _normalize_duration(
            kwargs.get("workflow_duration"),
            _normalize_duration(kwargs["duration_seconds"]),
        )
        max_new_tokens = min(int(kwargs["max_new_tokens"]), MAX_CHAT_NEW_TOKENS)
        if skill_id == PROFESSIONAL_STORYBOARD_SKILL_ID:
            max_new_tokens = max(max_new_tokens, PROFESSIONAL_MIN_NEW_TOKENS)
        if skill_id not in {BASE_SKILL_ID, PROFESSIONAL_STORYBOARD_SKILL_ID}:
            max_new_tokens = min(max_new_tokens, SPECIALIZED_MAX_NEW_TOKENS)
        system_text = _build_h3_system(
            skill_id,
            h3_mode,
            skill_language,
            reference_depth,
            output_format,
        )
        user_text = _build_h3_user(
            kwargs["source_prompt"],
            kwargs["extra_requirements"],
            h3_mode,
            duration_seconds,
            aspect_ratio,
            skill_id,
            output_format,
            skill_language,
        )

        if kwargs["release_comfy_vram"]:
            _release_comfy_models()

        model = _find_one(BONSAI_MODEL_DIR, "*PTQ1_0.gguf")
        mmproj = _find_one(BONSAI_MODEL_DIR, "*mmproj-Q8_0.gguf")
        SERVER.ensure(
            BONSAI_RUNTIME_DIR,
            model,
            mmproj,
            int(kwargs["context_size"]),
        )

        content = []
        for index in range(1, 5):
            image = kwargs.get(f"image_{index}")
            if image is None:
                continue
            pil = _tensor_frame_to_pil(image, int(kwargs["max_side"]))
            content.append({"type": "text", "text": f"H3 reference image_{index}"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _data_url(pil, quality=90)},
                }
            )
        content.append({"type": "text", "text": user_text})

        seed = int(kwargs["seed"])
        if seed < 0:
            seed = random.randint(0, 0x7FFFFFFF)
        payload = {
            "model": model.name,
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": content},
            ],
            "temperature": kwargs["temperature"],
            "top_p": kwargs["top_p"],
            "max_tokens": max_new_tokens,
            "seed": seed,
            "reasoning_effort": "none",
            "chat_template_kwargs": {"enable_thinking": False},
            "stream": False,
        }
        try:
            result = _post_json(SERVER.endpoint(), payload, timeout=1200)
            text = _clean_prompt(result["choices"][0]["message"]["content"])
            if not text:
                raise RuntimeError(f"Bonsai 模型返回了空结果：{result}")
            return {"ui": {"text": (text,)}, "result": (text,)}
        finally:
            if not kwargs["keep_model_loaded"]:
                SERVER.stop()


# Keep the legacy class name so existing H3 workflows load without relinking.
Qwen3VL_Li_H3PromptOptimizer = XinbaoH3PromptOptimizer
