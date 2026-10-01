from __future__ import annotations

import math
import random
import re
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
DEFAULT_BONSAI_CONTEXT_SIZE = 32768
EXPANDED_BONSAI_CONTEXT_SIZE = 65536
MAX_BONSAI_CONTEXT_SIZE = 131072
SKILL_LANGUAGE = {
    "严格遵循官方（英文）": "Official English",
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
MAX_CHAT_NEW_TOKENS = 8192
PROFESSIONAL_MIN_NEW_TOKENS = 2048
SKILL_NAMES = {
    "h3-prompt-writing": "H3 通用提示词编写",
    "3d-animation-short-generator": "3D 动画短片",
    "brand-promo-video-generator": "品牌宣传片",
    "co-op-game-intro-generator": "合作游戏开场",
    "handdrawn-live-video-generator": "手绘实拍视频",
    "minimalist-product-ad-generator": "极简产品广告",
    "music-video-subtitle-generator": "MV 字幕视频",
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
    return min(15.0, max(4.0, duration))


def _normalize_aspect_ratio(value: str) -> str:
    return value.split(" ", 1)[0]


def _aspect_ratio_from_dimensions(width: int, height: int) -> str:
    ratio = width / max(1, height)
    return min(
        ASPECT_RATIOS,
        key=lambda name: abs(ASPECT_RATIOS[name][0] / ASPECT_RATIOS[name][1] - ratio),
    )


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _append_file(
    parts: list[str],
    path: Path,
    title: str,
) -> None:
    if path.exists():
        text = _read_text(path)
        if text:
            parts.append(f"## {title}\n\n{text}")


def _append_reference_files(
    parts: list[str],
    skill_dir: Path,
    skill_id: str,
) -> None:
    ref_dir = skill_dir / "references"
    if not ref_dir.exists():
        return

    for path in sorted(ref_dir.glob("*.md")) + sorted(ref_dir.glob("*.txt")):
        _append_file(
            parts,
            path,
            f"{skill_id}/{path.relative_to(skill_dir).as_posix()}",
        )


def _append_skill_file(
    parts: list[str],
    skill_dir: Path,
    skill_id: str,
    skill_language: str,
) -> None:
    cn_path = skill_dir / "SKILL.cn.md"
    en_path = skill_dir / "SKILL.md"

    if skill_language in {"Official English", "English"}:
        _append_file(parts, en_path, f"{skill_id}/SKILL.md")
    elif skill_language == "Chinese":
        _append_file(
            parts,
            cn_path if cn_path.exists() else en_path,
            f"{skill_id}/SKILL",
        )
    elif skill_language == "Both":
        _append_file(parts, cn_path, f"{skill_id}/SKILL.cn.md")
        _append_file(parts, en_path, f"{skill_id}/SKILL.md")
    else:
        _append_file(
            parts,
            cn_path if cn_path.exists() else en_path,
            f"{skill_id}/SKILL",
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


def _append_professional_storyboard_role(parts: list[str]) -> None:
    parts.append(
        "When the selected skill is Professional Chinese sales storyboard, convert the official H3 prompt requirements "
        "into a polished Chinese short-video prompt using exactly this bracket-section structure. Preserve the original "
        "Chinese e-commerce director role, its detailed product analysis, timed sales copy, shot planning, transitions, "
        "cinematography, sound design, and strict constraints. Return only the prompt text, with no Markdown code fence, "
        "explanation, or notes.\n\n"
        "Required section order:\n"
        "【全局参数】\n"
        "State total duration, aspect ratio, generation mode, realism/style, target short-video purpose, subtitle/text "
        "policy, audio/dialogue policy, and shot count. Use the workflow duration and aspect ratio provided by the node. "
        "State explicitly that the whole video has no subtitles, captions, title text, stickers, floating text, corner "
        "labels, logos, or watermarks. Unless the user explicitly specifies different speech timing, keep the first 1 second "
        "and final 1 second silent for speech, and place all spoken content between those boundaries. User instructions about "
        "whether speech exists, which shots speak, exact dialogue, segment count, timing, voice, and lip-sync always take priority "
        "over defaults. For a 15-second sales video with no user-specified script density or speed, default to a fast e-commerce "
        "cadence with 4 compact spoken segments; use 3 only when longer product actions or transitions genuinely need more visual "
        "breathing room. Aim for roughly 60-75 spoken Chinese characters in total, normally 15-19 characters per segment, at about "
        "5.7-6.75 Chinese characters per second with natural 0.10-0.20 second micro-pauses. Treat 40-50 total characters and about "
        "3.8-4.5 characters per second as medium speed, not the default. Use medium or slow speed only when the user explicitly asks "
        "for it. Never exceed the assigned shot time or sacrifice intelligibility, pronunciation, audio stability, or lip-sync "
        "accuracy merely to reach the target character count.\n\n"
        "【人物与产品设定】\n"
        "Describe the main person, identity stability, wardrobe, environment, lighting, product/object appearance, material, "
        "color, scale, and consistency constraints. Preserve details visible in the reference images. Do not invent brands, "
        "prices, discounts, certifications, ingredients, performance figures, medical claims, or unsupported product features. "
        "Also define one stable natural speaking voice for each speaker. If the user specifies voice, accent, age impression, "
        "pitch, timbre, or speaking speed, follow it exactly. Otherwise infer a suitable voice conservatively from the visible "
        "adult model's apparent age range, gender presentation, expression, temperament, scene, product category, and sales style. "
        "Do not invent a regional accent or an exact identity; when visual cues are uncertain, use a neutral natural adult Mandarin "
        "voice. Keep the same speaker's voice consistent across every shot.\n\n"
        "【分镜设计】\n"
        "Break the full duration into numbered shots such as 分镜1（约0s-3.5s）. Each shot must include subject action, "
        "camera movement, composition, transition intent, texture details, background continuity, and product/reference "
        "handling. Apply this lip-sync decision hierarchy exactly: (1) if the user requests every spoken segment to lip-sync, "
        "make every speaking shot show the speaker's visible mouth opening and matching the assigned cue; (2) if the user names "
        "specific lip-sync shots or says later shots need no lip-sync, follow those exact shot assignments; (3) if the user says "
        "nothing about lip-sync, the first suitable shot—normally 分镜1—must be a clear on-camera sales presentation after the "
        "opening silent second, with S1 facing or nearly facing camera, visibly opening the mouth and accurately lip-syncing 口播1. "
        "For later cues, decide shot by shot: use visible lip-sync when the face and mouth are clear and direct selling fits the "
        "action; use an off-screen voiceover for product macro, fabric detail, transition, occluded face, distant view, or other "
        "shots where mouth movement would be visually inappropriate. For off-screen speech use the official phrase ‘says in an "
        "off-screen voiceover’ and state that any on-screen character's lips remain completely closed. Never ambiguously combine "
        "voiceover with mouth movement. Put all speech information directly inside the precise speaking shot: shot time, speaker "
        "identity and stable voice, on-camera lip-sync or off-screen mode, full exact dialogue, speaking action, mouth visibility, "
        "pace, and pauses. Do not create a separate 【口播文案】 section or a separate cue index. Put the exact dialogue inline at "
        "the precise action point of every speaking shot. For "
        "visible lip-sync use, for example: ‘年轻女主播 (S1) says: <d>[Chinese] 这件衣服轻盈柔软，穿着很舒服。</d>，她正面对"
        "镜头自然张嘴说话，嘴部清晰可见，嘴唇开合、音节、节奏和停顿与该句逐字同步’. For voiceover use the exact official pattern: "
        "‘女主播 (S1) says in an off-screen voiceover: <d>[Chinese] ...</d>，镜头内人物嘴唇全程闭合’. Never place the same "
        "exact words in another shot, 【音效与音乐】, or any other section. Unless "
        "the user overrides it, label the first 1 second and final 1 second 无口播. Time ranges must cover the full video continuously "
        "without gaps or overlaps. For 15-second product videos, prefer 4 shots unless the user explicitly asks otherwise.\n\n"
        "【转场要求】\n"
        "List natural cinematic transitions matched to the shot content, such as object wipe, reflection transition, "
        "hand/arm occlusion, hair wipe, foreground light wipe, liquid cover, or camera movement continuity. Forbid cheap "
        "jump cuts and spatially confusing transitions.\n\n"
        "【镜头与质感要求】\n"
        "Describe premium cinematography, macro detail, depth of field, real skin/object texture, stable motion, material "
        "highlights, foreground/midground/background layering, natural motion blur, and ad-level realism.\n\n"
        "【音效与音乐】\n"
        "Describe clean native audio, environmental sounds, object Foley, voice clarity, background music style, volume "
        "relationship, and emotional progression. Require a lifelike human voice with natural breath, micro-pauses, changing "
        "intonation, sentence stress, pitch movement, emotional warmth, and conversational rhythm matching the visible presenter. "
        "Forbid robotic cadence, constant pitch, mechanical text-to-speech delivery, metallic or vocoder-like tone, over-articulation, "
        "unnatural syllable spacing, and identical timing between phrases. Do not repeat or quote dialogue here, and do not let "
        "music cover speech.\n\n"
        "【严格约束】\n"
        "List negative constraints in Chinese and make the no-text rule unconditional: absolutely no subtitles, captions, "
        "title text, stickers, floating text, product callouts, corner labels, logos, or watermarks anywhere in the video. "
        "Also forbid extra people, identity drift, hand errors, product deformation, cheap filters, overexposure, plastic "
        "skin, abrupt spatial jumps, strong AI artifacts, any repeated or improvised spoken sentence, lip movement during an "
        "off-screen voiceover, a speaking shot whose visible mouth is static, robotic or metallic voice, flat monotone delivery, "
        "unnatural word spacing, unstable speaker timbre, and speech too fast for reliable pronunciation or lip-sync.\n\n"
        "Style requirements: retain the original dense, professional, concrete, production-ready Chinese sales-video style. "
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
    parts = [
        "You are a MiniMax H3 prompt optimizer running inside ComfyUI.",
        "Use the complete official MiniMax H3 skill and mode-specific reference guide below as the authority. Do not omit, summarize, or weaken their requirements.",
        "This node only writes prompts; do not browse, download, call tools, or ask follow-up questions.",
        "Infer missing details conservatively from the text and attached images.",
        "Preserve the exact MiniMax H3 field names, section order, reference labels, and timing notation required by the official guides.",
        "Before answering, silently audit the prompt for the required section order, reference-label consistency, exact duration, shot-level composition, subject appearance and position, environment, lighting, actions and state changes, camera movement, synchronized sound, and every connected image. Return a complete production-ready prompt, never a short plot summary.",
    ]
    if skill_id == PROFESSIONAL_STORYBOARD_SKILL_ID:
        parts.append(
            "Return only the final professional Chinese sales storyboard prompt, without commentary."
        )
    elif output_format == "H3 prompt only":
        parts.append("Return only the final optimized H3 prompt, without commentary.")
    else:
        parts.append("Return the final optimized H3 prompt first, then add a short notes section.")

    base_dir = H3_SKILL_ROOT / BASE_SKILL_ID
    _append_file(
        parts,
        base_dir / "SKILL.md",
        f"{BASE_SKILL_ID}/SKILL.md",
    )

    if h3_mode == "Ref2VA":
        _append_file(
            parts,
            base_dir / "references" / "ref-en.txt",
            f"{BASE_SKILL_ID}/references/ref-en.txt",
        )
    else:
        _append_file(
            parts,
            base_dir / "references" / "base-en.txt",
            f"{BASE_SKILL_ID}/references/base-en.txt",
        )

    if skill_id not in {BASE_SKILL_ID, PROFESSIONAL_STORYBOARD_SKILL_ID}:
        skill_dir = H3_SKILL_ROOT / skill_id
        _append_skill_file(
            parts,
            skill_dir,
            skill_id,
            skill_language,
        )
        if reference_depth == "Skill with references":
            _append_reference_files(
                parts,
                skill_dir,
                skill_id,
            )

    if skill_id == PROFESSIONAL_STORYBOARD_SKILL_ID:
        _append_professional_storyboard_role(parts)

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
    image_indices: list[int],
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

    if skill_language in {"Chinese if available", "Chinese"}:
        detail_floor = max(700, round(duration_seconds * 80))
        detail_policy = (
            f"minimum_detail_target: At least {detail_floor} Chinese descriptive characters in the main timeline "
            "section. Reach the target with concrete composition, identity, environment, lighting, action progression, "
            "camera movement, material, continuity, and synchronized sound detail instead of repetition."
        )
    else:
        detail_floor = max(250, round(duration_seconds * 25))
        if h3_mode == "Ref2VA":
            detail_floor = max(350, detail_floor)
        detail_policy = (
            f"minimum_detail_target: At least {detail_floor} English words in the main timeline section. "
            "Do not pad with repetition; satisfy the target through concrete shot-level visual, motion, camera, lighting, and audio detail."
        )

    format_policy = ""
    if skill_id == PROFESSIONAL_STORYBOARD_SKILL_ID:
        format_policy = (
            "\nprofessional_output_contract: Preserve the original exact Chinese bracket-section format: "
            "【全局参数】, 【人物与产品设定】, 【分镜设计】, 【转场要求】, "
            "【镜头与质感要求】, 【音效与音乐】, 【严格约束】. "
            "Keep the original detailed Chinese sales-video director role and dense production-ready output. "
            "Do not create a separate 【口播文案】 section or cue index. Put every speech detail directly inside its corresponding "
            "shot in 【分镜设计】: timing, stable speaker ID and voice, lip-sync or off-screen mode, full exact sentence, speaking "
            "action, mouth visibility, pace, and pauses. Write each exact sentence once and only once using the official inline "
            "syntax (S1) says: <d>[Chinese] ...</d>, beside the visible speaking action and precise mouth-sync requirement. Never "
            "quote, paraphrase, summarize, or restate those words elsewhere. User instructions "
            "about dialogue, cue count, timing, speed, and which "
            "shots lip-sync override every default. If the user gives no lip-sync instruction, make the first suitable shot—"
            "normally Shot 1 after the opening silent second—a visible mouth-opening, accurate lip-sync sales presentation; "
            "choose visible lip-sync or off-screen voiceover for each later cue according to whether the shot clearly shows a "
            "speaking face. Mark voiceover explicitly with ‘says in an off-screen voiceover’ and keep on-screen lips completely "
            "closed. If the user gives no voice specification, infer a stable suitable natural Mandarin voice from the visible "
            "adult model's apparent age range, gender presentation, temperament, expression, scene, product, and sales style; "
            "when uncertain, use a neutral natural adult voice. Require lifelike breath, micro-pauses, intonation, sentence stress, "
            "pitch movement, and conversational rhythm, never robotic, metallic, monotone, or mechanically even delivery. For an "
            "otherwise unspecified 15-second video, default to fast e-commerce speech: normally write 4 compact cues (3 only when "
            "the visual rhythm genuinely needs more room), about 60-75 spoken Chinese characters total, 15-19 characters per cue, "
            "at roughly 5.7-6.75 characters per second with 0.10-0.20 second natural micro-pauses. Treat 40-50 total characters and "
            "roughly 3.8-4.5 characters per second as medium speed. Use medium or slow speed only when explicitly requested, and "
            "never sacrifice intelligibility, pronunciation, audio stability, or lip-sync accuracy to hit a numeric target. Unless the user overrides it, "
            "the first 1 second and final 1 second are speech-free. Do not repeat "
            "dialogue in 【音效与音乐】 or any other "
            "section. State unconditionally that there are no subtitles or other on-screen text unless the user requests it. "
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
        f"{language_policy}\n{detail_policy}{format_policy}\n\n"
        f"connected_reference_image_count: {len(image_indices)}\n"
        + (
            f"Connected image sockets: {', '.join(f'image_{index}' for index in image_indices)}. "
            "Their official Picture numbers must match these socket numbers. "
            "Inspect and account for every connected image; never silently ignore one.\n\n"
            if image_indices
            else "There are no connected reference images. Do not invent Picture or Subject references.\n\n"
        )
        + f"source_prompt:\n{source_prompt.strip()}\n\n"
        f"extra_requirements:\n{extra_requirements.strip() if extra_requirements else 'None'}"
    )


def _detail_floor(duration_seconds: float, h3_mode: str, skill_language: str) -> int:
    if skill_language in {"Chinese if available", "Chinese"}:
        return max(700, round(duration_seconds * 80))
    floor = max(250, round(duration_seconds * 25))
    return max(350, floor) if h3_mode == "Ref2VA" else floor


def _normalize_dialogue_text(text: str) -> str:
    return re.sub(r"[\s，。！？、；：,.!?;:\"'“”‘’]", "", text).casefold()


def _professional_prompt_issues(text: str) -> list[str]:
    issues = []
    dialogue_matches = list(re.finditer(r"<d>\s*\[([^\]]+)]\s*(.*?)\s*</d>", text, re.DOTALL | re.IGNORECASE))
    if text.lower().count("<d>") != len(dialogue_matches) or text.lower().count("</d>") != len(dialogue_matches):
        issues.append("dialogue tags are incomplete or do not use <d>[Language] exact words</d>")

    if dialogue_matches and not re.search(r"\(S\d+(?:\s*,\s*S\d+)*\)", text):
        issues.append("spoken dialogue has no stable speaker ID such as (S1)")

    normalized_dialogue = [_normalize_dialogue_text(match.group(2)) for match in dialogue_matches]
    duplicates = sorted({line for line in normalized_dialogue if line and normalized_dialogue.count(line) > 1})
    if duplicates:
        issues.append("one or more spoken lines are repeated in multiple <d> blocks")

    without_dialogue = re.sub(r"<d>.*?</d>", "", text, flags=re.DOTALL | re.IGNORECASE)
    normalized_without_dialogue = _normalize_dialogue_text(without_dialogue)
    if any(len(line) >= 4 and line in normalized_without_dialogue for line in normalized_dialogue):
        issues.append("spoken words are restated outside their single <d> block")

    if "【口播文案】" in text:
        issues.append("do not create a separate 【口播文案】 section; put all speech details directly inside each speaking shot")

    storyboard_index = text.find("【分镜设计】")
    transition_index = text.find("【转场要求】", storyboard_index)
    if dialogue_matches and storyboard_index >= 0:
        storyboard_end = transition_index if transition_index >= 0 else len(text)
        if any(not (storyboard_index <= match.start() < storyboard_end) for match in dialogue_matches):
            issues.append("every exact <d> dialogue line must appear once directly inside its corresponding 【分镜设计】 shot")

        missing_inline_syntax = False
        for match in dialogue_matches:
            prefix = text[max(storyboard_index, match.start() - 180):match.start()]
            if not re.search(r"\(S\d+(?:\s*,\s*S\d+)*\).*?says(?:\s+in\s+an\s+off-screen\s+voiceover)?\s*:\s*$", prefix, re.DOTALL | re.IGNORECASE):
                missing_inline_syntax = True
                break
        if missing_inline_syntax:
            issues.append("inline shot dialogue must bind speaker and words with (S1) says: <d>[Chinese] ...</d>")

    if not dialogue_matches and re.search(r"(?:口播|对白|旁白|says|voiceover).{0,100}[“\"]", text, re.DOTALL | re.IGNORECASE):
        issues.append("spoken content is written as prose or quotation instead of one <d>[Language] block")
    return issues


def _prompt_issues(
    text: str,
    h3_mode: str,
    skill_id: str,
    skill_language: str,
    duration_seconds: float,
) -> list[str]:
    if skill_id == PROFESSIONAL_STORYBOARD_SKILL_ID:
        required = [
            "【全局参数】", "【人物与产品设定】", "【分镜设计】",
            "【转场要求】", "【镜头与质感要求】", "【音效与音乐】", "【严格约束】",
        ]
        main_text = text
    elif h3_mode == "Ref2VA":
        required = [
            "subject_definitions:", "summary:", "retention_analysis:",
            "detailed_description:", "overall_soundscape:", "non_diegetic_music:",
        ]
        main_text = text.split("overall_soundscape:", 1)[0]
    else:
        required = [
            "integrated_multimodal_description:",
            "overall_soundscape:",
            "non_diegetic_music:",
        ]
        main_text = text.split("overall_soundscape:", 1)[0]

    issues = [f"missing required section {name}" for name in required if name not in text]
    if skill_id == PROFESSIONAL_STORYBOARD_SKILL_ID:
        issues.extend(_professional_prompt_issues(text))
        floor = _detail_floor(duration_seconds, h3_mode, skill_language)
    else:
        floor = _detail_floor(duration_seconds, h3_mode, skill_language)
    if skill_language in {"Chinese if available", "Chinese"}:
        actual = len(re.findall(r"[\u3400-\u9fff]", main_text))
        if actual < floor:
            issues.append(f"main timeline has only {actual} Chinese characters; target at least {floor}")
    else:
        actual = len(re.findall(r"\b[A-Za-z]+(?:[-'][A-Za-z]+)*\b", main_text))
        if actual < floor:
            issues.append(f"main timeline has only {actual} English words; target at least {floor}")
    return issues


def _automatic_context_size(
    system_text: str,
    user_text: str,
    image_count: int,
    max_output_tokens: int,
) -> int:
    estimated_tokens = (
        (len(system_text) + len(user_text) + 3) // 4
        + image_count * 2048
        + max_output_tokens * 2
        + 2048
    )
    if estimated_tokens <= 28672:
        return DEFAULT_BONSAI_CONTEXT_SIZE
    if estimated_tokens <= 57344:
        return EXPANDED_BONSAI_CONTEXT_SIZE
    return MAX_BONSAI_CONTEXT_SIZE


def _main_section_parts(text: str, h3_mode: str) -> tuple[str, str, str] | None:
    label = "detailed_description:" if h3_mode == "Ref2VA" else "integrated_multimodal_description:"
    start = text.find(label)
    end = text.find("overall_soundscape:", start + len(label))
    if start < 0 or end < 0:
        return None
    return text[:start], label, text[end:]


def _segment_body(text: str, h3_mode: str) -> str:
    labels = ["detailed_description:", "integrated_multimodal_description:"]
    body = text.strip()
    for label in labels:
        if label in body:
            body = body.split(label, 1)[1].strip()
    if "overall_soundscape:" in body:
        body = body.split("overall_soundscape:", 1)[0].strip()
    if "non_diegetic_music:" in body:
        body = body.split("non_diegetic_music:", 1)[0].strip()
    if h3_mode != "Ref2VA" and body.startswith("For the target video"):
        body = body.split("\n", 1)[1].strip() if "\n" in body else ""
    return body


def _segment_language_matches(text: str, chinese_output: bool) -> bool:
    cjk_count = len(re.findall(r"[\u3400-\u9fff]", text))
    latin_count = len(re.findall(r"[A-Za-z]", text))
    if chinese_output:
        return cjk_count >= 80 and cjk_count >= latin_count * 0.5
    return latin_count >= 120 and latin_count >= cjk_count * 2


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
                "h3_skill": (
                    _skill_choices(),
                    {
                        "default": SKILL_NAMES[BASE_SKILL_ID],
                        "tooltip": "选择内容角色；中文带货视频把口播原文写进对应分镜，默认匹配真人音色，15秒通常生成4段快速口播；中速或慢速须由用户明确指定。",
                    },
                ),
                "h3_mode": (list(H3_MODES), {"default": "文本生成视频 (T2VA)"}),
                "duration_seconds": (
                    "FLOAT",
                    {"default": 6.0, "min": 4.0, "max": 15.0, "step": 0.5},
                ),
                "aspect_ratio": (list(ASPECT_RATIOS), {"default": "16:9"}),
                "megapixels": (
                    "FLOAT",
                    {"default": DEFAULT_MEGAPIXELS, "min": 0.1, "max": 16.0, "step": 0.1},
                ),
                "skill_language": (list(SKILL_LANGUAGE), {"default": "严格遵循官方（英文）"}),
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
                    {"default": 3072, "min": 512, "max": 8192, "step": 128},
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
                "release_comfy_vram": ("BOOLEAN", {"default": True}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "workflow_width": ("INT",),
                "workflow_height": ("INT",),
                "workflow_duration": ("FLOAT",),
                **{f"image_{index}": ("IMAGE",) for index in range(1, 11)},
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

        image_indices = [
            index for index in range(1, 11) if kwargs.get(f"image_{index}") is not None
        ]
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
            image_indices,
        )

        if kwargs["release_comfy_vram"]:
            _release_comfy_models()

        model = _find_one(BONSAI_MODEL_DIR, "*PTQ1_0.gguf")
        mmproj = _find_one(BONSAI_MODEL_DIR, "*mmproj-Q8_0.gguf")
        context_size = _automatic_context_size(
            system_text,
            user_text,
            len(image_indices),
            max_new_tokens,
        )
        SERVER.ensure(
            BONSAI_RUNTIME_DIR,
            model,
            mmproj,
            context_size,
        )

        content = []
        for index in image_indices:
            image = kwargs[f"image_{index}"]
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
            issues = _prompt_issues(
                text,
                h3_mode,
                skill_id,
                skill_language,
                duration_seconds,
            )
            missing_sections = [issue for issue in issues if issue.startswith("missing required section")]
            if missing_sections or skill_id == PROFESSIONAL_STORYBOARD_SKILL_ID:
                audit_issues = issues or [
                    "perform the final dense Chinese sales-storyboard consistency pass without changing its role or structure"
                ]
                professional_repair_rule = (
                    " Preserve the seven required Chinese bracket sections and remove any separate 【口播文案】 section or cue "
                    "index. Put every speech detail—time, speaker and stable voice, delivery mode, full exact words, mouth action, "
                    "pace, and pauses—directly inside its speaking shot in 【分镜设计】. Put each exact spoken line once and only "
                    "once using "
                    "(S1) says: <d>[Chinese] ...</d>, adjacent to the visible mouth-opening and precise lip-sync action. User "
                    "instructions about dialogue and lip-sync have highest "
                    "priority. If the user did not specify lip-sync, make the first suitable shot a visible mouth-opening accurate "
                    "lip-sync presentation, then choose later lip-sync or explicitly marked off-screen voiceover according to the "
                    "shot. If the user did not specify a voice, infer one stable natural human Mandarin voice from the visible "
                    "adult model and product mood; explicitly forbid robotic, metallic, monotone, mechanically even delivery. "
                    "For an otherwise unspecified 15-second video, default to 4 fast but controllable cues totaling about 60-75 "
                    "Chinese characters at 5.7-6.75 characters per second; 40-50 characters at 3.8-4.5 characters per second is "
                    "medium speed and should be used only when requested. Slow down only when requested. Never force extra words "
                    "into a cue whose duration cannot safely contain them or sacrifice pronunciation and lip-sync stability."
                    if skill_id == PROFESSIONAL_STORYBOARD_SKILL_ID
                    else ""
                )
                repair_messages = [
                    *payload["messages"],
                    {"role": "assistant", "content": text},
                    {
                        "role": "user",
                        "content": (
                            "Rewrite the draft in full. It failed the final H3 compliance audit for these reasons:\n- "
                            + "\n- ".join(audit_issues)
                            + "\nFollow the complete official guide already provided in the system message. "
                            "Preserve the user's intent, exact duration, all connected references, and required output language. "
                            "Add concrete shot-level composition, subject, environment, action, camera, lighting, material, continuity, "
                            "and synchronized audio detail instead of repetition."
                            + professional_repair_rule
                            + " Output only the complete replacement prompt."
                        ),
                    },
                ]
                repair_payload = {
                    **payload,
                    "messages": repair_messages,
                    "temperature": min(float(kwargs["temperature"]), 0.2),
                    "max_tokens": min(MAX_CHAT_NEW_TOKENS, max(max_new_tokens, 4096)),
                }
                repair = _post_json(SERVER.endpoint(), repair_payload, timeout=1200)
                repaired_text = _clean_prompt(repair["choices"][0]["message"]["content"])
                if repaired_text:
                    text = repaired_text

            issues = _prompt_issues(
                text,
                h3_mode,
                skill_id,
                skill_language,
                duration_seconds,
            )
            needs_more_detail = any(issue.startswith("main timeline has only") for issue in issues)
            section_parts = _main_section_parts(text, h3_mode)
            if needs_more_detail and section_parts and skill_id != PROFESSIONAL_STORYBOARD_SKILL_ID:
                floor = _detail_floor(duration_seconds, h3_mode, skill_language)
                chinese_output = skill_language in {"Chinese if available", "Chinese"}
                detail_per_segment = 420 if chinese_output else 200
                segment_count = min(4, max(2, math.ceil(floor / detail_per_segment)))
                segment_target = math.ceil(floor / segment_count)
                segments = []
                for segment_index in range(segment_count):
                    start_seconds = duration_seconds * segment_index / segment_count
                    end_seconds = duration_seconds * (segment_index + 1) / segment_count
                    length_rule = (
                        f"at least {segment_target} Chinese descriptive characters"
                        if chinese_output
                        else f"at least {segment_target} English words"
                    )
                    if segment_index == 0:
                        opening_rule = "Begin with [Shot 1] and establish the complete opening composition."
                    elif h3_mode == "FL2VA":
                        opening_rule = (
                            f"Continue the same uncut [Shot 1] from {start_seconds:.3f} seconds; do not introduce a new shot label."
                        )
                    else:
                        opening_rule = (
                            f"Begin with [Shot {segment_index + 1}] At 00:{start_seconds:06.3f}, using an official cut or transition expression."
                        )
                    if chinese_output:
                        if segment_index == 0:
                            opening_rule = "必须以 [Shot 1] 开头，并完整建立开场构图。"
                        elif h3_mode == "FL2VA":
                            opening_rule = (
                                f"从 {start_seconds:.3f} 秒继续同一个不切镜的 [Shot 1]，不要增加新的 Shot 标签。"
                            )
                        else:
                            opening_rule = (
                                f"必须以 [Shot {segment_index + 1}] At 00:{start_seconds:06.3f}, 开头，并使用官方允许的切镜或转场表达。"
                            )
                        segment_prompt = (
                            f"只编写完整时间线的第 {segment_index + 1}/{segment_count} 段，覆盖目标视频 "
                            f"{start_seconds:.3f}-{end_seconds:.3f} 秒。{opening_rule} "
                            f"本段描述性正文不得少于 {segment_target} 个简体中文汉字。全部描述性文字必须使用简体中文，"
                            "不得出现英文描述句；只保留 [Shot N]、时间码、引用标签和其他官方固定结构标记。"
                            "严格延续完整草稿、用户要求和每一张参考图，具体描写构图、稳定的主体身份与位置、环境、光线、材质、"
                            "动作发展、带幅度和速度的运镜、物理状态变化以及同步的画内声音，不得靠同义反复凑长度。"
                            "不要输出参考图对齐说明、字段名、overall_soundscape、non_diegetic_music、备注或解释，只输出这一段正文。"
                        )
                    else:
                        segment_prompt = (
                            f"Write only timeline segment {segment_index + 1} of {segment_count}, covering "
                            f"{start_seconds:.3f}-{end_seconds:.3f} seconds of the target video. {opening_rule} "
                            f"Write {length_rule}. Write every descriptive sentence in English. Preserve continuity with the complete "
                            "draft, the user's request, and every reference image. Describe composition, stable subject identity and position, "
                            "environment, lighting, materials, action progression, camera motion with amplitude and speed, physical state "
                            "changes, and synchronized diegetic sound. Do not output alignment instructions, field names, overall_soundscape, "
                            "non_diegetic_music, notes, or explanations."
                        )
                    segment_payload = {
                        **payload,
                        "messages": [
                            *payload["messages"],
                            {"role": "assistant", "content": text},
                            {"role": "user", "content": segment_prompt},
                        ],
                        "temperature": min(float(kwargs["temperature"]), 0.25),
                        "max_tokens": min(MAX_CHAT_NEW_TOKENS, max(1536, max_new_tokens // 2)),
                    }
                    segment_result = _post_json(SERVER.endpoint(), segment_payload, timeout=1200)
                    segment = _segment_body(
                        _clean_prompt(segment_result["choices"][0]["message"]["content"]),
                        h3_mode,
                    )
                    if segment and not _segment_language_matches(segment, chinese_output):
                        translation_rule = (
                            "将输入完整改写为简体中文，保留 [Shot N]、At 00:SS.mmm、<Picture N>、<Subject N>、"
                            "<d> 标签、专有名词和其他官方固定结构。不得缩写、概括或删除任何视觉、动作、运镜、光线、材质和声音细节。"
                            "只输出完整的中文段落，不要解释。"
                            if chinese_output
                            else "Rewrite the complete input in English while preserving [Shot N], timestamps, reference labels, "
                            "dialogue tags, proper nouns, and every visual, action, camera, lighting, material, and sound detail. "
                            "Do not summarize or omit content. Output the complete English segment only."
                        )
                        translation_payload = {
                            **payload,
                            "messages": [
                                {"role": "system", "content": translation_rule},
                                {"role": "user", "content": segment},
                            ],
                            "temperature": 0.1,
                            "max_tokens": min(MAX_CHAT_NEW_TOKENS, max(2048, max_new_tokens)),
                        }
                        translation_result = _post_json(
                            SERVER.endpoint(), translation_payload, timeout=1200
                        )
                        translated = _segment_body(
                            _clean_prompt(translation_result["choices"][0]["message"]["content"]),
                            h3_mode,
                        )
                        if translated:
                            segment = translated
                    if segment:
                        segments.append(segment)
                if len(segments) == segment_count:
                    prefix, main_label, suffix = section_parts
                    text = prefix + main_label + " " + "\n\n".join(segments) + "\n\n" + suffix.lstrip()
            return {"ui": {"text": (text,)}, "result": (text,)}
        finally:
            if not kwargs["keep_model_loaded"]:
                SERVER.stop()


# Keep the legacy class name so existing H3 workflows load without relinking.
Qwen3VL_Li_H3PromptOptimizer = XinbaoH3PromptOptimizer
