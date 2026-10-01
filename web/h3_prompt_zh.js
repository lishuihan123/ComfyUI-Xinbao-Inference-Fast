import { app } from "../../scripts/app.js";

const NODE_CLASS = "Qwen3VL_Li_H3PromptOptimizer";
const NODE_TITLE = "H3 专用反推（Bonsai）";
const DEFAULT_MEGAPIXELS = 0.4;
const NODE_MIN_WIDTH = 320;
const H3_SKILL_WIDGET_INDEX = 1;
const OUTPUT_FORMAT_WIDGET_INDEX = 8;
const PROFESSIONAL_STORYBOARD_LABEL = "中文带货视频";
const OLD_PROFESSIONAL_STORYBOARD_LABEL = "专业中文分镜带货格式";
const COMBO_DEFAULTS = {
  h3_skill: "H3 通用提示词编写",
  h3_mode: "首帧生成视频 (I2VA)",
  skill_language: "严格遵循官方（英文）",
  reference_depth: "Skill 与参考资料",
  output_format: "仅输出 H3 提示词",
};
const HIDDEN_WIDGETS = new Set([
  "duration_seconds",
  "aspect_ratio",
  "megapixels",
  "extra_requirements",
]);

const WIDGET_LABELS = {
  max_side: "最大图像边长",
  h3_skill: "H3 提示词 Skill",
  h3_mode: "生成模式",
  duration_seconds: "视频时长（秒）",
  aspect_ratio: "画幅比例",
  megapixels: "分辨率（百万像素）",
  skill_language: "Skill 语言",
  reference_depth: "参考资料深度",
  output_format: "输出格式",
  source_prompt: "用户指令",
  extra_requirements: "补充要求",
  temperature: "随机性",
  top_p: "采样范围",
  max_new_tokens: "最大输出 Token 数",
  seed: "随机种子",
  release_comfy_vram: "运行前释放 ComfyUI 显存",
  keep_model_loaded: "保持 Bonsai 模型加载",
};

const SOCKET_LABELS = {
  image_1: "参考图像 1",
  image_2: "参考图像 2",
  image_3: "参考图像 3",
  image_4: "参考图像 4",
  image_5: "参考图像 5",
  image_6: "参考图像 6",
  image_7: "参考图像 7",
  image_8: "参考图像 8",
  image_9: "参考图像 9",
  image_10: "参考图像 10",
  workflow_width: "工作流宽度",
  workflow_height: "工作流高度",
  workflow_duration: "工作流时长（秒）",
  H3提示词: "H3 提示词",
};

function isValidMegapixels(value) {
  return typeof value === "number" && Number.isFinite(value) && value >= 0.1 && value <= 16;
}

function migrateLegacyValues(values) {
  if (!Array.isArray(values)) return values;

  let repaired = [...values];
  const isLegacyQwenLayout =
    repaired.length >= 22 &&
    typeof repaired[0] === "string" &&
    ["SDPA", "Flash Attention 2"].includes(repaired[2]);
  if (isLegacyQwenLayout) {
    repaired = [
      repaired[3], repaired[4], repaired[5], repaired[6], repaired[7], repaired[8],
      repaired[9], repaired[10], repaired[11], repaired[12], repaired[13], repaired[14],
      repaired[15], repaired[16], repaired[17], true, repaired[21],
    ];
  }

  if (!isValidMegapixels(repaired[5])) repaired[5] = DEFAULT_MEGAPIXELS;
  if (
    repaired[OUTPUT_FORMAT_WIDGET_INDEX] === PROFESSIONAL_STORYBOARD_LABEL ||
    repaired[OUTPUT_FORMAT_WIDGET_INDEX] === OLD_PROFESSIONAL_STORYBOARD_LABEL
  ) {
    repaired[H3_SKILL_WIDGET_INDEX] = PROFESSIONAL_STORYBOARD_LABEL;
    repaired[OUTPUT_FORMAT_WIDGET_INDEX] = COMBO_DEFAULTS.output_format;
  }
  if (repaired[H3_SKILL_WIDGET_INDEX] === OLD_PROFESSIONAL_STORYBOARD_LABEL) {
    repaired[H3_SKILL_WIDGET_INDEX] = PROFESSIONAL_STORYBOARD_LABEL;
  }
  return repaired;
}

function repairComboWidgets(node) {
  for (const widget of node.widgets ?? []) {
    const values = widget.options?.values;
    if (!Array.isArray(values) || values.length === 0 || values.includes(widget.value)) continue;

    const token = String(widget.value).match(/\(([^)]+)\)/)?.[1];
    const tokenValue = token && values.find((value) => String(value).includes(token));
    widget.value = tokenValue || COMBO_DEFAULTS[widget.name] || values[0];
  }
}

function hideWidget(widget) {
  if (!widget || widget.hidden) return;
  widget.hidden = true;
  widget.computeSize = () => [0, -4];
}

function applyLabels(node) {
  node.title = NODE_TITLE;
  for (const widget of node.widgets ?? []) {
    if (HIDDEN_WIDGETS.has(widget.name)) {
      hideWidget(widget);
      continue;
    }
    const label = WIDGET_LABELS[widget.name];
    if (label) {
      widget.label = label;
      widget.options = { ...(widget.options || {}), label };
    }
  }
  for (const input of node.inputs ?? []) {
    const label = SOCKET_LABELS[input.name];
    if (label) input.label = input.localized_name = label;
  }
  for (const output of node.outputs ?? []) {
    const label = SOCKET_LABELS[output.name];
    if (label) output.label = output.localized_name = label;
  }
  const computed = node.computeSize?.() ?? node.size ?? [NODE_MIN_WIDTH, 0];
  node.setSize?.([Math.max(computed[0] ?? 0, NODE_MIN_WIDTH), computed[1]]);
}

app.registerExtension({
  name: "Xinbao.InferenceFast.H3PromptChinese",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_CLASS) return;

    const originalConfigure = nodeType.prototype.configure;
    nodeType.prototype.configure = function (info) {
      if (info && Array.isArray(info.widgets_values)) {
        info.widgets_values = migrateLegacyValues(info.widgets_values);
      }
      const result = originalConfigure?.call(this, info);
      repairComboWidgets(this);
      applyLabels(this);
      return result;
    };

    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      applyLabels(this);
      return result;
    };
  },
});
