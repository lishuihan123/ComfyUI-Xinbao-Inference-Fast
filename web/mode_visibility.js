import { app } from "../../scripts/app.js";

const NODE_CLASS = "Bonsai2ReversePrompt";
const MODE_WIDGET = "inference_mode";
const ROLE_WIDGET = "role_positioning";
const BONSAI_MODE_PREFIX = "Bonsai";

function refreshNodeSize(node) {
    const computed = node.computeSize?.();
    if (computed) {
        node.setSize?.([
            Math.max(node.size?.[0] ?? 0, computed[0]),
            computed[1],
        ]);
    }
    node.graph?.setDirtyCanvas?.(true, true);
}

function setWidgetHidden(node, widget, hidden) {
    if (!widget) return;

    if (!("_xinbaoOriginalComputeSize" in widget)) {
        widget._xinbaoOriginalComputeSize = widget.computeSize;
    }

    widget.hidden = hidden;
    widget.computeSize = hidden
        ? () => [0, -4]
        : widget._xinbaoOriginalComputeSize;
    refreshNodeSize(node);
}

function syncRoleVisibility(node) {
    const mode = node.widgets?.find((widget) => widget.name === MODE_WIDGET);
    const role = node.widgets?.find((widget) => widget.name === ROLE_WIDGET);
    if (!mode || !role) return;

    const isBonsai = String(mode.value ?? "").startsWith(BONSAI_MODE_PREFIX);
    setWidgetHidden(node, role, !isBonsai);
}

function installModeCallback(node) {
    const mode = node.widgets?.find((widget) => widget.name === MODE_WIDGET);
    if (!mode || mode._xinbaoVisibilityCallbackInstalled) return;

    mode._xinbaoVisibilityCallbackInstalled = true;
    const originalCallback = mode.callback;
    mode.callback = function () {
        const result = originalCallback?.apply(this, arguments);
        syncRoleVisibility(node);
        return result;
    };
}

app.registerExtension({
    name: "Xinbao.InferenceFast.ModeVisibility",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_CLASS) return;

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated?.apply(this, arguments);
            installModeCallback(this);
            setTimeout(() => syncRoleVisibility(this), 0);
            return result;
        };

        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = originalOnConfigure?.apply(this, arguments);
            installModeCallback(this);
            setTimeout(() => syncRoleVisibility(this), 0);
            return result;
        };
    },
});
