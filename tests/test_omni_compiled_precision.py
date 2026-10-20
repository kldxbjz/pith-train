"""Production fullgraph stages preserve explicit BF16 rounding boundaries on CPU.

Stage5 executes its complete original body. Stage1 fixes the attention output
to isolate residual and norm arithmetic; it is not FA4 or router acceptance.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from tests.test_omni_stage5_compile import layer_type

ROOT = Path(__file__).parents[1]


def value_and_grads(call, layer, sources, indices, cotangent):
    outputs, weights, residual = [
        None if value is None else value.detach().clone().requires_grad_() for value in sources
    ]
    result = call(layer, outputs, indices, weights, residual)
    values = (outputs, residual) if weights is None else (outputs, weights, residual)
    gradients = torch.autograd.grad((result.float() * cotangent.float()).sum(), values)
    return result, gradients


@pytest.mark.parametrize("kind", ["text", "image"])
@pytest.mark.parametrize("is_moe,ep", [(False, 1), (True, 1), (True, 2)])
def test_compiled_stage5_preserves_bf16_output_and_every_input_gradient(kind, is_moe, ep):
    torch.manual_seed(19)
    image = layer_type(ep)
    layer = (image if kind == "image" else image.__bases__[0])()
    layer.is_moe, layer.idx, layer.text_hidden_size, layer.deepstack_count = is_moe, 1, 128, 2
    residual = torch.randn(1, 69, 384 if kind == "image" else 128, dtype=torch.bfloat16) * 0.02
    weights = torch.randn(69, 2, dtype=torch.bfloat16) if is_moe else None
    outputs = torch.randn((138, 128) if is_moe else (1, 69, 128), dtype=torch.bfloat16) * 0.02
    cotangent = torch.randn_like(residual)
    sources = (outputs, weights, residual)
    indexes = torch.arange(138) if is_moe else None
    compiled = type(layer).forward_stage5
    eager = compiled.__wrapped__
    actual, gradients = value_and_grads(compiled, layer, sources, indexes, cotangent)
    expected, target_gradients = value_and_grads(eager, layer, sources, indexes, cotangent)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for gradient, target in zip(gradients, target_gradients, strict=True):
        torch.testing.assert_close(gradient, target, rtol=0, atol=0)


def controlled_stage1_type():
    namespace = dict(torch=torch, nn=nn, F=F)
    path = ROOT / "pithtrain/models/qwen3_omni_moe.py"
    tree = ast.parse(path.read_text())
    for name in ("Qwen3OmniMoeThinkerTextRMSNorm", "Qwen3OmniMoeThinkerTextTopKRouter"):
        node = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name
        )
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    decoder = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Qwen3OmniMoeThinkerTextDecoderLayer"
    )
    method = next(
        node
        for node in decoder.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward_stage1_compute"
    )
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)

    class FixedAttention(nn.Module):
        def __init__(self, value):
            super().__init__()
            self.value = nn.Parameter(value)

        def forward(self, _hidden, _rotary, _cu):
            return self.value

    class ControlledLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.is_moe = True
            self.input_layernorm = nn.Identity()
            self.self_attn = FixedAttention(torch.randn(1, 69, 128, dtype=torch.bfloat16) * 0.02)
            self.post_attention_layernorm = namespace["Qwen3OmniMoeThinkerTextRMSNorm"](
                128, 1e-6
            ).to(torch.bfloat16)
            self.mlp = nn.Module()
            self.mlp.gate = namespace["Qwen3OmniMoeThinkerTextTopKRouter"](
                SimpleNamespace(
                    hidden_size=128, num_experts_per_tok=2, num_experts=4, norm_topk_prob=True
                )
            ).to(torch.bfloat16)
            nn.init.normal_(self.mlp.gate.weight, std=0.02)

    ControlledLayer.forward_stage1_compute = namespace["forward_stage1_compute"]
    return ControlledLayer


@pytest.mark.parametrize("is_moe", [False, True])
def test_compiled_stage1_preserves_bf16_residual_norm_boundaries(is_moe):
    torch.manual_seed(739)
    layer = controlled_stage1_type()()
    layer.is_moe = is_moe
    source = torch.randn(1, 69, 128, dtype=torch.bfloat16) * 0.02
    probes = (torch.randn_like(source), torch.randn_like(source))
    # Only residual/norm outputs enter this controlled objective. Softmax's
    # FP32 reductions can differ between eager and compiled backends even when
    # explicit BF16 casts are retained; the full GPU parameter gate covers them.
    parameters = (layer.self_attn.value, layer.post_attention_layernorm.weight)
    actual_input, expected_input = [source.detach().clone().requires_grad_() for _ in range(2)]
    actual = layer.forward_stage1_compute(actual_input, None)
    expected = layer.forward_stage1_compute.__wrapped__(layer, expected_input, None)
    for value, target in zip(actual[:2], expected[:2], strict=True):
        torch.testing.assert_close(value, target, rtol=0, atol=0)
    objectives = [
        sum(
            (value.float() * probe.float()).sum()
            for value, probe in zip(result[:2], probes, strict=True)
        )
        for result in (actual, expected)
    ]
    grads = torch.autograd.grad(objectives[0], (actual_input, *parameters))
    targets = torch.autograd.grad(objectives[1], (expected_input, *parameters))
    for gradient, target in zip(grads, targets, strict=True):
        torch.testing.assert_close(gradient, target, rtol=0, atol=0)
