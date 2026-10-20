"""Compile the production text stage5 with CPU Dynamo/Inductor.

Extracting these pure methods avoids unrelated CUDA-only kernel imports. The
production bodies and fullgraph decorators are executed without replacement.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def layer_type(ep):
    source = Path(__file__).parents[1] / "pithtrain/models/qwen3_omni_moe.py"
    tree = ast.parse(source.read_text())
    name = "Qwen3OmniMoeThinkerTextDecoderLayer"
    definition = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    namespace = dict(torch=torch, distributed=SimpleNamespace(ep_size=ep))
    cls = type(name, (torch.nn.Module,), {})
    for method in ("_forward_stage5", "forward_stage5"):
        node = next(
            n for n in definition.body if isinstance(n, ast.FunctionDef) and n.name == method
        )
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        setattr(cls, method, namespace[method])
    return cls


@pytest.mark.parametrize("is_moe,ep", [(False, 1), (True, 1), (True, 2)])
def test_text_stage5_fullgraph_output_and_all_gradients(is_moe, ep):
    torch.manual_seed(19)
    layer = layer_type(ep)()
    layer.is_moe = is_moe
    residual = torch.randn(1, 4, 8, requires_grad=True)
    weights = torch.randn(4, 2, requires_grad=True) if is_moe else None
    indexes = torch.tensor([6, 1, 4, 3, 2]) if ep == 2 else torch.arange(8)
    output = (
        torch.randn(len(indexes), 8, requires_grad=True)
        if is_moe
        else torch.randn_like(residual, requires_grad=True)
    )
    actual = layer.forward_stage5(output, indexes if is_moe else None, weights, residual)
    if is_moe:
        # Independent per-token weighted sum also covers sparse, unordered EP2
        # assignments and tokens with no local experts.
        contributions = []
        for token in range(4):
            selected = indexes // 2 == token
            contributions.append(
                (output[selected] * weights.reshape(-1)[indexes[selected], None]).sum(0)
            )
        expected = residual + torch.stack(contributions).reshape_as(residual)
    else:
        expected = residual + output
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
    values = (residual, output, weights) if is_moe else (residual, output)
    actual_grads = torch.autograd.grad(actual.square().sum(), values, retain_graph=True)
    expected_grads = torch.autograd.grad(expected.square().sum(), values)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=2e-6)
