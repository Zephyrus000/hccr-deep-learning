"""Checks for state preservation and exact arithmetic intervention semantics."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from run_precision_forensics import arithmetic, bypass_qdq, state_digest, write_json
from localize_stem_precision import separate_onnx_head_backbone


class PrecisionForensicsTests(unittest.TestCase):
    def test_atomic_receipt_survives_a_transient_reader_lock(self):
        import json
        import os
        replace = os.replace
        calls = []
        def temporarily_locked(source, destination):
            calls.append(source)
            if len(calls) == 1:
                raise PermissionError("transient Windows reader lock")
            return replace(source, destination)
        owned = Path(__file__).resolve().parents[1]/"artifacts/local-post-training/remaining-work-20261026/precision"
        owned.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=owned) as directory:
            path = Path(directory)/"receipt.json"
            path.write_text('{"old": true}')
            with patch("run_precision_forensics.os.replace", temporarily_locked):
                write_json(path, {"complete": [1, 2, 3]})
            self.assertEqual(json.loads(path.read_text()), {"complete": [1, 2, 3]})
            self.assertEqual(len(calls), 2)
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_weight_rounding_changes_output_without_state_change(self):
        model = nn.Sequential(nn.Conv2d(1, 1, 1, bias=False), nn.BatchNorm2d(1)).eval()
        with torch.no_grad():
            model[0].weight.fill_(1.003)
            model[1].running_mean.fill_(0.25)
            model[1].running_var.fill_(0.5)
        before = copy.deepcopy(model.state_dict())
        digest = state_digest(model)
        x = torch.full((1, 1, 2, 2), 2.)
        with arithmetic(model, "conv_weights_round_bf16"), torch.inference_mode():
            output = model(x)
        expected = torch.full_like(output, (2.-0.25)/(0.5+1e-5)**0.5)
        torch.testing.assert_close(output, expected)
        self.assertEqual(digest, state_digest(model))
        for name, tensor in model.state_dict().items():
            self.assertTrue(torch.equal(tensor, before[name]))
        with torch.inference_mode():
            original = model(x)
        self.assertFalse(torch.equal(original, output))

    def test_forward_restored_even_after_exception(self):
        model = nn.Conv2d(1, 1, 1, bias=False)
        original = model.forward
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with arithmetic(model, "conv_outputs_round_bf16"):
                raise RuntimeError("intentional")
        self.assertEqual(original, model.forward)

    def test_qdq_bypass_reuses_exact_initializer_and_keeps_source_graph(self):
        import onnx
        from onnx import helper, numpy_helper, TensorProto
        w = numpy_helper.from_array(np.array([1.003], dtype=np.float32), "w")
        fp = helper.make_model(helper.make_graph([], "fp", [], [], [w]))
        q = helper.make_model(helper.make_graph([
            helper.make_node("QuantizeLinear", ["x", "s", "z"], ["xq"], name="xq"),
            helper.make_node("DequantizeLinear", ["xq", "s", "z"], ["xd"], name="xd"),
            helper.make_node("DequantizeLinear", ["w_quantized", "s", "z"], ["wd"], name="w_DequantizeLinear"),
            helper.make_node("Mul", ["xd", "wd"], ["y"], name="mul")], "q", [], [], [
                numpy_helper.from_array(np.array([100], np.uint8), "w_quantized"),
                numpy_helper.from_array(np.array(0.01, np.float32), "s"),
                numpy_helper.from_array(np.array(0, np.uint8), "z")]))
        before = q.SerializeToString()
        graph, records = bypass_qdq(q, fp, True, True)
        self.assertEqual(list(graph.graph.node[-1].input), ["x", "w"])
        restored = next(t for t in graph.graph.initializer if t.name == "w")
        np.testing.assert_array_equal(numpy_helper.to_array(restored), np.array([1.003], np.float32))
        self.assertEqual(before, q.SerializeToString())
        self.assertEqual(len(records), 2)

    def test_head_backbone_bypass_keeps_classifier_weight_qdq_in_head_group(self):
        from onnx import helper, numpy_helper
        weight = numpy_helper.from_array(np.array([1.003], np.float32), "stem.0.weight")
        fp = helper.make_model(helper.make_graph([], "fp", [], [], [weight]))
        nodes = [
            helper.make_node("QuantizeLinear", ["images", "s", "z"], ["xq"]),
            helper.make_node("DequantizeLinear", ["xq", "s", "z"], ["xd"]),
            helper.make_node("DequantizeLinear", ["wq", "s", "z"], ["wd"], name="stem.0.weight_DequantizeLinear"),
            helper.make_node("Mul", ["xd", "wd"], ["/classifier/Div"]),
            helper.make_node("QuantizeLinear", ["/classifier/Div", "s", "z"], ["hq"]),
            helper.make_node("DequantizeLinear", ["hq", "s", "z"], ["hd"]),
            helper.make_node("QuantizeLinear", ["/classifier/Transpose", "s", "z"], ["cq"]),
            helper.make_node("DequantizeLinear", ["cq", "s", "z"], ["cd"]),
            helper.make_node("Mul", ["hd", "cd"], ["scores"])]
        q = helper.make_model(helper.make_graph(nodes, "q", [], [], [
            numpy_helper.from_array(np.array([100], np.uint8), "wq")]))
        head_exempt, changes = separate_onnx_head_backbone(fp, q, head=True)
        self.assertEqual(list(head_exempt.graph.node[-1].input), ["/classifier/Div", "/classifier/Transpose"])
        self.assertEqual(list(head_exempt.graph.node[3].input), ["xd", "wd"])
        self.assertEqual([r["kind"] for r in changes], ["head_runtime", "head_runtime"])
        backbone_exempt, changes = separate_onnx_head_backbone(fp, q, backbone=True)
        self.assertEqual(list(backbone_exempt.graph.node[3].input), ["images", "stem.0.weight"])
        self.assertEqual(list(backbone_exempt.graph.node[-1].input), ["hd", "cd"])
        self.assertEqual({r["kind"] for r in changes}, {"backbone_runtime", "backbone_exact_weight"})


if __name__ == "__main__":
    unittest.main()
