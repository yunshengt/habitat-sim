#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Opt-in glTF alpha cutout (SimulatorConfiguration.enable_alpha_mask)."""

import io
import json
import math
import struct

import numpy as np
import pytest

import habitat_sim

magnum = pytest.importorskip("magnum")
Image = pytest.importorskip("PIL.Image")

RED, GREEN, BLUE = (220, 30, 30), (71, 112, 76), (30, 60, 220)


def _png(rgba: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return buf.getvalue()


def _write_glb(path, quads):
    """quads: list of (y, half_size, rgba texture, alphaMode). Z-up file, so a quad
    in the x/z plane at y=d sits at habitat z=-d in front of a camera at the origin."""
    blobs, views, accs, meshes, images, mats = [], [], [], [], [], []

    def add(raw: bytes, target=None):
        off = sum(len(b) for b in blobs)
        blobs.append(raw + b"\0" * (-len(raw) % 4))
        view = {"buffer": 0, "byteOffset": off, "byteLength": len(raw)}
        if target:
            view["target"] = target
        views.append(view)
        return len(views) - 1

    for i, (y, h, tex, mode) in enumerate(quads):
        pos = np.array([[-h, y, -h], [h, y, -h], [h, y, h], [-h, y, h]], "<f4")
        uv = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], "<f4")
        idx = np.array([0, 1, 2, 0, 2, 3], "<u2")
        accs.append({"bufferView": add(pos.tobytes(), 34962), "componentType": 5126, "count": 4,
                     "type": "VEC3", "min": pos.min(0).tolist(), "max": pos.max(0).tolist()})
        accs.append({"bufferView": add(uv.tobytes(), 34962), "componentType": 5126, "count": 4, "type": "VEC2"})
        accs.append({"bufferView": add(idx.tobytes(), 34963), "componentType": 5123, "count": 6, "type": "SCALAR"})
        images.append({"bufferView": add(_png(tex)), "mimeType": "image/png"})
        mat = {"alphaMode": mode, "doubleSided": True,
               "pbrMetallicRoughness": {"baseColorTexture": {"index": i}, "metallicFactor": 0.0}}
        if mode == "MASK":
            mat["alphaCutoff"] = 0.5
        mats.append(mat)
        meshes.append({"primitives": [{"attributes": {"POSITION": 3 * i, "TEXCOORD_0": 3 * i + 1},
                                       "indices": 3 * i + 2, "material": i}]})
    gltf = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": list(range(len(quads)))}],
            "nodes": [{"mesh": i} for i in range(len(quads))], "meshes": meshes, "accessors": accs,
            "bufferViews": views, "images": images, "textures": [{"source": i} for i in range(len(quads))],
            "materials": mats, "buffers": [{"byteLength": sum(len(b) for b in blobs)}]}
    js = json.dumps(gltf).encode()
    js += b" " * (-len(js) % 4)
    bn = b"".join(blobs)
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, 28 + len(js) + len(bn))
                     + struct.pack("<I4s", len(js), b"JSON") + js
                     + struct.pack("<I4s", len(bn), b"BIN\0") + bn)
    return str(path)


def _render(scene, alpha, *, lit=False, cutoff=0.5, dist=0.0, size=200, hfov=60.0):
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = scene
    cfg.enable_physics = False
    cfg.enable_alpha_mask = alpha
    cfg.alpha_blend_cutoff = cutoff
    if lit:
        cfg.scene_light_setup = habitat_sim.gfx.DEFAULT_LIGHTING_KEY
        cfg.override_scene_light_defaults = True
    specs = []
    for uuid, kind in (("rgb", habitat_sim.SensorType.COLOR), ("depth", habitat_sim.SensorType.DEPTH)):
        s = habitat_sim.CameraSensorSpec()
        s.uuid, s.sensor_type, s.resolution, s.hfov = uuid, kind, [size, size], hfov
        s.position = [0.0, 0.0, 0.0]
        s.clear_color = magnum.Color4(1.0, 1.0, 1.0, 1.0)
        specs.append(s)
    agent = habitat_sim.agent.AgentConfiguration()
    agent.sensor_specifications = specs
    with habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent])) as sim:
        state = sim.get_agent(0).get_state()
        state.position = np.array([0.0, 0.0, dist], np.float32)
        state.rotation = np.array([0, 0, 0, 1], np.float32)
        sim.get_agent(0).set_state(state)
        obs = sim.get_sensor_observations()
        return obs["rgb"][:, :, :3].astype(int), obs["depth"]


def _half_transparent(mode_alpha_left=0):
    tex = np.zeros((64, 64, 4), np.uint8)
    tex[:, :32] = (*GREEN, mode_alpha_left)
    tex[:, 32:] = (*BLUE, 255)
    return tex


def _red():
    tex = np.zeros((4, 4, 4), np.uint8)
    tex[:] = (*RED, 255)
    return tex


@pytest.fixture(scope="module")
def two_quads(tmp_path_factory):
    d = tmp_path_factory.mktemp("alpha")
    return {mode: _write_glb(d / f"{mode}.glb", [(3.0, 1.0, _red(), "OPAQUE"),
                                                  (2.0, 1.0, _half_transparent(), mode)])
            for mode in ("OPAQUE", "BLEND", "MASK")}


@pytest.mark.gfxtest
@pytest.mark.parametrize("lit", [False, True])
@pytest.mark.parametrize("mode", ["BLEND", "MASK"])
def test_alpha_cutout_shows_what_is_behind(two_quads, mode, lit):
    rgb_off, depth_off = _render(two_quads[mode], False, lit=lit)
    rgb_on, depth_on = _render(two_quads[mode], True, lit=lit)
    left, right = (slice(70, 130), slice(65, 95)), (slice(70, 130), slice(105, 135))
    # Off: alpha ignored, the colour stored under transparent texels shows.
    assert rgb_off[left].mean((0, 1))[1] > rgb_off[left].mean((0, 1))[0]
    # On: the red quad behind shows through, colour and depth.
    assert rgb_on[left].mean((0, 1))[0] > rgb_on[left].mean((0, 1))[1] + 30
    assert np.median(depth_on[left]) == pytest.approx(3.0, abs=0.05)
    assert np.median(depth_off[left]) == pytest.approx(2.0, abs=0.05)
    # The opaque half is untouched.
    np.testing.assert_array_equal(rgb_off[right], rgb_on[right])


@pytest.mark.gfxtest
def test_opaque_material_unaffected(two_quads):
    rgb_off, _ = _render(two_quads["OPAQUE"], False)
    rgb_on, _ = _render(two_quads["OPAQUE"], True)
    np.testing.assert_array_equal(rgb_off, rgb_on)


@pytest.mark.gfxtest
@pytest.mark.parametrize("cutoff", [0.25, 0.75])
def test_blend_cutoff_moves_threshold(tmp_path, cutoff):
    ramp = np.zeros((64, 64, 4), np.uint8)
    ramp[..., :3] = GREEN
    ramp[..., 3] = np.linspace(0, 255, 64).round().astype(np.uint8)[None, :]
    scene = _write_glb(tmp_path / "ramp.glb", [(3.0, 2.0, _red(), "OPAQUE"), (2.0, 1.0, ramp, "BLEND")])
    rgb, _ = _render(scene, True, cutoff=cutoff)
    # Front quad spans x in [-1, 1] at 2 m: columns where red shows through ~ alpha < cutoff.
    half = int(round(100 * 1.0 / (2.0 * math.tan(math.radians(30)))))
    row = rgb[100, 100 - half:100 + half]
    red_share = float((row[:, 0] > row[:, 1] + 40).mean())
    assert red_share == pytest.approx(cutoff, abs=0.1)


@pytest.mark.gfxtest
def test_sparse_texture_survives_at_distance(tmp_path):
    rng = np.random.RandomState(0)
    tex = np.zeros((1024, 1024, 4), np.uint8)
    tex[..., :3] = (40, 160, 60)
    tex[..., 3] = np.kron(rng.rand(128, 128) < 0.08, np.ones((8, 8))).astype(np.uint8) * 255
    scene = _write_glb(tmp_path / "dots.glb", [(0.0, 2.0, tex, "BLEND")])
    # 4 m quad at 128 m covers ~44 px: plain box-filtered mips average the 8% dots
    # below the cutoff and the quad vanishes; coverage-preserving mips keep it.
    rgb, _ = _render(scene, True, dist=128.0, size=512, hfov=20.0)
    r = 22
    crop = rgb[256 - r:256 + r, 256 - r:256 + r]
    assert float((crop < 245).any(-1).mean()) > 0.04
