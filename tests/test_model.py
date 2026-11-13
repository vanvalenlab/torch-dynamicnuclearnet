"""Architecture tests for ``PanopticNet`` and its modules (DynamicNuclearNet config).

The FPN wiring tests use a tiny pyramid and write the expected computation out
by hand, following deepcell-tf ``model_zoo/fpn.py``. The full-network tests
build the real model (EfficientNetV2-L backbone) on CPU (a few seconds each).
"""

import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import StochasticDepth

from torch_dnn.model import (ARCH_VERSION, DNN_MODEL_KWARGS, DNN_OUTPUT_CHANNELS, PanopticNet,
                             checkpoint_arch_version)
from torch_dnn.modules import BackboneNetwork, FeaturePyramidNetwork, SemanticHead

OLD_TORCH_DNN_WEIGHTS = Path.home() / '.deepcell' / 'models' / 'dnn' / 'saved_model_best_dict.pth'

# Parameter counts, recorded so that an unintended architecture change shows up.
# Backbone: torchvision EfficientNetV2-L features (117,234,284). FPN: the five lateral
# 1x1 convs (C1-C5) plus the P1 3x3 conv, the only pyramid output the heads read.
# Heads: 3 x (Conv 3x3 256->64, Conv 1x1 64->128, BN) + output convs for 1+1+2 classes.
N_PARAMS = 119_089_072
N_PARAMS_FPN = 1_385_984
N_PARAMS_HEADS = 468_804


def up(x):
    return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)


@pytest.fixture
def tiny_pyramid():
    torch.manual_seed(0)
    channels = {'C3': 8, 'C4': 16, 'C5': 32}
    feats = {'C3': torch.randn(2, 8, 16, 16), 'C4': torch.randn(2, 16, 8, 8), 'C5': torch.randn(2, 32, 4, 4)}
    return channels, feats


def build_fpn(channels, **kwargs):
    torch.manual_seed(1)
    return FeaturePyramidNetwork(levels=['P3', 'P4', 'P5', 'P6', 'P7'], backbone_levels=['C3', 'C4', 'C5'],
                                 feature_size=4, backbone_channels=channels, **kwargs)


def test_fpn_top_down_passes_merged_lateral(tiny_pyramid):
    channels, c = tiny_pyramid
    fpn = build_fpn(channels)
    lat = {k: fpn.levels[k].lateral_conv for k in ('P3', 'P4', 'P5')}
    smooth = {k: fpn.levels[k].smooth_conv for k in ('P3', 'P4', 'P5')}
    with torch.no_grad():
        out = fpn(c)
        m5 = lat['P5'](c['C5'])
        m4 = lat['P4'](c['C4']) + up(m5)
        m3 = lat['P3'](c['C3']) + up(m4)
        p6 = fpn.crown['P6'](c['C5'])                  # P6 comes from C5
        expected = {'P3': smooth['P3'](m3), 'P4': smooth['P4'](m4), 'P5': smooth['P5'](m5),
                    'P6': p6, 'P7': fpn.crown['P7'](torch.relu(p6))}  # ReLU before P7
    assert out.keys() == expected.keys()
    for k in expected:
        torch.testing.assert_close(out[k], expected[k], msg=k)


def test_fpn_output_levels_finest_only(tiny_pyramid):
    """The FPN builds only what the finest level needs: every lateral, but no 3x3 conv on
    the coarser levels and no crown. The finest level equals the full pyramid's."""
    channels, c = tiny_pyramid
    full = build_fpn(channels)
    finest = build_fpn(channels, output_levels=['P3'])
    assert {k: v.smooth_conv is not None for k, v in finest.levels.items()} == {'P5': False, 'P4': False, 'P3': True}
    assert len(finest.crown) == 0
    finest.load_state_dict(full.state_dict(), strict=False)
    with torch.no_grad():
        out = finest(c)
        assert out.keys() == {'P3'}
        torch.testing.assert_close(out['P3'], full(c)['P3'])


def test_fpn_output_levels_must_be_known():
    with pytest.raises(ValueError, match='output_levels'):
        FeaturePyramidNetwork(levels=['P3', 'P4', 'P5'], backbone_channels={'C3': 8, 'C4': 16, 'C5': 32},
                              output_levels=['P2'])


def test_crown_levels_sorted_numerically():
    channels = {'C3': 8, 'C4': 16, 'C5': 32}
    levels = [f'P{i}' for i in range(3, 11)]
    fpn = FeaturePyramidNetwork(levels=levels, backbone_channels=channels, feature_size=4)
    assert fpn.crown_levels == ['P6', 'P7', 'P8', 'P9', 'P10']


def test_crown_levels_must_be_contiguous():
    with pytest.raises(ValueError, match='contiguous'):
        FeaturePyramidNetwork(levels=['P3', 'P4', 'P5', 'P7'], backbone_channels={'C3': 8, 'C4': 16, 'C5': 32})


@pytest.mark.parametrize('n_classes, activation', [(1, nn.ReLU), (2, nn.Softmax)])
def test_semantic_head_matches_deepcell_for_p1(n_classes, activation):
    """deepcell's PanopticNet with C1-C5 builds P1-P7, so target_level = 1 and each head
    is semantic_upsample with n_upsample=1: Conv 3x3 64 (linear) -> bilinear 2x, then
    Conv 1x1 128 -> BN -> ReLU -> Conv 1x1 n_classes -> ReLU (1 class) / softmax."""
    head = SemanticHead(n_classes=n_classes, level='P1', feature_size=256, n_upsample=1)
    assert [type(m) for m in head.upsample] == [nn.Conv2d, nn.Upsample]
    conv = head.upsample[0]
    assert (conv.in_channels, conv.out_channels, conv.kernel_size) == (256, 64, (3, 3))
    assert head.upsample[1].mode == 'bilinear'
    assert (head.dense_conv.in_channels, head.dense_conv.out_channels) == (64, 128)
    assert head.output_conv.out_channels == n_classes
    assert isinstance(head.final_activation, activation)
    assert sum(p.numel() for p in head.parameters()) == 147_520 + 8_320 + 256 + 129 * n_classes
    with torch.no_grad():
        out = head.eval()({'P1': torch.rand(2, 256, 64, 64)})
    assert out.shape == (2, n_classes, 128, 128)


def test_dnn_model_kwargs():
    """The deepcell notebook's PanopticNet config, which train.py and the app both use."""
    assert DNN_MODEL_KWARGS == {
        'backbone': 'efficientnetv2bl',
        'crop_size': 256,
        'in_channels': 1,
        'backbone_levels': ['C1', 'C2', 'C3', 'C4', 'C5'],
        'pyramid_levels': ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7'],
        'n_semantic_classes': [1, 1, 2],
    }
    assert DNN_OUTPUT_CHANNELS == 4


@pytest.fixture(scope='module')
def model():
    torch.manual_seed(0)
    return PanopticNet(**DNN_MODEL_KWARGS, use_imagenet=False).eval()


@pytest.mark.slow
def test_defaults_are_dnn():
    torch.manual_seed(0)
    a = PanopticNet(use_imagenet=False)
    for k, v in DNN_MODEL_KWARGS.items():
        assert getattr(a, k) == v, k


@pytest.mark.slow
def test_forward_contract(model):
    """(B, 1, 256, 256) -> (B, 4, 256, 256): inner distance, outer distance (ReLU),
    then fgbg softmax over 2 classes."""
    with torch.inference_mode():
        out = model(torch.rand(2, 1, 256, 256))
    assert out.shape == (2, 4, 256, 256)
    torch.testing.assert_close(out[:, 2:4].sum(dim=1), torch.ones(2, 256, 256))
    assert (out[:, :2] >= 0).all()


@pytest.mark.slow
def test_heads_read_p1_only(model):
    assert [h.level for h in model.semantic_heads] == ['P1', 'P1', 'P1']
    assert model.fpn.output_levels == ['P1']
    assert {k: v.smooth_conv is not None for k, v in model.fpn.levels.items()} == \
        {'P5': False, 'P4': False, 'P3': False, 'P2': False, 'P1': True}
    assert len(model.fpn.crown) == 0
    assert model.min_size == 128


@pytest.mark.slow
def test_channels_last_matches_channels_first(model):
    x = torch.rand(1, 1, 256, 256)
    with torch.inference_mode():
        torch.testing.assert_close(model(x), model(x.permute(0, 2, 3, 1), format='channels_last'))
    with pytest.raises(ValueError, match='format'):
        model(x, format='channel_first')


@pytest.mark.slow
def test_parameter_count(model):
    assert sum(p.numel() for p in model.parameters()) == N_PARAMS
    assert sum(p.numel() for p in model.fpn.parameters()) == N_PARAMS_FPN
    assert sum(p.numel() for p in model.semantic_heads.parameters()) == N_PARAMS_HEADS


@pytest.mark.slow
@pytest.mark.parametrize('shape', [(1, 1, 256, 128), (1, 1, 200, 200), (1, 1, 64, 64), (1, 2, 256, 256),
                                   (2, 256, 256)])
def test_input_validation(model, shape):
    with pytest.raises(ValueError):
        model(torch.rand(*shape))


def test_crop_size_validation():
    with pytest.raises(ValueError, match='power-of-2'):
        PanopticNet(use_imagenet=False, crop_size=200)


@pytest.mark.slow
def test_channel_conv_always_applied(model):
    """The 1x1 channel conv takes the image + 2 location channels and is always applied."""
    assert model.channel_conv.in_channels == 3
    calls = []
    handle = model.channel_conv.register_forward_hook(lambda *a: calls.append(1))
    try:
        with torch.inference_mode():
            model(torch.rand(1, 1, 128, 128))
    finally:
        handle.remove()
    assert calls == [1]


@pytest.mark.slow
def test_interpolation_reaches_heads():
    net = PanopticNet(use_imagenet=False, interpolation='nearest', crop_size=128)
    modes = {m.mode for m in net.modules() if isinstance(m, nn.Upsample)}
    assert modes == {'nearest'}


@pytest.mark.slow
def test_keras_batchnorm_and_init(model):
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            # torchvision's EfficientNetV2 already uses Keras's values (eps 1e-3, Keras momentum
            # 0.9 = PyTorch 0.1); the heads use Keras's BatchNormalization defaults.
            expected = (1e-3, 0.1) if name.startswith('bbnetwork.') else (1e-3, 0.01)
            assert (m.eps, m.momentum) == expected, name
    for part in (model.channel_conv, model.fpn, model.semantic_heads):
        for m in part.modules():
            if isinstance(m, nn.Conv2d):
                k = m.kernel_size[0] * m.kernel_size[1]
                bound = math.sqrt(6 / (m.in_channels * k + m.out_channels * k))  # Glorot uniform
                assert m.weight.abs().max() <= bound
                assert m.weight.abs().max() > 0.9 * bound
                assert (m.bias == 0).all()


@pytest.mark.slow
def test_stochastic_depth_matches_keras(model):
    """Keras EfficientNetV2 drops blocks with rate 0.2 * b / n_blocks; torchvision the same.
    torchvision has 79 such modules; the two stride-2 blocks cut at their expand conv
    (``_SplitMBConv``) have no residual, so their drop never applied and is not kept."""
    ps = [m.p for m in model.modules() if isinstance(m, StochasticDepth)]
    assert len(ps) == 77
    assert ps[0] == 0 and max(ps) < 0.2 and ps == sorted(ps)


@pytest.mark.slow
def test_every_parameter_gets_gradient():
    """No dead layers: one training step's loss reaches every parameter."""
    torch.manual_seed(0)
    net = PanopticNet(use_imagenet=False, crop_size=128).train()
    for m in net.modules():  # random block dropping could leave a block without gradient
        if isinstance(m, StochasticDepth):
            m.eval()
    x = torch.rand(2, 1, 128, 128)
    out = net(x)
    distance_target = torch.rand(2, 2, 128, 128)
    fgbg_target = torch.randint(0, 2, (2, 128, 128))
    loss = F.mse_loss(out[:, :2], distance_target) + F.nll_loss(torch.log(out[:, 2:4] + 1e-7), fgbg_target)
    loss.backward()
    dead = [n for n, p in net.named_parameters() if p.grad is None or not p.grad.abs().sum() > 0]
    assert dead == []


@pytest.mark.slow
def test_checkpoint_version_guard(tmp_path):
    net = PanopticNet(use_imagenet=False, crop_size=128)
    sd = net.state_dict()
    assert checkpoint_arch_version(sd) == ARCH_VERSION
    torch.save(sd, tmp_path / 'w.pth')
    PanopticNet(use_imagenet=False, crop_size=128).load_state_dict(torch.load(tmp_path / 'w.pth'))

    no_version = {k: v for k, v in sd.items() if k != 'arch_version'}
    assert checkpoint_arch_version(no_version) == 1
    with pytest.raises(ValueError, match='original torch_dnn network'):
        PanopticNet(use_imagenet=False, crop_size=128).load_state_dict(no_version)

    other = dict(sd, arch_version=torch.tensor(2))
    with pytest.raises(ValueError, match='does not build'):
        PanopticNet(use_imagenet=False, crop_size=128).load_state_dict(other)


@pytest.mark.slow
def test_checkpoint_of_another_network_is_rejected():
    """A version-3 checkpoint with other layers (here: 2 input channels and Mesmer's heads)
    raises ValueError, not a bare RuntimeError."""
    mesmer_like = PanopticNet(use_imagenet=False, crop_size=128, in_channels=2,
                              n_semantic_classes=[1, 3, 1, 3]).state_dict()
    with pytest.raises(ValueError, match='does not fit'):
        PanopticNet(use_imagenet=False, crop_size=128).load_state_dict(mesmer_like)


@pytest.mark.slow
@pytest.mark.skipif(not OLD_TORCH_DNN_WEIGHTS.exists(), reason='old torch_dnn weights not found')
def test_old_torch_dnn_weights_are_rejected():
    sd = torch.load(OLD_TORCH_DNN_WEIGHTS, map_location='cpu')
    assert checkpoint_arch_version(sd) == 1
    with pytest.raises(ValueError, match='python -m torch_dnn.train'):
        PanopticNet(use_imagenet=False).load_state_dict(sd)


# --- EfficientNetV2-L backbone ---

@pytest.fixture(scope='module')
def effnet():
    torch.manual_seed(0)
    return BackboneNetwork('efficientnetv2bl', use_imagenet=False).eval()


@pytest.mark.slow
def test_efficientnet_taps(effnet):
    """Taps match deepcell's Keras layers: block1b_add, block2c_add, block4a_expand_activation,
    block6a_expand_activation, top_activation (backbone_utils.py:418-432), at strides 2-32."""
    assert effnet.out_channels == {'C1': 32, 'C2': 64, 'C3': 384, 'C4': 1344, 'C5': 1280}
    with torch.inference_mode():
        feats = effnet(torch.rand(1, 3, 128, 128))
    assert {k: v.shape[-1] for k, v in feats.items()} == {'C1': 64, 'C2': 32, 'C3': 16, 'C4': 8, 'C5': 4}


@pytest.mark.slow
def test_efficientnet_split_is_exact():
    """Cutting MBConv blocks at their expand conv must not change the network: C5 equals
    torchvision's own ``features`` forward on the same modules."""
    from torchvision.models import efficientnet_v2_l
    torch.manual_seed(0)
    reference = efficientnet_v2_l().eval()
    torch.manual_seed(0)
    ours = BackboneNetwork('efficientnetv2bl', use_imagenet=False).eval()
    assert sum(v.numel() for v in ours.state_dict().values()) == sum(
        v.numel() for k, v in reference.state_dict().items() if k.startswith('features.'))
    x = torch.rand(1, 3, 128, 128)
    with torch.inference_mode():
        torch.testing.assert_close(ours(x)['C5'], reference.features(x))
