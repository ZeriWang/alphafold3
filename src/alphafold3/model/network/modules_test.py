"""Tests for Pairformer triangle acceleration adapters."""

import types

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alphafold3.model import model_config
from alphafold3.model.network import modules


class _FakeCue:

  def __init__(self):
    self.triangle_multiplication_calls = []
    self.triangle_attention_calls = []

  def triangle_multiplicative_update(self, **kwargs):
    self.triangle_multiplication_calls.append(kwargs)
    return jnp.zeros_like(kwargs['x'])

  def triangle_attention(self, q, k, v, bias, mask, scale):
    self.triangle_attention_calls.append(
        types.SimpleNamespace(
            q=q, k=k, v=v, bias=bias, mask=mask, scale=scale
        )
    )
    return jnp.zeros_like(q), None, None


def _global_config(*, flash_attention_implementation='triton'):
  return model_config.GlobalConfig(
      bfloat16='none',
      final_init='zeros',
      flash_attention_implementation=flash_attention_implementation,
  )


def test_pairformer_cue_flags_default_to_legacy_path():
  config = modules.PairFormerIteration.Config(num_layer=1)

  assert not config.use_cue_triangle_multiplication
  assert not config.use_cue_triangle_attention
  assert not hasattr(
      modules.EvoformerIteration.Config(num_layer=1),
      'use_cue_triangle_multiplication',
  )


def test_pairformer_default_path_does_not_require_cue(monkeypatch):
  def fail_if_called(*, flag):
    raise AssertionError(f'cuEq loader unexpectedly called for {flag}')

  monkeypatch.setattr(modules, '_require_cuequivariance_jax', fail_if_called)
  config = modules.PairFormerIteration.Config(num_layer=1)
  act = jnp.ones((4, 4, 32), dtype=jnp.float32)
  pair_mask = jnp.ones((4, 4), dtype=jnp.float32)

  def forward(x, m):
    return modules.PairFormerIteration(
        config,
        _global_config(flash_attention_implementation='xla'),
        name='pairformer',
    )(x, m)

  transformed = hk.transform(forward)
  params = transformed.init(jax.random.PRNGKey(0), act, pair_mask)
  out = transformed.apply(params, None, act, pair_mask)

  assert out.shape == act.shape


@pytest.mark.parametrize(
    ('equation', 'direction'),
    [
        ('ikc,jkc->ijc', 'outgoing'),
        ('kjc,kic->ijc', 'incoming'),
    ],
)
@pytest.mark.parametrize('dtype', [jnp.float32, jnp.bfloat16])
def test_cue_triangle_multiplication_layout(monkeypatch, equation, direction, dtype):
  fake = _FakeCue()
  monkeypatch.setattr(
      modules, '_require_cuequivariance_jax', lambda *, flag: fake
  )
  config = modules.TriangleMultiplication.Config(equation=equation)
  act = jnp.ones((4, 4, 8), dtype=dtype)
  mask = jnp.array(
      [[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [0, 0, 0, 0]],
      dtype=dtype,
  )

  def forward(x, m):
    return modules.CueTriangleMultiplication(
        config, _global_config(), name='triangle'
    )(x, m)

  transformed = hk.transform(forward)
  params = transformed.init(jax.random.PRNGKey(0), act, mask)
  fake.triangle_multiplication_calls.clear()

  out = transformed.apply(params, None, act, mask)

  call = fake.triangle_multiplication_calls[-1]
  assert out.shape == act.shape
  assert call['direction'] == direction
  assert call['mask'].shape == mask.shape
  assert call['mask'].dtype == act.dtype
  assert call['p_in_weight'].shape == (16, 8)
  assert call['g_in_weight'].shape == (16, 8)
  assert call['p_out_weight'].shape == (8, 8)
  assert call['g_out_weight'].shape == (8, 8)
  assert call['norm_in_weight'].shape == (8,)
  assert call['norm_out_weight'].shape == (8,)
  if dtype == jnp.bfloat16:
    assert call['norm_in_weight'].dtype == jnp.float32


@pytest.mark.parametrize('transpose', [False, True])
def test_cue_grid_self_attention_layout(monkeypatch, transpose):
  fake = _FakeCue()
  monkeypatch.setattr(
      modules, '_require_cuequivariance_jax', lambda *, flag: fake
  )
  config = modules.GridSelfAttention.Config(num_head=2)
  act = jnp.ones((4, 4, 32), dtype=jnp.float32)
  pair_mask = jnp.array(
      [[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [0, 0, 0, 0]],
      dtype=jnp.float32,
  )

  def forward(x, m):
    return modules.CueGridSelfAttention(
        config, _global_config(), transpose=transpose, name='attention'
    )(x, m)

  transformed = hk.transform(forward)
  params = transformed.init(jax.random.PRNGKey(0), act, pair_mask)
  fake.triangle_attention_calls.clear()

  out = transformed.apply(params, None, act, pair_mask)

  call = fake.triangle_attention_calls[-1]
  assert out.shape == act.shape
  assert call.q.shape == (1, 4, 2, 4, 16)
  assert call.k.shape == (1, 4, 2, 4, 16)
  assert call.v.shape == (1, 4, 2, 4, 16)
  assert call.bias.shape == (1, 1, 2, 4, 4)
  assert call.mask.shape == (1, 4, 1, 1, 4)
  assert call.mask.dtype == jnp.bool_
  assert call.scale == 16 ** -0.5


def test_cue_triangle_attention_layout_helpers():
  q = jnp.zeros((3, 5, 2, 7))
  k = jnp.zeros((3, 5, 2, 7))
  v = jnp.zeros((3, 5, 2, 7))
  bias = jnp.zeros((2, 5, 5))
  mask = jnp.ones((3, 1, 1, 5))

  q, k, v, bias, mask = modules._cue_triangle_attention_inputs(
      q, k, v, bias, mask
  )
  out = modules._cue_triangle_attention_output(q)

  assert q.shape == (1, 3, 2, 5, 7)
  assert k.shape == (1, 3, 2, 5, 7)
  assert v.shape == (1, 3, 2, 5, 7)
  assert bias.shape == (1, 1, 2, 5, 5)
  assert mask.shape == (1, 3, 1, 1, 5)
  assert mask.dtype == jnp.bool_
  assert out.shape == (3, 5, 2, 7)


def test_cue_dependency_error_mentions_flag(monkeypatch):
  def raise_os_error(_):
    raise OSError('broken CUDA shared library')

  monkeypatch.setattr(modules.importlib, 'import_module', raise_os_error)

  with pytest.raises(ImportError, match='use_cue_triangle_attention=True'):
    modules._require_cuequivariance_jax(flag='use_cue_triangle_attention')


def _real_cue_gpu_available():
  try:
    modules._require_cuequivariance_jax(
        flag='use_cue_triangle_multiplication'
    )
  except ImportError:
    return False
  return any(device.platform == 'gpu' for device in jax.devices())


@pytest.mark.skipif(
    not _real_cue_gpu_available(),
    reason='requires working cuequivariance_jax CUDA ops and a JAX GPU device',
)
def test_cue_triangle_multiplication_forward_and_input_grad_sanity():
  config = modules.TriangleMultiplication.Config(equation='ikc,jkc->ijc')
  act = jax.random.normal(jax.random.PRNGKey(1), (4, 4, 8))
  mask = jnp.array(
      [[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [0, 0, 0, 0]],
      dtype=act.dtype,
  )

  def old_forward(x, m):
    return modules.TriangleMultiplication(
        config, _global_config(), name='triangle'
    )(x, m)

  def cue_forward(x, m):
    return modules.CueTriangleMultiplication(
        config, _global_config(), name='triangle'
    )(x, m)

  old = hk.transform(old_forward)
  cue = hk.transform(cue_forward)
  params = old.init(jax.random.PRNGKey(0), act, mask)
  old_out = old.apply(params, None, act, mask)
  cue_out = cue.apply(params, None, act, mask)

  np.testing.assert_allclose(cue_out, old_out, rtol=5e-3, atol=5e-3)

  def old_loss(x):
    return jnp.sum(old.apply(params, None, x, mask).astype(jnp.float32))

  def cue_loss(x):
    return jnp.sum(cue.apply(params, None, x, mask).astype(jnp.float32))

  old_grad = jax.grad(old_loss)(act)
  cue_grad = jax.grad(cue_loss)(act)
  assert jnp.all(jnp.isfinite(cue_grad))
  np.testing.assert_allclose(cue_grad, old_grad, rtol=5e-3, atol=5e-3)
