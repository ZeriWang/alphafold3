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
    self.triangle_attention_calls = []

  def triangle_attention(self, q, k, v, bias, mask, scale):
    self.triangle_attention_calls.append(
        types.SimpleNamespace(
            q=q, k=k, v=v, bias=bias, mask=mask, scale=scale
        )
    )
    logits = jnp.einsum('...ai,...bi->...ab', q * scale, k)
    logits += bias
    logits = jnp.where(mask, logits, -1e9)
    weights = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
    out = jnp.einsum('...ab,...bi->...ai', weights.astype(v.dtype), v)
    return out, None, None


class _FakeCueGemm:

  def __init__(self):
    self.sigmoid_gated_dual_gemm_calls = []
    self.sigmoid_gated_dual_gemm_dual_x_calls = []

  def sigmoid_gated_dual_gemm(
      self, x, w1, w2, *, b1=None, b2=None, mask=None, transpose_out=False,
      **unused_kwargs
  ):
    self.sigmoid_gated_dual_gemm_calls.append(
        types.SimpleNamespace(
            x=x, w1=w1, w2=w2, b1=b1, b2=b2, mask=mask,
            transpose_out=transpose_out,
            fallback=unused_kwargs.get('fallback'),
        )
    )
    gate = jnp.einsum('...c,dc->...d', x, w1)
    proj = jnp.einsum('...c,dc->...d', x, w2)
    if b1 is not None:
      gate += b1
    if b2 is not None:
      proj += b2
    out = jax.nn.sigmoid(gate) * proj
    if mask is not None:
      out *= mask[..., None]
    if transpose_out:
      out = jnp.transpose(out, tuple(range(out.ndim - 1, -1, -1)))
    return out

  def sigmoid_gated_dual_gemm_dual_x(
      self, x1, x2, w1, w2, *, b1=None, b2=None, mask=None,
      transpose_out=False, **unused_kwargs
  ):
    self.sigmoid_gated_dual_gemm_dual_x_calls.append(
        types.SimpleNamespace(
            x1=x1, x2=x2, w1=w1, w2=w2, b1=b1, b2=b2, mask=mask,
            transpose_out=transpose_out,
            fallback=unused_kwargs.get('fallback'),
        )
    )
    gate = jnp.einsum('...c,dc->...d', x1, w1)
    proj = jnp.einsum('...c,dc->...d', x2, w2)
    if b1 is not None:
      gate += b1
    if b2 is not None:
      proj += b2
    out = jax.nn.sigmoid(gate) * proj
    if mask is not None:
      out *= mask[..., None]
    if transpose_out:
      out = jnp.transpose(out, tuple(range(out.ndim - 1, -1, -1)))
    return out


def _global_config(*, flash_attention_implementation='triton',
                   final_init='zeros'):
  return model_config.GlobalConfig(
      bfloat16='none',
      final_init=final_init,
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


def test_cue_triangle_gemm_fallback_depends_on_triton_and_tile_shape(
    monkeypatch,
):
  real_import_module = modules.importlib.import_module

  def import_module_with_triton(name):
    if name == 'triton':
      return types.SimpleNamespace()
    return real_import_module(name)

  monkeypatch.setattr(
      modules.importlib, 'import_module', import_module_with_triton
  )

  assert modules._cue_triangle_gemm_fallback(
      input_channels=128, output_channels=256
  )

  monkeypatch.setenv('ALPHAFOLD3_CUE_GEMM_MODE', 'perf')

  assert modules._cue_triangle_gemm_fallback(
      input_channels=8, output_channels=16
  )
  assert modules._cue_triangle_gemm_fallback(
      input_channels=128, output_channels=8
  )
  assert not modules._cue_triangle_gemm_fallback(
      input_channels=128, output_channels=256
  )

  def raise_missing_triton(name):
    if name == 'triton':
      raise ImportError('missing triton')
    return real_import_module(name)

  monkeypatch.setattr(modules.importlib, 'import_module', raise_missing_triton)
  assert modules._cue_triangle_gemm_fallback(
      input_channels=128, output_channels=256
  )


@pytest.mark.parametrize(
    ('equation', 'direction'),
    [
        ('ikc,jkc->ijc', 'outgoing'),
        ('kjc,kic->ijc', 'incoming'),
    ],
)
@pytest.mark.parametrize('dtype', [jnp.float32, jnp.bfloat16])
def test_cue_triangle_multiplication_layout(monkeypatch, equation, direction, dtype):
  del direction
  fake = _FakeCueGemm()
  monkeypatch.setattr(
      modules, '_require_cue_triangle_gemm_ops', lambda *, flag: fake
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
  fake.sigmoid_gated_dual_gemm_calls.clear()
  fake.sigmoid_gated_dual_gemm_dual_x_calls.clear()

  out = transformed.apply(params, None, act, mask)

  in_call = fake.sigmoid_gated_dual_gemm_calls[-1]
  out_call = fake.sigmoid_gated_dual_gemm_dual_x_calls[-1]
  assert out.shape == act.shape
  assert in_call.x.shape == act.shape
  assert in_call.w1.shape == (16, 8)
  assert in_call.w2.shape == (16, 8)
  assert in_call.mask is None
  assert in_call.fallback
  assert out_call.x1.shape == act.shape
  assert out_call.x2.shape == act.shape
  assert out_call.w1.shape == (8, 8)
  assert out_call.w2.shape == (8, 8)
  assert out_call.fallback


@pytest.mark.parametrize(
    'equation', ['ikc,jkc->ijc', 'kjc,kic->ijc']
)
@pytest.mark.parametrize('dtype', [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize('mask_kind', ['full', 'partial'])
def test_cue_triangle_multiplication_matches_legacy_with_fake_helpers(
    monkeypatch, equation, dtype, mask_kind
):
  fake = _FakeCueGemm()
  monkeypatch.setattr(
      modules, '_require_cue_triangle_gemm_ops', lambda *, flag: fake
  )
  config = modules.TriangleMultiplication.Config(equation=equation)
  act = jax.random.normal(jax.random.PRNGKey(1), (4, 4, 8), dtype=dtype)
  if mask_kind == 'full':
    mask = jnp.ones((4, 4), dtype=dtype)
  else:
    mask = jnp.array(
        [[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [0, 0, 0, 0]],
        dtype=dtype,
    )

  def old_forward(x, m):
    return modules.TriangleMultiplication(
        config, _global_config(final_init='linear'), name='triangle'
    )(x, m)

  def cue_forward(x, m):
    return modules.CueTriangleMultiplication(
        config, _global_config(final_init='linear'), name='triangle'
    )(x, m)

  old = hk.transform(old_forward)
  cue = hk.transform(cue_forward)
  params = old.init(jax.random.PRNGKey(0), act, mask)
  old_out = old.apply(params, None, act, mask)
  cue_out = cue.apply(params, None, act, mask)

  atol = 2e-2 if dtype == jnp.bfloat16 else 5e-3
  np.testing.assert_allclose(
      np.asarray(cue_out, dtype=np.float32),
      np.asarray(old_out, dtype=np.float32),
      rtol=5e-3,
      atol=atol,
  )


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


@pytest.mark.parametrize('transpose', [False, True])
def test_cue_grid_self_attention_matches_legacy_with_fake_helpers(
    monkeypatch, transpose
):
  fake = _FakeCue()
  monkeypatch.setattr(
      modules, '_require_cuequivariance_jax', lambda *, flag: fake
  )
  config = modules.GridSelfAttention.Config(num_head=4)
  act = jax.random.normal(jax.random.PRNGKey(2), (4, 4, 32))
  pair_mask = jnp.array(
      [[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [0, 0, 0, 0]],
      dtype=jnp.float32,
  )

  def old_forward(x, m):
    return modules.GridSelfAttention(
        config,
        _global_config(
            flash_attention_implementation='xla', final_init='linear'
        ),
        transpose=transpose,
        name='attention',
    )(x, m)

  def cue_forward(x, m):
    return modules.CueGridSelfAttention(
        config,
        _global_config(final_init='linear'),
        transpose=transpose,
        name='attention',
    )(x, m)

  old = hk.transform(old_forward)
  cue = hk.transform(cue_forward)
  params = old.init(jax.random.PRNGKey(0), act, pair_mask)
  old_out = old.apply(params, None, act, pair_mask)
  cue_out = cue.apply(params, None, act, pair_mask)

  np.testing.assert_allclose(cue_out, old_out, rtol=5e-3, atol=5e-3)


def test_pairformer_iteration_cue_flags_match_legacy_with_fake_helpers(
    monkeypatch,
):
  fake_cue = _FakeCue()
  fake_gemm = _FakeCueGemm()
  monkeypatch.setattr(
      modules, '_require_cuequivariance_jax', lambda *, flag: fake_cue
  )
  monkeypatch.setattr(
      modules, '_require_cue_triangle_gemm_ops', lambda *, flag: fake_gemm
  )
  old_config = modules.PairFormerIteration.Config(num_layer=1)
  cue_config = modules.PairFormerIteration.Config(
      num_layer=1,
      use_cue_triangle_multiplication=True,
      use_cue_triangle_attention=True,
  )
  global_config = _global_config(
      flash_attention_implementation='xla', final_init='linear'
  )
  act = jax.random.normal(jax.random.PRNGKey(3), (4, 4, 32))
  pair_mask = jnp.array(
      [[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [0, 0, 0, 0]],
      dtype=jnp.float32,
  )

  def old_forward(x, m):
    return modules.PairFormerIteration(
        old_config, global_config, name='pairformer'
    )(x, m)

  def cue_forward(x, m):
    return modules.PairFormerIteration(
        cue_config, global_config, name='pairformer'
    )(x, m)

  old = hk.transform(old_forward)
  cue = hk.transform(cue_forward)
  params = old.init(jax.random.PRNGKey(0), act, pair_mask)
  old_out = old.apply(params, None, act, pair_mask)
  cue_out = cue.apply(params, None, act, pair_mask)

  np.testing.assert_allclose(cue_out, old_out, rtol=5e-3, atol=5e-3)


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
    modules._require_cue_triangle_gemm_ops(
        flag='use_cue_triangle_multiplication'
    )
  except ImportError:
    return False
  return any(device.platform == 'gpu' for device in jax.devices())


@pytest.mark.skipif(
    not _real_cue_gpu_available(),
    reason='requires working cuequivariance_jax CUDA ops and a JAX GPU device',
)
@pytest.mark.parametrize(
    'equation', ['ikc,jkc->ijc', 'kjc,kic->ijc']
)
def test_cue_triangle_multiplication_forward_and_input_grad_sanity(equation):
  config = modules.TriangleMultiplication.Config(equation=equation)
  act = jax.random.normal(jax.random.PRNGKey(1), (4, 4, 8))
  mask = jnp.array(
      [[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [0, 0, 0, 0]],
      dtype=act.dtype,
  )

  def old_forward(x, m):
    return modules.TriangleMultiplication(
        config, _global_config(final_init='linear'), name='triangle'
    )(x, m)

  def cue_forward(x, m):
    return modules.CueTriangleMultiplication(
        config, _global_config(final_init='linear'), name='triangle'
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
