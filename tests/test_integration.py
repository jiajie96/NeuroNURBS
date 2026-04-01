"""
Integration tests — skipped in CI (require GPU + full dependencies).

Run locally with:
    pytest tests/test_integration.py -v
"""

import pytest

pytest.importorskip("torch")


@pytest.mark.skip(reason="Requires full environment: OCC, PyTorch GPU, data files")
def test_surface_vae_forward_pass():
    """Smoke-test a forward pass of the surface VAE on a random batch."""
    import torch
    from network import nurbs_vae

    class _FakeArgs:
        max_ctrlPts = 4
        max_kv = 4
        option = "surface"

    model = nurbs_vae(_FakeArgs())
    model.eval()
    # Build a minimal fake input — shape follows SurfData output convention
    batch = torch.randn(2, 1, 4, 4, 4)
    with torch.no_grad():
        out = model(batch)
    assert out is not None


@pytest.mark.skip(reason="Requires full environment: OCC, PyTorch GPU, data files")
def test_edge_vae_forward_pass():
    """Smoke-test a forward pass of the edge VAE on a random batch."""
    import torch
    from network import nurbs_vae

    class _FakeArgs:
        max_ctrlPts = 4
        max_kv = 4
        option = "edge"

    model = nurbs_vae(_FakeArgs())
    model.eval()
    batch = torch.randn(2, 1, 4, 4)
    with torch.no_grad():
        out = model(batch)
    assert out is not None
