import pytest


@pytest.fixture(autouse=True)
def _isolate_working_directory(tmp_path, monkeypatch):
    """Run every test from its own temp directory.

    The training and evaluation code writes to paths relative to the working
    directory (``outputs/checkpoints/<experiment>/...``). Run from the repo root, a
    test that trains a model — test_cross_dataset's pooled_evaluation, which falls
    back to the ``phase4_unified`` experiment — therefore overwrote the real
    ``outputs/checkpoints/phase4_unified/stgcn_best.pt`` with a one-epoch model
    trained on synthetic noise, every time the suite ran. Imports are unaffected:
    pytest resolves the ``pythonpath`` setting to an absolute path at startup.
    """
    monkeypatch.chdir(tmp_path)
