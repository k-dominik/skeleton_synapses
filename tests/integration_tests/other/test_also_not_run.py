import pytest


@pytest.mark.integration
def test_is_also_not_run():
    assert False
