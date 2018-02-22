import pytest

from skeleton_synapses.helpers.files import get_credentials

ENV_VARS = {
    'CATMAID_BASE_URL': 'http://catmaid-base.url',
    'CATMAID_TOKEN': 'th1s1s4t0k3n',
    'CATMAID_PROJECT_ID': 1
}

MORE_ENV_VARS = {
    'CATMAID_AUTH_PASS': 'mypassword',
    'CATMAID_AUTH_NAME': 'myname'
}


@pytest.fixture
def no_env_vars(monkeypatch):
    for key in list(ENV_VARS) + list(MORE_ENV_VARS):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def min_env_vars(monkeypatch, no_env_vars):
    for key, value in ENV_VARS.items():
        monkeypatch.setenv(key, value)


@pytest.fixture
def all_env_vars(monkeypatch, min_env_vars):
    for key, value in MORE_ENV_VARS.items():
        monkeypatch.setenv(key, value)


def test_get_credentials_missing_raises(no_env_vars):
    with pytest.raises(KeyError):
        get_credentials(None)


def test_get_credentials_can_get_min(min_env_vars):
    creds = get_credentials(None)
    expected = {key[8:].lower(): value for key, value in ENV_VARS.items()}
    assert creds == expected


def test_get_credentials_can_get_all(all_env_vars):
    creds = get_credentials(None)
    expected = {key[8:].lower(): value for key, value in ENV_VARS.items()}
    expected.update({key[8:].lower(): value for key, value in MORE_ENV_VARS.items()})
    assert creds == expected
