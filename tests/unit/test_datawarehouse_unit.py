import pytest
import xdmod.datawarehouse as xdw


VALID_XDMOD_URL = 'https://xdmod.access-ci.org'
INVALID_STR = 'asdlkfjsdlkfisdjkfjd'


def test___init___TypeError_xdmod_host():
    with pytest.raises(TypeError, match='`xdmod_host` must be a string.'):
        xdw.DataWarehouse(2, '')


def test___init___TypeError_api_token():
    with pytest.raises(TypeError, match='`api_token` must be a string.'):
        xdw.DataWarehouse('', 2)


def test___enter___RuntimeError_xdmod_host_malformed():
    with pytest.raises(
            RuntimeError,
            match='Could not connect to xdmod_host \'\': Malformed URL.'):
        with xdw.DataWarehouse('', ''):
            pass


def test___enter___RuntimeError_xdmod_host_unresolved():
    invalid_host = INVALID_STR + '.xdmod.org'
    with pytest.raises(
            RuntimeError,
            match='Could not connect to xdmod_host \'' + invalid_host
            + '\': Could not resolve host: ' + invalid_host):
        with xdw.DataWarehouse(invalid_host, ''):
            pass


def test___enter___RuntimeError_xdmod_host_unsupported_protocol():
    invalid_host = INVALID_STR + '://' + INVALID_STR
    with pytest.raises(
            RuntimeError,
            match='Could not connect to xdmod_host \'' + invalid_host
            + '\': Protocol "' + INVALID_STR
            + '" not supported or disabled in libcurl'):
        with xdw.DataWarehouse(invalid_host, ''):
            pass
