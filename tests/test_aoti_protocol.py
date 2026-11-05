"""Compatibility with the existing Recsys version-1 C++ wire format."""

import pickle
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from flexkv.common.request import KVResponse, KVResponseStatus
from flexkv.server import aoti_protocol as wire
from flexkv.server.request import GetMatchRequest, RegisterTPClientRequest, Response, StartRequest
from flexkv.server.server import KVServer


def packet(opcode, payload):
    return b"FKAO\x01" + bytes([opcode]) + payload


def ids(values):
    return struct.pack('<i', len(values)) + np.array(values, dtype='<i8').tobytes()


def namespaces():
    return struct.pack('<ii', 1, 5) + b'uid:7'


def test_python_protocol_unchanged():
    expected = GetMatchRequest(0, np.array([11, 22]), None, task_id=42, namespace=['uid:7'])
    actual, raw = wire.decode_request(pickle.dumps(expected))
    assert not raw
    assert isinstance(actual, GetMatchRequest)
    np.testing.assert_array_equal(actual.token_ids, expected.token_ids)
    assert actual.task_id == 42 and actual.namespace == ['uid:7']


@pytest.mark.parametrize('opcode', [wire.OP_GET, wire.OP_GET_MATCH])
@pytest.mark.parametrize('layers', [-1, 3])
def test_all_layer_requests_use_current_api(opcode, layers):
    payload = struct.pack('<i', 0) + ids([11, 22, 33])
    if opcode == wire.OP_GET:
        payload += ids([0, 1, 2])
    payload += b'\x00'  # no token mask
    if opcode == wire.OP_GET:
        payload += struct.pack('<qi', 42, layers)
    else:
        payload += struct.pack('<iq', layers, 42)
    request, raw = wire.decode_request(packet(opcode, payload + namespaces()), num_layers=3)
    assert raw and request.dp_client_id == 0 and request.task_id == 42
    assert request.namespace == ['uid:7'] and request.token_mask is None
    np.testing.assert_array_equal(request.token_ids, [11, 22, 33])
    if opcode == wire.OP_GET:
        np.testing.assert_array_equal(request.slot_mapping, [0, 1, 2])
    else:
        assert request.cpu_only is False and request.swa_aware is False


def test_subset_layer_request_is_not_silently_reinterpreted():
    payload = struct.pack('<i', 0) + ids([11]) + b'\x00' + struct.pack('<iq', 1, 42) + namespaces()
    with pytest.raises(ValueError, match='all-layer'):
        wire.decode_request(packet(wire.OP_GET_MATCH, payload), num_layers=3)


@pytest.mark.parametrize('opcode', [wire.OP_WAIT, wire.OP_TRY_WAIT])
def test_wait_request_fields(opcode):
    payload = struct.pack('<iiqq', 0, 2, 41, 42)
    if opcode == wire.OP_WAIT:
        payload += struct.pack('<dB', 1.25, 1)
    request, raw = wire.decode_request(packet(opcode, payload))
    assert raw and request.dp_client_id == 0
    if opcode == wire.OP_WAIT:
        assert request.wait_task_ids == [41, 42]
        assert request.wait_timeout == 1.25 and request.completely
    else:
        assert request.try_wait_task_ids == [41, 42]


def test_registration_layout_and_current_worker_identity(monkeypatch):
    monkeypatch.setattr(wire, 'TensorSharedHandle', lambda data, **kwargs: SimpleNamespace(data=data, **kwargs))
    shape = (3, 4, 2, 32, 4, 128)
    payload = struct.pack('<iiii', 0, 0, 1, 64) + bytes(64)
    payload += struct.pack('<i6qiq', len(shape), *shape, 2, 0)
    payload += struct.pack('<6iB', 0, 3, 4, 32, 4, 128, 0)
    request, raw = wire.decode_request(packet(wire.OP_REGISTER_TP_CLIENT, payload))
    assert raw and isinstance(request, RegisterTPClientRequest)
    assert request.registration_key == (0, 0) and request.pp_rank == 0
    assert tuple(request.gpu_layout.kv_shape) == shape
    assert request.gpu_layout.kv_dim == 2
    assert request.gpu_layout.get_layer_stride() == 4 * 2 * 32 * 4 * 128
    assert request.gpu_layout.get_block_stride() == 2 * 32 * 4 * 128
    assert request.gpu_layout.get_kv_stride() == 32 * 4 * 128
    assert request.handles[0].tensor_shape == shape


def test_truncated_message_rejected():
    with pytest.raises(ValueError, match='Unexpected end'):
        wire.decode_request(packet(wire.OP_START, b'\x00'))
    request, raw = wire.decode_request(packet(wire.OP_START, struct.pack('<i', 0)))
    assert raw and isinstance(request, StartRequest)


class Socket:
    def __init__(self):
        self.sent = []

    def send(self, value):
        self.sent.append(('binary', value))

    def send_pyobj(self, value):
        self.sent.append(('pickle', value))


@pytest.fixture
def server():
    instance = KVServer.__new__(KVServer)
    instance.kv_task_engine = SimpleNamespace(shutdown=lambda: None)
    instance.client_manager = SimpleNamespace(client_dict={
        0: SimpleNamespace(use_raw_protocol=True, send_to_client=Socket()),
        1: SimpleNamespace(use_raw_protocol=False, send_to_client=Socket()),
    })
    return instance


@pytest.mark.parametrize('dp_client_id', [0, 1])
def test_response_transport_matches_client(server, dp_client_id):
    response = Response(dp_client_id, is_ready=True)
    server._send_response(response, wire.RESP_IS_READY)
    transport, value = server.client_manager.client_dict[dp_client_id].send_to_client.sent[-1]
    if dp_client_id == 0:
        assert transport == 'binary' and value == packet(wire.RESP_IS_READY, b'\x01')
    else:
        assert transport == 'pickle' and value is response


def test_async_wait_uses_binary_response(server):
    response = KVResponse(KVResponseStatus.SUCCESS, 42, None)
    server.kv_task_engine._wait_impl = object()
    server.kv_task_engine._runtime = SimpleNamespace(call=lambda *args, **kwargs: {42: response})
    request, _ = wire.decode_request(packet(wire.OP_WAIT, struct.pack('<iiqdB', 0, 1, 42, 0.0, 1)))
    server._pending_waits = [(request, 0.0, {})]
    server._drain_pending_waits()
    assert not server._pending_waits
    assert server.client_manager.client_dict[0].send_to_client.sent == [
        ('binary', packet(wire.RESP_WAIT, struct.pack('<iqB', 1, 42, 0)))
    ]


def test_error_response_is_not_reported_as_ready(server):
    server._send_response(Response(0, error_msg='worker failed'), wire.RESP_IS_READY)
    assert server.client_manager.client_dict[0].send_to_client.sent == [
        ('binary', packet(wire.RESP_ERROR, struct.pack('<i', 13) + b'worker failed'))
    ]
