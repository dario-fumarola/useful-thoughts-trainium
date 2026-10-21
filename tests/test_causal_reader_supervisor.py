"""No model/runtime calls: independent workers must not share bootstrap ports."""
import socket

from train.causal_reader_supervisor import worker_ports


def test_port_allocation_skips_occupied_addresses_and_never_reuses_a_pair():
    first,cursor=worker_ports(61000)
    with socket.socket() as occupied:
        occupied.bind(('127.0.0.1',first[0]))
        replacement,_=worker_ports(first[0])
        assert first[0] not in replacement
    second,next_cursor=worker_ports(cursor)
    assert len(set(first+second))==4
    assert next_cursor>cursor>max(first)


def test_port_space_exhaustion_fails_before_launch():
    import pytest
    with pytest.raises(RuntimeError,match='communication ports'):
        worker_ports(65534)
