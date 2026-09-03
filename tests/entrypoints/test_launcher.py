# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import signal
from unittest.mock import Mock, call

from vllm.entrypoints import launcher


def test_register_signal_handlers_on_windows(monkeypatch):
    loop = Mock()
    handler = Mock()
    register = Mock()
    monkeypatch.setattr(launcher.sys, "platform", "win32")
    monkeypatch.setattr(launcher.signal, "signal", register)

    launcher._register_signal_handlers(loop, handler)

    loop.add_signal_handler.assert_not_called()
    assert [registered.args[0] for registered in register.call_args_list] == [
        signal.SIGINT,
        signal.SIGTERM,
    ]
    for registered in register.call_args_list:
        registered.args[1](None, None)
    assert handler.call_count == 2


def test_register_signal_handlers_on_non_windows(monkeypatch):
    loop = Mock()
    handler = Mock()
    register = Mock()
    monkeypatch.setattr(launcher.sys, "platform", "linux")
    monkeypatch.setattr(launcher.signal, "signal", register)

    launcher._register_signal_handlers(loop, handler)

    assert loop.add_signal_handler.call_args_list == [
        call(signal.SIGINT, handler),
        call(signal.SIGTERM, handler),
    ]
    register.assert_not_called()
