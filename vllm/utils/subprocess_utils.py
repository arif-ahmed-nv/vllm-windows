# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import socket
import subprocess
import sys


def popen_with_inherited_socket(
    cmd: list[str], sock: socket.socket
) -> subprocess.Popen[bytes]:
    """Launch a child process that inherits one socket."""
    fd = sock.fileno()
    if sys.platform != "win32":
        return subprocess.Popen(cmd, pass_fds=(fd,))

    was_inheritable = sock.get_inheritable()
    sock.set_inheritable(True)
    startup_info = subprocess.STARTUPINFO()
    startup_info.lpAttributeList = {"handle_list": [fd]}
    try:
        return subprocess.Popen(
            cmd,
            close_fds=True,
            startupinfo=startup_info,
        )
    finally:
        sock.set_inheritable(was_inheritable)
