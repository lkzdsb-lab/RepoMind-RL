"""Wait for ownership attachment before executing a local command."""
import json
import os
import subprocess
import sys

if __name__ == "__main__":
    # Avoid buffered read-ahead swallowing the application's first stdin input.
    line = sys.stdin.buffer.raw.readline(128000)
    if not line.endswith(b"\n"):
        raise SystemExit(1)
    argv = json.loads(line)
    if os.name == "nt":
        # Windows CRT exec with virtualenv redirectors is unreliable. The child
        # inherits the already-attached Job Object; this launcher stays alive.
        raise SystemExit(subprocess.call(argv, stdin=sys.stdin, stdout=sys.stdout,
                                       stderr=sys.stderr, creationflags=subprocess.CREATE_NO_WINDOW))
    os.execvpe(argv[0], argv, os.environ)
