"""Sequential local tool mailbox; no external service or agent framework."""
import time
from .artifacts import read_json, write_json


class ToolClient:
    """ 运行在 subagent worker 中"""
    def __init__(self, directory, deadline, evidence):
        self.directory, self.deadline, self.evidence = directory, deadline, evidence
        self.sequence = 0

    def dispatch(self, name, arguments):
        self.sequence += 1
        write_json(self.directory / "tool_call.json", {
            "sequence": self.sequence, "name": name, "arguments": arguments})
        # Wait for the Tool Host; the parent independently supervises both processes.
        while True:
            self.deadline.remaining()
            path = self.directory / "tool_reply.json"
            if path.is_file():
                reply = read_json(path)
                if reply.get("sequence") == self.sequence:
                    self.evidence.records.clear()
                    self.evidence.load()
                    return reply["result"]
            time.sleep(0.05)


class ToolServer:
    """Runs in the disposable Tool Host, never in the supervisor."""
    def __init__(self, directory, tools, limit, operation_timeout=30):
        self.directory, self.tools, self.limit = directory, tools, limit
        self.operation_timeout = operation_timeout
        self.sequence = 0

    def poll(self):
        path = self.directory / "tool_call.json"
        if not path.is_file():
            return
        # 1. 读取请求
        call = read_json(path)
        sequence = call.get("sequence")
        # 2. 判断是不是已经处理过
        if sequence == self.sequence:
            return
        # 3. 检查调用顺序和预算
        if sequence != self.sequence + 1 or sequence > self.limit:
            raise ValueError("Invalid tool sequence or exhausted tool budget")
        if not isinstance(call.get("name"), str) or not isinstance(call.get("arguments"), dict):
            raise ValueError("Malformed tool call")
        timeout = self.operation_timeout
        if call["name"] == "local_request":
            requested = call["arguments"].get("timeout", 5)
            if isinstance(requested, (int, float)) and 1 <= requested <= 20:
                # Allow a small delivery margin after the tool's own timeout.
                timeout = min(timeout, requested + 1)
        write_json(self.directory / "host_status.json", {
            "phase": "operation", "sequence": sequence, "tool": call["name"],
            "deadline": min(self.tools.backend.deadline, time.monotonic() + timeout)})
        result = self.tools.dispatch(call["name"], call["arguments"])
        # Never re-execute a call if delivery of its reply fails.
        self.sequence = sequence
        write_json(self.directory / "tool_reply.json", {"sequence": sequence, "result": result})
        write_json(self.directory / "host_status.json", {"phase": "ready", "deadline": 0})
