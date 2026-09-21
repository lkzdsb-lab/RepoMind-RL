"""Bounded numeric-loopback HTTP; no proxies or redirects."""
import http.client
import time


def failure(code, message, *, category="runtime", retryable=False, effects="none"):
    return {"ok": False, "effects": effects, "error": {
        "code": code, "category": category, "message": str(message)[:2000], "retryable": retryable}}


class LocalHttpClient:
    def request(self, *, port, path="/", method="GET", body="", content_type="application/json", timeout=5):
        if not path.startswith("/") or path.startswith("//") or any(c in path for c in "\r\n"):
            raise ValueError("path must be origin-relative without CR/LF")
        if any(c in content_type for c in "\r\n"):
            raise ValueError("Invalid content_type")
        deadline = time.monotonic() + timeout
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
        sent = False
        try:
            connection.connect()
            sent = True
            connection.request(method, path, body=body.encode("utf-8"), headers={"Content-Type": content_type})
            sock = connection.sock
            response = connection.getresponse()
            data = bytearray()
            while len(data) <= 16000:
                if response.isclosed():
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("HTTP deadline exceeded")
                sock.settimeout(remaining)
                chunk = response.read1(16001 - len(data))
                if not chunk:
                    break
                data.extend(chunk)
            return {"ok": True, "effects": "none" if method in {"GET", "HEAD"} else "applied",
                    "status_code": response.status,
                    "headers": {key[:100]: value[:500] for key, value in response.getheaders()[:32]},
                    "body": data[:16000].decode("utf-8", errors="replace"), "truncated": len(data) > 16000}
        except (OSError, http.client.HTTPException) as exc:
            code = "CONNECTION_REFUSED" if isinstance(exc, ConnectionRefusedError) else "REQUEST_FAILED"
            if isinstance(exc, TimeoutError):
                code = "REQUEST_TIMEOUT"
            return failure(code, exc, category="transient", retryable=not sent or method in {"GET", "HEAD"},
                           effects="unknown" if sent else "none")
        finally:
            connection.close()
