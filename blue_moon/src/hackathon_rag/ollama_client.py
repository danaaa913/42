"""Small Ollama HTTP client with no framework dependency."""

from typing import Any, Dict, Iterable, List


class OllamaClient:
    def __init__(self, base_url: str, timeout: int = 180):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _requests(self):
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("Install requests to communicate with Ollama") from exc
        return requests

    def tags(self) -> List[str]:
        response = self._requests().get(f"{self.base_url}/api/tags", timeout=10)
        response.raise_for_status()
        return [model.get("name", "") for model in response.json().get("models", [])]

    def embed(self, model: str, texts: Iterable[str], batch_size: int = 32) -> List[List[float]]:
        values = list(texts)
        vectors = []
        requests = self._requests()
        for start in range(0, len(values), batch_size):
            batch = values[start:start + batch_size]
            response = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": model, "input": batch, "truncate": False, "keep_alive": "15m"},
                timeout=self.timeout,
            )
            response.raise_for_status()
            returned = response.json().get("embeddings", [])
            if len(returned) != len(batch):
                raise RuntimeError("Ollama returned an unexpected number of embeddings")
            vectors.extend(returned)
        return vectors

    def chat_json(self, model: str, messages: List[Dict[str, str]], schema: Dict[str, Any]) -> Dict[str, Any]:
        response = self._requests().post(
            f"{self.base_url}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "format": schema,
                "options": {"temperature": 0},
                "keep_alive": "15m",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        import json
        content = response.json().get("message", {}).get("content", "")
        try:
            value = json.loads(content)
        except (json.JSONDecodeError, TypeError) as exc:
            raise RuntimeError("Ollama returned an invalid structured response") from exc
        if not isinstance(value, dict):
            raise RuntimeError("Ollama structured response was not an object")
        return value
