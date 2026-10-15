# NovaCortex Competition Submission

## Required Ollama models

- command-r7b-arabic:7b
- embeddinggemma

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r competition_requirements.txt
```

## Run competition questions

```powershell
$env:OLLAMA_CHAT_MODEL="command-r7b-arabic:7b"
$env:OLLAMA_EMBED_MODEL="embeddinggemma"
.\.venv\Scripts\python.exe .\main.py .\questions.json
```

Output:

`answers.json`

The output preserves `chats`, `chat_id`, `messages`, `message_id`, and `question`, and adds a string `answer` and a list of string `contexts` to every message.

## Optional local UI

```powershell
.\.venv\Scripts\python.exe .\hackathon_api.py
```

Open:

`http://127.0.0.1:8000`
