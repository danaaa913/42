# NovaCortex — نسخة المسابقة (Submission)

مساعد حكومي أردني **محلي** يجيب من الوثائق الرسمية فقط عبر نماذج Ollama المحلية.
يخدم 12 خدمة (جواز السفر، دفتر العائلة، البطاقة الذكية) × 4 حالات لكل خدمة.

> النظرة الكاملة والمعمارية في [README الرئيسي للمستودع](../README.md).

---

## الموديلات المطلوبة (Ollama)

```bash
ollama pull command-r7b-arabic:7b   # التوليد
ollama pull embeddinggemma          # التضمين
```

## التثبيت

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r competition_requirements.txt
.\.venv\Scripts\python.exe -m pip install -r hackathon_requirements.txt   # للواجهة
```

## التشغيل

```powershell
$env:OLLAMA_CHAT_MODEL="command-r7b-arabic:7b"
$env:OLLAMA_EMBED_MODEL="embeddinggemma"
.\.venv\Scripts\python.exe .\main.py .\questions.json
```

**المخرجات:** `answers.json` — يحافظ على `chats`/`chat_id`/`messages`/`message_id`/`question`
ويضيف لكل رسالة `answer` (نص) و`contexts` (قائمة مقاطع المصادر).

## الواجهة المحلية (اختيارية)

```powershell
.\.venv\Scripts\python.exe .\hackathon_api.py
```

ثم افتح `http://127.0.0.1:8000`.

## الاختبارات

```powershell
python -m unittest test_competition_main.py
python -m unittest test_competition_api.py
```
