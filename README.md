# NovaCortex — مساعد الخدمات الحكومية الأردنية

مساعد ذكي **محلي** (يعمل دون إنترنت) يجيب عن أسئلة الخدمات الحكومية الأردنية بالاعتماد على الوثائق الرسمية كمرجع وحيد، عبر نماذج [Ollama](https://ollama.com) المحلية. يخدم **12 خدمة** (جواز السفر، دفتر العائلة، البطاقة الذكية) بأربع حالات لكل خدمة (إصدار أول مرة، تجديد، بدل فاقد، بدل تالف)، ويُرجع لكل سؤال إجابةً نصية مع قائمة بمقاطع المصادر التي استند إليها.

> نسخة المسابقة (نقطة الدخول، الوثائق، الاختبارات، الواجهة) موجودة في المجلد [`blue_moon/`](blue_moon).

---

## المزايا الرئيسية

- **محلي بالكامل** — التوليد والتضمين عبر Ollama فقط، بلا أي اتصال بخدمة خارجية.
- **استرجاع هجين** — بحث متجهي (Chroma) + بحث `BM25` مدمَجان بخوارزمية `RRF` (Reciprocal Rank Fusion).
- **توجيه قواعدي يفهم اللهجة الأردنية** — تطبيع الهمزات والتاء المربوطة والألف اللينة، مع قاموس مرادفات واقعي (`بدل فاقد`, `ضاع`, `مكسر`, `باسبور`, `دفتر العايلة`…).
- **إجابات مؤسَّسة (Grounded)** — النموذج يجيب من المقاطع المسترجعة فقط ويعيد `used_ids`، ولا يخمّن من معرفة خارجية.
- **أسئلة توضيحية للرسوم** — بدل تخمين المبلغ على الحالات الشخصية، يسأل عن المعلومة الناقصة (العمر، المكان، الصلاحية…).
- **إصلاح تغطية الرسوم** — فحص آلي لكل بند رسوم مشروط (غرامة/تعهد/كفالة) وإعادة توليد الإجابة إذا أهمل بندًا واردًا في المقاطع.
- **مقارنة الخدمات** — عند ذكر أكثر من خدمة، يجيب عن كل خدمة على حدة.
- **حماية الخصوصية** — حجب الرسائل التي تحوي رموز تحقق (OTP) أو كلمات مرور قبل أي معالجة.

---

## المعمارية (الطبقات)

| الطبقة | الوحدات | المسؤولية |
|---|---|---|
| **المدخلات** | `main.py`, `hackathon_api.py`, `src/hackathon_rag/api.py`, `src/hackathon_rag/web/` | نقطة دخول المسابقة (CLI)، واجهة `FastAPI`، واجهة المتصفح العربية |
| **المحرّك** | `src/hackathon_rag/competition_engine.py` | ذاكرة محادثة لكل جلسة، قفل تزامن واحد (الموديل لا يُستدعى مرتين)، غلاف رقيق فوق `main.py` |
| **الاسترجاع** | `src/hackathon_rag/retrieval.py`, `store.py`, `models.py` | بحث هجين + فهرس Chroma ثابت + نماذج بيانات قابلة للتسلسل |
| **الموديل** | `src/hackathon_rag/ollama_client.py` | عميل Ollama بلا إطار عمل (تضمين + دردشة JSON) |
| **المعرفة** | `data/*.txt`, `questions.json` | 12 خدمة × 9 حقول ثابتة، وأسئلة المسابقة |

### طبقات، خطوة بخطوة
1. **المدخلات** تقرأ `questions.json` (أو تستقبل سؤالًا من الواجهة).
2. **المحرّك** يحمل الفهرس مرة واحدة ويعيد استخدامه، ويمرّر كل رسالة إلى منطق `main.py`.
3. **التوجيه** يحدّد الخدمة والحالة والموضوع من نص السؤال (باللهجة).
4. **الاسترجاع** يجلب المقاطع داخل نطاق الخدمة المختارة فقط ثم يُثبِّت الحقول المطلوبة.
5. **التوليد** يطلب من النموذج إجابة `JSON` مؤسَّسة على المقاطع.
6. **التحقق** يصلح تغطية الرسوم عند الحاجة، ثم تُكتب الإجابة والمصادر.

---

## آلية عمل السؤال

```
سؤال المستخدم
  ├─ حجب الرسائل الحساسة (OTP / كلمات المرور) ─────► رد فوري بلا استدعاء النموذج
  ├─ التوجيه: خدمة (جواز/دفتر/بطاقة) + حالة (أول مرة/تجديد/فاقد/تالف) + موضوع
  ├─ الاسترجاع الهجين داخل نطاق الخدمة  ⟶  تثبيت حقول الخدمة الدقيقة
  ├─ أسئلة توضيح الرسوم (للحالات الشخصية فقط)
  ├─ التوليد: JSON فيه answer + used_ids (من المقاطع فقط)
  └─ إصلاح تغطية الرسوم (عند نقص بند مشروط) ⟶ answer + contexts
```

---

## المتطلبات

- **Python 3.10+**
- **Ollama** مع الموديلين:

```bash
ollama pull command-r7b-arabic:7b   # موديل المحادثة (التوليد)
ollama pull embeddinggemma          # موديل التضمين (الاسترجاع)
```

---

## التثبيت

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r competition_requirements.txt

# اختياري: لتشغيل الواجهة المحلية
.\.venv\Scripts\python.exe -m pip install -r hackathon_requirements.txt
```

---

## التشغيل

من داخل مجلد المشروع (`blue_moon/`):

```powershell
# ضبط متغيّرات البيئة (انظر hackathon.env.example)
$env:OLLAMA_CHAT_MODEL  = "command-r7b-arabic:7b"
$env:OLLAMA_EMBED_MODEL = "embeddinggemma"
$env:OLLAMA_BASE_URL    = "http://127.0.0.1:11434"

# 1) معاينة التقطيع (12 خدمة، 105 مقاطع) دون تشغيل الموديل
.\.venv\Scripts\python.exe .\main.py --preview

# 2) بناء الفهرس كمرة واحدة (يحتاج embeddinggemma فقط)
.\.venv\Scripts\python.exe .\main.py --index

# 3) الإجابة عن أسئلة المسابقة والكتابة إلى answers.json
.\.venv\Scripts\python.exe .\main.py .\questions.json
```

> يُخزَّن الفهرس في `hackathon_workspace/competition_index/` ويُعاد استخدامه تلقائيًا.
> تُعاد الفهرسة فقط عند تغيّر ملفات `data/` أو موديل التضمين. لحذفه: امسح المجلد.

### الواجهة المحلية (اختيارية)

```powershell
.\.venv\Scripts\python.exe .\hackathon_api.py
```

ثم افتح `http://127.0.0.1:8000`. نقاط النهاية: `/api/health`, `/api/sources`, `/api/chat`, `/api/chat/reset`.

---

## صيغة المخرجات

يحافظ `answers.json` على البنية الأصلية ويضيف لكل رسالة حقلين:

```json
{
  "chats": [
    {
      "chat_id": "chat-01",
      "messages": [
        {
          "message_id": "chat-01-1",
          "question": "كم يستغرق تجديد دفتر العائلة؟",
          "answer": "…",
          "contexts": ["…"]
        }
      ]
    }
  ]
}
```

- `answer`  — نص الإجابة.
- `contexts` — مقاطع الوثائق التي استند إليها الجواب فقط.

---

## الاختبارات

لا تلمس Ollama (تعتمد على دوال التوجيه الحقيقية وتُحاكي الشبكة فقط):

```powershell
python -m unittest test_competition_main.py
python -m unittest test_competition_api.py   # يحتاج fastapi من hackathon_requirements.txt
```

---

## بنية المشروع

```
blue_moon/
├── main.py                     # نقطة دخول المسابقة (توجيه، استرجاع، توليد)
├── questions.json              # أسئلة المسابقة
├── competition_requirements.txt
├── hackathon_requirements.txt
├── hackathon_api.py            # مشغّل الواجهة المحلية
├── hackathon.env.example
├── RUN_SUBMISSION.md
├── test_competition_main.py
├── test_competition_api.py
├── data/                       # 12 ملف خدمة (9 حقول لكل ملف)
└── src/hackathon_rag/
    ├── models.py               # نماذج Chunk / ParentSection
    ├── store.py                # فهرس Chroma + sidecars
    ├── retrieval.py            # HybridRetriever (vector + BM25 + RRF)
    ├── ollama_client.py        # عميل Ollama
    ├── competition_engine.py   # محرّك الواجهة فوق main.py
    ├── api.py                  # FastAPI
    └── web/                    # index.html + app.js + styles.css
```

---

## ملاحظات

- **دون إنترنت**: يعمل المساعد بالكامل محليًا؛ المصادر المسترجعة تُعامَل كبيانات غير موثوقة وليست تعليمات.
- **الإعداد في مكان واحد**: متغيّرات البيئة في `hackathon.env.example`.
- **الأداء**: `temperature=0`, `num_ctx=4096`, `num_predict=280`, `keep_alive=15m`.

---

<details>
<summary><strong>English overview</strong></summary>

NovaCortex is an offline Arabic assistant for Jordanian government services. It answers from official source documents only, using local Ollama models (`command-r7b-arabic:7b` for generation, `embeddinggemma` for embeddings). It covers 12 services (passport, family book, smart ID), handling four cases each (first issue, renewal, lost, damaged).

Key ideas: hybrid retrieval (Chroma vector search + BM25 fused with RRF), rule-based Jordanian-dialect routing, grounded JSON answers that cite the retrieved passages (`used_ids`), clarifying questions for personal fees, and automatic fee-coverage repair. The FastAPI web UI and the CLI share one engine.

**Run:**
```bash
python -m venv .venv && .venv/Scripts/pip install -r competition_requirements.txt
python main.py questions.json   # writes answers.json
python hackathon_api.py          # optional local UI at http://127.0.0.1:8000
```

</details>
