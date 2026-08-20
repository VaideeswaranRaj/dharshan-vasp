"""AIHR — Artificial Intelligence Helping Robot (English, Tamil, Hindi)."""

from __future__ import annotations

import ast
import json
import logging
import math
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from io import BytesIO

from flask import Flask, jsonify, render_template, request, send_file
from gtts import gTTS
from google import genai

log = logging.getLogger("aihr")
if not log.handlers:
    logging.basicConfig(level=logging.INFO)


def _load_local_env() -> None:
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.isfile(env_path):
        return
    with open(env_path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


_load_local_env()

SUPPORTED_LANGS = ("en", "ta", "hi")
GEMINI_MODEL = (os.environ.get("GEMINI_MODEL") or "gemini-3.6-flash").strip()
GEMINI_API_KEY = (
    os.environ.get("GEMINI_API_KEY")
    or os.environ.get("GOOGLE_API_KEY")
    or os.environ.get("GOOGLE_GENAI_API_KEY")
    or ""
).strip().strip('"').strip("'")
FALLBACK_MODELS = (
    GEMINI_MODEL,
    "gemini-3.6-flash",
    "gemini-2.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-flash-latest",
)
_working_model: str | None = None

SYSTEM_PROMPT = """You are AIHR, Artificial Intelligence Helping Robot, a friendly school talking robot.
Reply in the language given by the language code:
- en = English
- ta = Tamil (தமிழ்)
- hi = Hindi (हिन्दी)
Keep answers short and easy to speak aloud. No markdown, no bullet walls, no emojis.
If the user asks you to open a website or do device actions, explain that they should use the voice commands the microphone already supports (YouTube, Google, search).
"""

MATH_PREFIX = {
    "en": "The result is",
    "ta": "விடை",
    "hi": "परिणाम है",
}

OFFLINE_HELLO = {
    "en": "Hello! I am AIHR, Artificial Intelligence Helping Robot. Add GEMINI_API_KEY on the server for full conversation.",
    "ta": "வணக்கம்! நான் AIHR, Artificial Intelligence Helping Robot. முழு உரையாடலுக்கு சேவையில் GEMINI_API_KEY சேர்க்கவும்.",
    "hi": "नमस्ते! मैं AIHR, Artificial Intelligence Helping Robot हूँ। पूरी बातचीत के लिए सर्वर पर GEMINI_API_KEY जोड़ें।",
}

WORD_MATH_PATTERNS = (
    (r"(?:math\s+)?add\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)", lambda a, b: a + b),
    (r"(?:math\s+)?subtract\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)", lambda a, b: a - b),
    (r"(?:math\s+)?multiply\s+(\d+(?:\.\d+)?)\s+(?:and|by)\s+(\d+(?:\.\d+)?)", lambda a, b: a * b),
    (r"(?:math\s+)?divide\s+(\d+(?:\.\d+)?)\s+by\s+(\d+(?:\.\d+)?)", lambda a, b: a / b),
    (r"(\d+(?:\.\d+)?)\s*(?:plus|கூட்டு|जोड़)\s*(\d+(?:\.\d+)?)", lambda a, b: a + b),
    (r"(\d+(?:\.\d+)?)\s*(?:minus|கழி|घटा)\s*(\d+(?:\.\d+)?)", lambda a, b: a - b),
    (r"(\d+(?:\.\d+)?)\s*(?:times|பெருக்க|गुणा)\s*(\d+(?:\.\d+)?)", lambda a, b: a * b),
)


def _safe_pow(a, b):
    if abs(b) > 12 or abs(a) > 1_000_000:
        raise ValueError("exponent too large")
    return a ** b


def _safe_factorial(n):
    if not isinstance(n, int) or n < 0 or n > 20:
        raise ValueError("factorial out of range")
    return math.factorial(n)


class SafeMathVisitor(ast.NodeVisitor):
    OPERATORS = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.FloorDiv: lambda a, b: a // b,
        ast.Mod: lambda a, b: a % b,
        ast.Pow: _safe_pow,
        ast.USub: lambda a: -a,
        ast.UAdd: lambda a: +a,
    }
    FUNCTIONS = {
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "sqrt": math.sqrt,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "abs": abs,
        "radians": math.radians,
        "degrees": math.degrees,
        "factorial": _safe_factorial,
    }
    CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}

    def visit_Constant(self, node):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("unsupported constant")

    def visit_Num(self, node):  # pragma: no cover - py3.7 compat
        return node.n

    def visit_Name(self, node):
        if node.id in self.CONSTANTS:
            return self.CONSTANTS[node.id]
        raise NameError(node.id)

    def visit_BinOp(self, node):
        op = type(node.op)
        if op not in self.OPERATORS:
            raise TypeError("operator not allowed")
        return self.OPERATORS[op](self.visit(node.left), self.visit(node.right))

    def visit_UnaryOp(self, node):
        op = type(node.op)
        if op not in self.OPERATORS:
            raise TypeError("unary operator not allowed")
        return self.OPERATORS[op](self.visit(node.operand))

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name) or node.func.id not in self.FUNCTIONS:
            raise TypeError("function not allowed")
        if node.keywords:
            raise TypeError("keywords not allowed")
        args = [self.visit(arg) for arg in node.args]
        return self.FUNCTIONS[node.func.id](*args)

    def generic_visit(self, node):
        raise SyntaxError(type(node).__name__)


def _format_math_result(value, lang: str) -> str:
    if isinstance(value, float):
        value = int(value) if value.is_integer() else round(value, 6)
    prefix = MATH_PREFIX.get(lang, MATH_PREFIX["en"])
    return f"{prefix} {value}"


def evaluate_math(expression: str, lang: str) -> str | None:
    text = expression.strip()
    if not text:
        return None

    lowered = text.lower()
    for pattern, op in WORD_MATH_PATTERNS:
        match = re.search(pattern, lowered)
        if match:
            try:
                a, b = float(match.group(1)), float(match.group(2))
                if pattern.startswith(r"(?:math\s+)?divide") and b == 0:
                    return None
                return _format_math_result(op(a, b), lang)
            except (ValueError, ZeroDivisionError):
                return None

    compact = text.replace("^", "**")
    compact = re.sub(r"[^0-9+\-*/().,%\sA-Za-z_]", "", compact)
    if not re.search(r"\d", compact):
        return None
    if not re.search(r"[+\-*/%]|sqrt|sin|cos|tan|log", compact, re.I):
        return None
    if len(compact.split()) > 8:
        return None

    try:
        tree = ast.parse(compact, mode="eval")
        result = SafeMathVisitor().visit(tree.body)
        return _format_math_result(result, lang)
    except Exception:
        return None


LLM_ERROR = {
    "en": "AIHR could not reach the language model. Try again.",
    "ta": "AIHR மாதிரியை அணுக முடியவில்லை. மீண்டும் முயல்க.",
    "hi": "AIHR भाषा मॉडल तक नहीं पहुँच सका। फिर कोशिश करें।",
}
KEY_ERROR = {
    "en": "The Gemini API key is not valid. Put a new key in the .env file as GEMINI_API_KEY, then restart the app.",
    "ta": "Gemini API விசை தவறானது. .env கோட்டில் புதிய GEMINI_API_KEY போட்டு, ஆப்பை மீண்டும் தொடங்கவும்.",
    "hi": "Gemini API कुंजी सही नहीं है। .env में नई GEMINI_API_KEY डालकर ऐप फिर से चालू करें।",
}
QUOTA_ERROR = {
    "en": "The Gemini quota is finished for now. Wait a little, then try again.",
    "ta": "Gemini ஒதுக்கீடு தற்காலிகமாக முடிந்துவிட்டது. சிறிது நேரம் கழித்து முயல்க.",
    "hi": "Gemini कोटा अभी खत्म है। थोड़ी देर बाद फिर कोशिश करें।",
}


class GeminiAuthError(Exception):
    pass


class GeminiQuotaError(Exception):
    pass


def _models_to_try() -> list[str]:
    models: list[str] = []
    if _working_model:
        models.append(_working_model)
    for name in FALLBACK_MODELS:
        if name and name not in models:
            models.append(name)
    return models


def _classify_gemini_error(err: Exception, detail: str = "") -> Exception:
    text = f"{err} {detail}".lower()
    if "api_key_invalid" in text or "api key not valid" in text or "invalid api key" in text:
        return GeminiAuthError("invalid api key")
    if "resource_exhausted" in text or "resource has been exhausted" in text or "429" in text:
        return GeminiQuotaError("quota exhausted")
    return err


def _sdk_text(response) -> str:
    try:
        text = (response.text or "").strip()
        if text:
            return text
    except Exception:
        pass
    try:
        for cand in getattr(response, "candidates", None) or []:
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None) or []
            bits = [str(getattr(part, "text", "") or "") for part in parts]
            text = " ".join(bit for bit in bits if bit).strip()
            if text:
                return text
    except Exception:
        pass
    return ""


def _rest_text(payload: dict) -> str:
    for cand in payload.get("candidates") or []:
        parts = ((cand.get("content") or {}).get("parts") or [])
        bits = [str(part.get("text") or "") for part in parts]
        text = " ".join(bit for bit in bits if bit).strip()
        if text:
            return text
    return ""


def _generate_sdk(client, model: str, user_msg: str, lang: str) -> str:
    prompt = f"{SYSTEM_PROMPT}\nLanguage code: {lang}\nUser: {user_msg}"
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                "temperature": 0.4,
                "max_output_tokens": 320,
            },
        )
        return _sdk_text(response)
    except Exception as err:
        raise _classify_gemini_error(err) from err


def _generate_rest(model: str, user_msg: str, lang: str) -> str:
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={urllib.parse.quote(GEMINI_API_KEY)}"
    )
    prompt = f"{SYSTEM_PROMPT}\nLanguage code: {lang}\nUser: {user_msg}"
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": 320},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", errors="replace")
        log.warning("Gemini REST %s HTTP %s: %s", model, err.code, detail[:300])
        raise _classify_gemini_error(err, detail) from err
    text = _rest_text(payload)
    if text:
        return text
    log.warning("Gemini REST %s empty: %s", model, payload.get("promptFeedback") or payload.get("error"))
    return ""


def ask_gemini(user_msg: str, lang: str) -> str:
    global _working_model
    if not GEMINI_API_KEY:
        return OFFLINE_HELLO[lang]

    client = None
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as err:
        log.warning("Gemini client init failed: %s", err)

    last_err: Exception | None = None
    for model in _models_to_try():
        attempts = []
        if client is not None:
            attempts.append(("sdk", lambda m=model: _generate_sdk(client, m, user_msg, lang)))
        attempts.append(("rest", lambda m=model: _generate_rest(m, user_msg, lang)))
        for label, call in attempts:
            try:
                text = call()
                if text:
                    _working_model = model
                    return text
            except GeminiAuthError:
                log.error("Gemini API key rejected")
                return KEY_ERROR[lang]
            except GeminiQuotaError:
                log.error("Gemini quota exhausted")
                return QUOTA_ERROR[lang]
            except Exception as err:
                last_err = err
                log.warning("Gemini %s %s failed: %s", label, model, err)

    log.error("Gemini failed for %r: %s", user_msg[:80], last_err)
    return LLM_ERROR[lang]


app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates"),
)


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/favicon.ico")
def favicon():
    return ("", 204)


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "gemini": bool(GEMINI_API_KEY),
            "model": _working_model or GEMINI_MODEL,
            "time": datetime.now(timezone.utc).isoformat(),
        }
    )


@app.post("/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    user_msg = str(payload.get("message") or "").strip()
    lang = payload.get("language")
    if lang not in SUPPORTED_LANGS:
        lang = "en"

    if not user_msg:
        return jsonify({"error": "empty message"}), 400

    math_reply = evaluate_math(user_msg, lang)
    if math_reply:
        return jsonify({"reply": math_reply, "source": "math"})

    return jsonify({"reply": ask_gemini(user_msg, lang), "source": "llm"})


TTS_LANG = {"en": "en", "ta": "ta", "hi": "hi"}


@app.post("/speak")
def speak_audio():
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text") or "").strip()
    lang = payload.get("language")
    if lang not in SUPPORTED_LANGS:
        lang = "en"
    if not text:
        return jsonify({"error": "empty text"}), 400
    text = re.sub(r"\s+", " ", text)[:400]
    audio = BytesIO()
    try:
        gTTS(text=text, lang=TTS_LANG[lang], slow=False, lang_check=False).write_to_fp(audio)
    except Exception:
        return jsonify({"error": "tts failed"}), 502
    audio.seek(0)
    return send_file(audio, mimetype="audio/mpeg", download_name="reply.mp3")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    host = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"
    app.run(host=host, port=port, debug=debug)
