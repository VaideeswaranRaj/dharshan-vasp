"""dharshan-vasp — school voice assistant (English, Tamil, Hindi)."""

from __future__ import annotations

import ast
import math
import os
import re
from datetime import datetime, timezone

from io import BytesIO

from flask import Flask, jsonify, render_template, request, send_file
from gtts import gTTS
from google import genai

SUPPORTED_LANGS = ("en", "ta", "hi")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

SYSTEM_PROMPT = """You are dharshan-vasp, a friendly school talking robot.
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
    "en": "Hello! I am dharshan-vasp. Add GEMINI_API_KEY on the server for full conversation.",
    "ta": "வணக்கம்! நான் dharshan-vasp. முழு உரையாடலுக்கு சேவையில் GEMINI_API_KEY சேர்க்கவும்.",
    "hi": "नमस्ते! मैं dharshan-vasp हूँ। पूरी बातचीत के लिए सर्वर पर GEMINI_API_KEY जोड़ें।",
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


def ask_gemini(user_msg: str, lang: str) -> str:
    if not GEMINI_API_KEY:
        return OFFLINE_HELLO[lang]

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=(
                f"{SYSTEM_PROMPT}\nLanguage code: {lang}\nUser: {user_msg}"
            ),
        )
        text = (response.text or "").strip()
        return text or OFFLINE_HELLO[lang]
    except Exception:
        return {
            "en": "dharshan-vasp could not reach the language model. Try again.",
            "ta": "dharshan-vasp மாதிரியை அணுக முடியவில்லை. மீண்டும் முயல்க.",
            "hi": "dharshan-vasp भाषा मॉडल तक नहीं पहुँच सका। फिर कोशिश करें।",
        }[lang]


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
