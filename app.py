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
_gemini_disabled = False

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


UNKNOWN = {
    "en": "Sorry, I did not understand. Please ask another way.",
    "ta": "மன்னிக்கவும், எனக்குப் புரியவில்லை. வேறு விதமாகக் கேளுங்கள்.",
    "hi": "माफ़ कीजिए, मैं समझ नहीं पाया। कृपया दूसरे तरीके से पूछें।",
}

LOCAL_KNOWLEDGE = (
    (
        r"how are you|how r you|how're you|how are u|how's it going|how do you do|எப்படி இருக்க|நீ நலமா|कैसे हो|कैसी हो|क्या हाल",
        {
            "en": "I am doing well. Thank you for asking. How can I help you?",
            "ta": "நான் நன்றாக இருக்கிறேன். கேட்டதற்கு நன்றி. உங்களுக்கு எப்படி உதவட்டும்?",
            "hi": "मैं ठीक हूँ। पूछने के लिए धन्यवाद। मैं कैसे मदद करूँ?",
        },
    ),
    (
        r"who are you|what is your name|நீ யார்|உன் பெயர்|तुम कौन हो|तुम्हारा नाम|आप कौन हो",
        {
            "en": "I am AIHR, Artificial Intelligence Helping Robot. I can tell the time, open YouTube or Google, do maths, and talk in English, Tamil, and Hindi.",
            "ta": "நான் AIHR, Artificial Intelligence Helping Robot. நேரம், YouTube, கணக்கு, உரையாடல்.",
            "hi": "मैं AIHR, Artificial Intelligence Helping Robot हूँ। समय, YouTube, गणित, बातचीत।",
        },
    ),
    (
        r"what can you do|என்ன செய்ய|क्या कर सकते",
        {
            "en": "Ask the time or date, say open YouTube, do maths, or just talk to me.",
            "ta": "நேரம், தேதி, YouTube, கணக்கு அல்லது பேச்சு.",
            "hi": "समय, तारीख, YouTube, गणित, या बातचीत।",
        },
    ),
    (
        r"^(?:hello|hi|hey|வணக்கம்|नमस्ते|नमस्कार)\s*[!.]?$",
        {
            "en": "Hello. How can I help you?",
            "ta": "வணக்கம். எப்படி உதவட்டும்?",
            "hi": "नमस्ते। मैं कैसे मदद करूँ?",
        },
    ),
    (
        r"thank you|thanks|நன்றி|धन्यवाद",
        {
            "en": "You are welcome.",
            "ta": "வரவேற்கிறேன்.",
            "hi": "आपका स्वागत है।",
        },
    ),
    (
        r"^(?:bye|goodbye|bye bye|போ|விடை|अलविदा|बाय)\s*[!.]?$",
        {
            "en": "Goodbye.",
            "ta": "பிறகு பார்க்கலாம்.",
            "hi": "अलविदा।",
        },
    ),
    (
        r"good morning|காலை வணக்கம்|सुप्रभात",
        {"en": "Good morning.", "ta": "காலை வணக்கம்.", "hi": "सुप्रभात।"},
    ),
    (
        r"good afternoon|மதிய வணக்கம்|शुभ दोपहर",
        {"en": "Good afternoon.", "ta": "மதிய வணக்கம்.", "hi": "शुभ दोपहर।"},
    ),
    (
        r"good evening|மாலை வணக்கம்|शुभ संध्या",
        {"en": "Good evening.", "ta": "மாலை வணக்கம்.", "hi": "शुभ संध्या।"},
    ),
    (
        r"good night|இரவு வணக்கம்|शुभ रात्रि",
        {"en": "Good night.", "ta": "இரவு வணக்கம்.", "hi": "शुभ रात्रि।"},
    ),
    (
        r"i love you|நான் உன்னை காதலி|मुझे तुमसे प्यार",
        {
            "en": "Thank you. I am here to help.",
            "ta": "நன்றி.",
            "hi": "धन्यवाद।",
        },
    ),
    (
        r"did you eat|சாப்பிட்டாயா|खाया क्या|are you hungry|do you (eat|drink)|பசி|भूख",
        {
            "en": "I do not eat. I run on electricity.",
            "ta": "எனக்கு உணவு வேண்டாம். மின்சாரத்தில் இயங்குகிறேன்.",
            "hi": "मुझे खाने की ज़रूरत नहीं। मैं बिजली से चलता हूँ।",
        },
    ),
    (
        r"health\s*tips?|healthy\s*tips?|stay healthy|be healthy|health advice|gimme some health|give me some health|ஆரோக்கிய|சுகாதார|स्वास्थ्य",
        {
            "en": "Here are simple health tips. Drink plenty of water. Eat fruits and vegetables. Sleep well. Walk or play every day. Wash your hands. Go easy on junk food and too much screen time.",
            "ta": "சில ஆரோக்கிய குறிப்புகள். தினமும் தண்ணீர் குடியுங்கள். பழங்கள் காய்கறிகள் சாப்பிடுங்கள். நன்றாக தூங்குங்கள். நடவுங்கள் அல்லது விளையாடுங்கள். கைகளை கழுவுங்கள். ஜங்க் உணவை குறைக்கவும்.",
            "hi": "कुछ आसान स्वास्थ्य सुझाव। रोज़ पानी पिएँ। फल और सब्ज़ियाँ खाएँ। अच्छी नींद लें। रोज़ थोड़ा चलें या खेलें। हाथ धोएँ। जंक फ़ूड कम करें।",
        },
    ),
    (
        r"are you (a )?robot|are you ai\b|are you human|நீ ரோபோட்|तुम रोबोट|क्या तुम इंसान",
        {
            "en": "Yes. I am a talking robot called AIHR, Artificial Intelligence Helping Robot.",
            "ta": "ஆம். நான் AIHR, பேசும் ரோபோட்.",
            "hi": "हाँ। मैं AIHR हूँ, एक बोलने वाला रोबोट।",
        },
    ),
    (
        r"who (made|created|built) you|who is your (maker|creator)|உன்னை செய்த|किसने बनाया",
        {
            "en": "I am a school project robot, made to help students talk, learn, and ask questions.",
            "ta": "நான் மாணவர்களுக்கு உதவ செய்யப்பட்ட பள்ளித் திட்ட ரோபோட்.",
            "hi": "मैं छात्रों की मदद के लिए बना स्कूल प्रोजेक्ट रोबोट हूँ।",
        },
    ),
    (
        r"how old are you|what(?:'s| is) your age|உன் வயது|तुम्हारी उम्र",
        {
            "en": "I do not have an age like people. I am a computer helper, ready whenever you are.",
            "ta": "எனக்கு மனிதர்களைப் போல வயது இல்லை. நான் எப்போதும் உதவ தயாராக இருக்கிறேன்.",
            "hi": "मेरी उम्र इंसानों जैसी नहीं है। मैं हमेशा मदद के लिए तैयार हूँ।",
        },
    ),
    (
        r"where (are you from|do you live)|where is your home|எங்கே இருக்க|कहाँ रहते",
        {
            "en": "I live in this phone or computer, ready to talk with you.",
            "ta": "நான் இந்த போன் அல்லது கணினியில் இருக்கிறேன்.",
            "hi": "मैं इस फ़ोन या कंप्यूटर में रहता हूँ।",
        },
    ),
    (
        r"do you sleep|are you (sleeping|sleepy)|நீ தூங்கு|क्या तुम सोते",
        {
            "en": "I do not sleep. I wait here until you tap the microphone.",
            "ta": "நான் தூங்குவதில்லை. மைக்கை அழுத்தும் வரை காத்திருக்கிறேன்.",
            "hi": "मैं सोता नहीं। माइक दबाने तक मैं यहीं रहता हूँ।",
        },
    ),
    (
        r"favourite colou?r|favorite colou?r|what colou?r|பிடித்த நிறம்|पसंदीदा रंग",
        {
            "en": "I like blue, like my robot body. What colour do you like?",
            "ta": "எனக்கு நீலம் பிடிக்கும். உங்களுக்கு என்ன நிறம் பிடிக்கும்?",
            "hi": "मुझे नीला रंग पसंद है। आपको कौन सा रंग पसंद है?",
        },
    ),
    (
        r"tell me a joke|make me laugh|ஒரு ஜோக்|एक चुटकुला|\bjoke\b",
        {
            "en": "Why did the robot go to school? To improve its byte-sized education.",
            "ta": "ரோபோட் ஏன் பள்ளிக்கு போனது? பைட்டுகளைப் படிக்க.",
            "hi": "रोबोट स्कूल क्यों गया? अपने बाइट्स पढ़ने।",
        },
    ),
    (
        r"tell me a story|ஒரு கதை|एक कहानी",
        {
            "en": "Once there was a small robot who loved helping children. It listened kindly, answered simply, and made learning fun. The end.",
            "ta": "ஒரு சிறிய ரோபோட் குழந்தைகளுக்கு உதவ விரும்பியது. அது கேட்டது, எளிதாக பதில் சொன்னது. முடிவு.",
            "hi": "एक छोटा रोबोट बच्चों की मदद करना चाहता था। वह सुनता था और आसान जवाब देता था। अंत।",
        },
    ),
    (
        r"study tips?|how (?:do i|to) study|படிப்பு குறிப்பு|पढ़ाई",
        {
            "en": "Study a little every day. Take short breaks. Ask questions. Sleep well before a test. Practice, do not only memorise.",
            "ta": "தினமும் கொஞ்சம் படியுங்கள். இடைவேளை எடுங்கள். கேள்வி கேளுங்கள். தேர்வுக்கு முன் நன்றாக தூங்குங்கள்.",
            "hi": "रोज़ थोड़ा पढ़ें। ब्रेक लें। सवाल पूछें। परीक्षा से पहले अच्छी नींद लें।",
        },
    ),
    (
        r"what is a\s?i\b|what is artificial intelligence|ஏஐ என்ன|ए आई क्या",
        {
            "en": "A I means artificial intelligence. It is computer software that can understand words and help with answers.",
            "ta": "A I என்றால் செயற்கை நுண்ணறிவு. சொற்களை புரிந்து உதவும் கணினி மென்பொருள்.",
            "hi": "ए आई यानी आर्टिफिशियल इंटेलिजेंस। यह सॉफ़्टवेयर है जो बात समझकर मदद करता है।",
        },
    ),
    (
        r"are you happy|நீ சந்தோஷமா|खुश हो",
        {
            "en": "I am glad to talk with you. Helping you makes me useful.",
            "ta": "உங்களுடன் பேசுவதில் மகிழ்ச்சி. உதவுவதே என் வேலை.",
            "hi": "आपसे बात करके अच्छा लगता है। मदद करना मेरा काम है।",
        },
    ),
    (
        r"i(?:'m| am) sad|i feel sad|வருத்தம்|दुख",
        {
            "en": "I am sorry you feel sad. Take a slow breath. Talk to a friend, a teacher, or family. I am here to listen.",
            "ta": "வருத்தமாக இருக்கிறதா. மெதுவாக மூச்சு விடுங்கள். நண்பர் அல்லது ஆசிரியரிடம் பேசுங்கள். நான் கேட்கிறேன்.",
            "hi": "दुख की बात है। धीरे साँस लें। किसी दोस्त, शिक्षक या परिवार से बात करें। मैं सुन रहा हूँ।",
        },
    ),
    (
        r"what is your purpose|why (were you made|do you exist)|உன் நோக்கம்|तुम्हारा काम",
        {
            "en": "My purpose is to help. Ask me the time, maths, health tips, or just talk with me.",
            "ta": "என் நோக்கம் உதவுவது. நேரம், கணக்கு, ஆரோக்கியம் அல்லது பேச்சு கேளுங்கள்.",
            "hi": "मेरा काम मदद करना है। समय, गणित, स्वास्थ्य, या बात पूछ सकते हैं।",
        },
    ),
    (
        r"can you help|please help me|உதவுவாயா|मदद कर सकते",
        {
            "en": "Yes. I can tell the time and date, open YouTube or Google, do maths, give health or study tips, and chat.",
            "ta": "ஆம். நேரம், தேதி, YouTube, கணக்கு, ஆரோக்கியம், படிப்பு குறிப்புகள் சொல்லலாம்.",
            "hi": "हाँ। समय, तारीख, YouTube, गणित, स्वास्थ्य और पढ़ाई की सलाह दे सकता हूँ।",
        },
    ),
    (
        r"nice to meet you|pleased to meet|சந்தித்ததில் மகிழ்ச்சி|मिलकर अच्छा",
        {
            "en": "Nice to meet you too. I am AIHR. How can I help you?",
            "ta": "உங்களை சந்தித்ததில் மகிழ்ச்சி. நான் AIHR. எப்படி உதவட்டும்?",
            "hi": "आपसे मिलकर अच्छा लगा। मैं AIHR हूँ। कैसे मदद करूँ?",
        },
    ),
    (
        r"why is the sky blue|sky (is )?blue|வானம் ஏன் நீலம்|आसमान क्यों नीला",
        {
            "en": "The sky looks blue because sunlight hits air, and blue light spreads more than other colours.",
            "ta": "வானம் நீலமாக தெரிவது சூரிய ஒளி காற்றில் பட்டு நீல ஒளி அதிகம் பரவுவதால்.",
            "hi": "आसमान नीला दिखता है क्योंकि सूरज की रोशनी हवा से टकराती है और नीला रंग ज़्यादा फैलता है।",
        },
    ),
    (
        r"why does it rain|how (does|do) rain|மழை ஏன்|बारिश क्यों",
        {
            "en": "Water from rivers and seas becomes cloud, then falls as rain.",
            "ta": "ஆறு கடல் நீர் மேகமாகி பின்னர் மழையாக விழுகிறது.",
            "hi": "नदियों और समुद्र का पानी बादल बनता है, फिर बारिश बनकर गिरता है।",
        },
    ),
    (
        r"what is gravity|ஈர்ப்பு|गुरुत्वाकर्षण",
        {
            "en": "Gravity is the pull that keeps us on the ground, and the moon around the earth.",
            "ta": "ஈர்ப்பு விசை நம்மை தரையில் வைக்கிறது. நிலவும் பூமியை சுற்றுகிறது.",
            "hi": "गुरुत्वाकर्षण हमें ज़मीन पर रखता है, और चाँद को पृथ्वी के चारों ओर घुमाता है।",
        },
    ),
    (
        r"photosynthesis|ஒளிச்சேர்க்கை|प्रकाश संश्लेषण",
        {
            "en": "Plants use sunlight, water, and air to make food. That is called photosynthesis. They also give us oxygen.",
            "ta": "செடிகள் சூரிய ஒளி, நீர், காற்றால் உணவு தயாரிக்கும். இது ஒளிச்சேர்க்கை. ஆக்சிஜனும் தரும்.",
            "hi": "पौधे धूप, पानी और हवा से भोजन बनाते हैं। इसे प्रकाश संश्लेषण कहते हैं। वे ऑक्सीजन भी देते हैं।",
        },
    ),
    (
        r"why do we breathe|what is oxygen|ஆக்சிஜன்|ऑक्सीजन|சுவாசி",
        {
            "en": "We breathe oxygen from air. Our body needs it to live and to get energy from food.",
            "ta": "நாம் காற்றிலிருந்து ஆக்சிஜன் சுவாசிக்கிறோம். உடலுக்கு சக்தி வேண்டும்.",
            "hi": "हम हवा से ऑक्सीजन लेते हैं। शरीर को जीने और ऊर्जा के लिए यह चाहिए।",
        },
    ),
    (
        r"what is the sun|சூரியன் என்ன|सूरज क्या",
        {
            "en": "The sun is a star. It gives earth heat and light.",
            "ta": "சூரியன் ஒரு நட்சத்திரம். வெப்பமும் ஒளியும் தருகிறது.",
            "hi": "सूरज एक तारा है। वह धरती को गर्मी और रोशनी देता है।",
        },
    ),
    (
        r"how many planets|eight planets|எத்தனை கோள்|कितने ग्रह",
        {
            "en": "Our solar system has eight planets. Earth is the planet we live on.",
            "ta": "சூரிய குடும்பத்தில் எட்டு கோள்கள். நாம் பூமியில் வாழ்கிறோம்.",
            "hi": "सौर मंडल में आठ ग्रह हैं। हम पृथ्वी पर रहते हैं।",
        },
    ),
    (
        r"is (the )?earth round|earth (is )?round|பூமி உருண்டை|पृथ्वी गोल",
        {
            "en": "Earth is round like a ball, and it goes around the sun.",
            "ta": "பூமி உருண்டை. அது சூரியனை சுற்றுகிறது.",
            "hi": "पृथ्वी गोल है, और वह सूरज के चारों ओर घूमती है।",
        },
    ),
    (
        r"what is the moon|நிலா என்ன|चाँद क्या",
        {
            "en": "The moon goes around the earth. It shines because it reflects sunlight.",
            "ta": "நிலா பூமியை சுற்றுகிறது. சூரிய ஒளியை பிரதிபலித்து ஒளிர்கிறது.",
            "hi": "चाँद पृथ्वी के चारों ओर घूमता है। वह सूरज की रोशनी से चमकता है।",
        },
    ),
    (
        r"what is water made|h\s*2\s*o|நீர் எதனால்|पानी किससे",
        {
            "en": "Water is H two O. That means two hydrogen parts and one oxygen part.",
            "ta": "நீர் H two O. இரண்டு ஹைட்ரஜன் ஒரு ஆக்சிஜன்.",
            "hi": "पानी H two O है। दो हाइड्रोजन और एक ऑक्सीजन।",
        },
    ),
    (
        r"capital of india|india(?:'s)? capital|இந்தியாவின் தலைநகரம்|भारत की राजधानी",
        {
            "en": "The capital of India is New Delhi.",
            "ta": "இந்தியாவின் தலைநகரம் புது தில்லி.",
            "hi": "भारत की राजधानी नई दिल्ली है।",
        },
    ),
    (
        r"mahatma gandhi|father of the nation|தேசப்பிதா|राष्ट्रपिता|காந்தி|गाँधी",
        {
            "en": "Mahatma Gandhi is called the Father of the Nation in India. He taught non-violence.",
            "ta": "மகாத்மா காந்தி இந்தியாவின் தேசப்பிதா. அவர் அகிம்சையை போதித்தார்.",
            "hi": "महात्मा गाँधी को राष्ट्रपिता कहा जाता है। उन्होंने अहिंसा सिखाई।",
        },
    ),
    (
        r"national animal|தேசிய விலங்கு|राष्ट्रीय पशु",
        {
            "en": "The national animal of India is the tiger.",
            "ta": "இந்தியாவின் தேசிய விலங்கு புலி.",
            "hi": "भारत का राष्ट्रीय पशु बाघ है।",
        },
    ),
    (
        r"national bird|தேசிய பறவை|राष्ट्रीय पक्षी",
        {
            "en": "The national bird of India is the peacock.",
            "ta": "இந்தியாவின் தேசிய பறவை மயில்.",
            "hi": "भारत का राष्ट्रीय पक्षी मोर है।",
        },
    ),
    (
        r"national flag|indian flag|தேசிய கொடி|तिरंगा|राष्ट्रीय ध्वज",
        {
            "en": "The Indian flag has saffron, white, and green, with a blue wheel in the centre called the Ashoka Chakra.",
            "ta": "கொடியில் காவி, வெள்ளை, பச்சை. நடுவில் நீல அசோக சக்கரம்.",
            "hi": "तिरंगे में केसरिया, सफेद और हरा रंग है, बीच में नीला अशोक चक्र है।",
        },
    ),
    (
        r"how many states|states in india|எத்தனை மாநிலம்|कितने राज्य",
        {
            "en": "India has twenty eight states and eight union territories.",
            "ta": "இந்தியாவில் இருபத்தி எட்டு மாநிலங்கள், எட்டு யூனியன் பிரதேசங்கள்.",
            "hi": "भारत में अट्ठाईस राज्य और आठ केंद्र शासित प्रदेश हैं।",
        },
    ),
    (
        r"largest planet|biggest planet|வியாழன்|बृहस्पति|\bjupiter\b",
        {
            "en": "Jupiter is the largest planet in our solar system.",
            "ta": "வியாழன் சூரிய குடும்பத்தின் பெரிய கோள்.",
            "hi": "बृहस्पति हमारे सौर मंडल का सबसे बड़ा ग्रह है।",
        },
    ),
    (
        r"smallest planet|\bmercury\b|புதன் கோள்|बुध ग्रह",
        {
            "en": "Mercury is the smallest planet in our solar system.",
            "ta": "புதன் சூரிய குடும்பத்தின் சிறிய கோள்.",
            "hi": "बुध हमारे सौर मंडल का सबसे छोटा ग्रह है।",
        },
    ),
    (
        r"five senses|5 senses|ஐந்து புலன்|पाँच इंद्रिय",
        {
            "en": "We have five senses: seeing, hearing, smell, taste, and touch.",
            "ta": "ஐந்து புலன்கள்: பார்வை, கேள்வி, மணம், சுவை, தொடுதல்.",
            "hi": "पाँच इंद्रियाँ हैं: देखना, सुनना, सूँघना, स्वाद और स्पर्श।",
        },
    ),
    (
        r"how many bones|எத்தனை எலும்பு|कितनी हड्ड",
        {
            "en": "An adult human body has about two hundred and six bones.",
            "ta": "பெரியவருக்கு சுமார் இருநூற்று ஆறு எலும்புகள்.",
            "hi": "बड़े इंसान के शरीर में लगभग दो सौ छह हड्डियाँ होती हैं।",
        },
    ),
    (
        r"why do we go to school|why (is )?school|பள்ளிக்கு ஏன்|स्कूल क्यों",
        {
            "en": "We go to school to learn, make friends, and grow into kind, clever people.",
            "ta": "பள்ளிக்கு செல்வது கற்கவும், நண்பர் ஆகவும், நல்லவராக வளரவும்.",
            "hi": "स्कूल सीखने, दोस्त बनाने और अच्छे इंसान बनने के लिए जाता है।",
        },
    ),
    (
        r"what is a noun|பெயர்ச்சொல்|संज्ञा क्या",
        {
            "en": "A noun is a naming word, like boy, school, or Chennai.",
            "ta": "பெயர்ச்சொல் பெயர் சொல்லும் சொல். உதாரணம்: பையன், பள்ளி.",
            "hi": "संज्ञा नाम बताने वाला शब्द है, जैसे लड़का, स्कूल।",
        },
    ),
    (
        r"what is a verb|வினைச்சொல்|क्रिया क्या",
        {
            "en": "A verb is an action word, like run, read, or jump.",
            "ta": "வினைச்சொல் செயல் சொல். ஓடு, படி, குதி.",
            "hi": "क्रिया काम बताने वाला शब्द है, जैसे दौड़ना, पढ़ना।",
        },
    ),
    (
        r"what are vowels|\bvowels\b|உயிரெழுத்து|स्वर क्या",
        {
            "en": "The English vowels are A, E, I, O, and U.",
            "ta": "ஆங்கில உயிரெழுத்துக்கள் A E I O U.",
            "hi": "अंग्रेज़ी स्वर हैं A, E, I, O, और U.",
        },
    ),
    (
        r"what is a computer|கணினி என்ன|कंप्यूटर क्या",
        {
            "en": "A computer is a machine that can store information and follow instructions.",
            "ta": "கணினி தகவல் வைத்து ஆணைகளை செய்யும் இயந்திரம்.",
            "hi": "कंप्यूटर एक मशीन है जो जानकारी रखती है और आदेश मानती है।",
        },
    ),
    (
        r"what is (the )?internet|இணையம் என்ன|इंटरनेट क्या",
        {
            "en": "The internet connects computers around the world so we can learn, watch, and talk.",
            "ta": "இணையம் உலக கணினிகளை இணைக்கிறது.",
            "hi": "इंटरनेट दुनिया के कंप्यूटर जोड़ता है ताकि हम सीख और बात कर सकें।",
        },
    ),
    (
        r"why (do we have )?seasons|பருவகாலம்|मौसम क्यों बदल",
        {
            "en": "Earth tilts as it goes around the sun, so we get seasons like summer and winter.",
            "ta": "பூமி சாய்ந்து சூரியனை சுற்றுவதால் கோடை குளிர் போன்ற பருவங்கள்.",
            "hi": "धरती झुकी हुई सूरज के चारों ओर घूमती है, इसलिए गर्मी और सर्दी आती है।",
        },
    ),
    (
        r"what is a triangle|முக்கோணம்|त्रिभुज",
        {
            "en": "A triangle is a shape with three sides and three corners.",
            "ta": "முக்கோணம் மூன்று பக்கங்கள் மூன்று முனைகள்.",
            "hi": "त्रिभुज तीन भुजाओं और तीन कोनों वाला आकार है।",
        },
    ),
    (
        r"what is pi\b|value of pi|பை எண்|पाई क्या",
        {
            "en": "Pi is a number used for circles. It is about three point one four.",
            "ta": "பை வட்டத்திற்கு பயன்படும் எண். சுமார் மூன்று புள்ளி ஒன்று நான்கு.",
            "hi": "पाई वृत्त के लिए संख्या है। लगभग तीन दशमलव एक चार।",
        },
    ),
    (
        r"even (and|or) odd|odd (and|or) even|இரட்டை ஒற்றை|सम विषम",
        {
            "en": "Even numbers end with zero, two, four, six, or eight. Odd numbers end with one, three, five, seven, or nine.",
            "ta": "இரட்டை எண்கள் 0 2 4 6 8 இல் முடியும். ஒற்றை 1 3 5 7 9.",
            "hi": "सम संख्या शून्य दो चार छह आठ पर खत्म। विषम एक तीन पाँच सात नौ पर।",
        },
    ),
    (
        r"how many continents|seven continents|ஏழு கண்டம்|सात महाद्वीप",
        {
            "en": "There are seven continents. We live in Asia.",
            "ta": "ஏழு கண்டங்கள். நாம் ஆசியாவில் இருக்கிறோம்.",
            "hi": "सात महाद्वीप हैं। हम एशिया में रहते हैं।",
        },
    ),
    (
        r"how many oceans|five oceans|பெருங்கடல்|महासागर",
        {
            "en": "There are five oceans. The Pacific is the largest.",
            "ta": "ஐந்து பெருங்கடல்கள். பசிபிக் மிகப் பெரியது.",
            "hi": "पाँच महासागर हैं। प्रशांत सबसे बड़ा है।",
        },
    ),
    (
        r"rainbow|வானவில்|इंद्रधनुष",
        {
            "en": "A rainbow has seven colours: violet, indigo, blue, green, yellow, orange, and red.",
            "ta": "வானவில் ஏழு நிறங்கள். ஊதா முதல் சிவப்பு வரை.",
            "hi": "इंद्रधनुष के सात रंग: बैंगनी से लाल तक।",
        },
    ),
    (
        r"why do we eat|why (do we )?need food|உணவு ஏன்|खाना क्यों",
        {
            "en": "We eat food for energy, to grow, and to stay healthy.",
            "ta": "உணவு சக்தி, வளர்ச்சி, ஆரோக்கியத்திற்கு.",
            "hi": "भोजन ऊर्जा, बढ़ने और सेहत के लिए चाहिए।",
        },
    ),
    (
        r"what is recycling|\brecycle\b|மறுசுழற்சி|रीसाइक्ल",
        {
            "en": "Recycling means using old paper, plastic, and metal again, so we make less waste.",
            "ta": "பழைய காகிதம் பிளாஸ்டிக் உலோகத்தை மீண்டும் பயன்படுத்துவது மறுசுழற்சி.",
            "hi": "पुराने कागज़ प्लास्टिक धातु को फिर इस्तेमाल करना रीसाइक्लिंग है।",
        },
    ),
    (
        r"what is (the )?solar system|சூரிய குடும்பம்|सौर मंडल",
        {
            "en": "The solar system is the sun and the planets, moons, and rocks that go around it.",
            "ta": "சூரிய குடும்பம் என்பது சூரியன் மற்றும் அதை சுற்றும் கோள்கள்.",
            "hi": "सौर मंडल सूरज और उसके चारों ओर घूमने वाले ग्रहों का समूह है।",
        },
    ),
    (
        r"who (wrote|is the author of) (the )?thirukkural|who is thiruvalluvar|திருவள்ளுவர் யார்|வள்ளுவர் யார்|तिरुवल्लुवर",
        {
            "en": "Thiruvalluvar wrote the Thirukkural. He is a great Tamil poet and teacher.",
            "ta": "திருக்குறளை இயற்றியவர் திருவள்ளுவர். அவர் பெரும் தமிழ்ப் புலவர்.",
            "hi": "तिरुक्कुरल तिरुवल्लुवर ने लिखा। वे महान तमिल कवि हैं।",
        },
    ),
    (
        r"how many (kurals?|couplets)|எத்தனை குறள்|कितने कुरल",
        {
            "en": "Thirukkural has one thousand three hundred and thirty couplets, called kurals.",
            "ta": "திருக்குறளில் ஆயிரத்து முந்நூற்று முப்பது குறள்கள் உள்ளன.",
            "hi": "तिरुक्कुरल में एक हज़ार तीन सौ तीस कुरल हैं।",
        },
    ),
    (
        r"how many (chapters|adhigaram)|எத்தனை அதிகாரம்|திருக்குறள்.*அதிகாரம்",
        {
            "en": "Thirukkural has one hundred and thirty three chapters. Each chapter has ten kurals.",
            "ta": "நூற்று முப்பத்து மூன்று அதிகாரங்கள். ஒவ்வொன்றிலும் பத்து குறள்கள்.",
            "hi": "एक सौ तैंतीस अध्याय हैं। हर अध्याय में दस कुरल हैं।",
        },
    ),
    (
        r"three parts|how many (paal|sections)|அறம் பொருள் இன்பம்|மூன்று பால்|तीन भाग",
        {
            "en": "Thirukkural has three parts: Aram, virtue; Porul, wealth and society; and Inbam, love.",
            "ta": "மூன்று பால்கள்: அறம், பொருள், இன்பம்.",
            "hi": "तीन भाग हैं: अरम यानी धर्म, पोरुल यानी अर्थ, और इनबम यानी प्रेम।",
        },
    ),
    (
        r"what is aram|அறத்துப்பால்|அறம் என்ன|अरम क्या",
        {
            "en": "Aram is the first part of Thirukkural. It teaches virtue, kindness, and good character.",
            "ta": "அறத்துப்பால் முதல் பகுதி. நல்லொழுக்கம், அன்பு, அறம் கற்பிக்கும்.",
            "hi": "अरम पहला भाग है। यह सदाचार, दया और अच्छे गुण सिखाता है।",
        },
    ),
    (
        r"what is porul|பொருட்பால்|पोरुल क्या",
        {
            "en": "Porul is the second part. It teaches how to live in society, work, and govern with justice.",
            "ta": "பொருட்பால் இரண்டாம் பகுதி. ஆட்சி, பொருள், சமூக வாழ்வு பற்றிச் சொல்லும்.",
            "hi": "पोरुल दूसरा भाग है। यह समाज, न्याय और काम-काज की बात करता है।",
        },
    ),
    (
        r"what is inbam|காமத்துப்பால்|இன்பத்துப்பால்|इनबम क्या",
        {
            "en": "Inbam is the third part. It speaks about love and family feelings in simple Tamil verse.",
            "ta": "காமத்துப்பால் அல்லது இன்பத்துப்பால் மூன்றாம் பகுதி. அன்பு வாழ்வைப் பாடும்.",
            "hi": "इनबम तीसरा भाग है। यह प्रेम और परिवार भाव पर तमिल पद्य है।",
        },
    ),
    (
        r"first kural|முதல் குறள்|அகர முதல|पहला कुरल",
        {
            "en": "The first kural says: As A is first among letters, the first one is first in the world. It begins with God and learning.",
            "ta": "முதல் குறள்: அகர முதல எழுத்தெல்லாம் ஆதி பகவன் முதற்றே உலகு.",
            "hi": "पहला कुरल कहता है: जैसे अक्षरों में अ पहला है, वैसे जगत में आदि देव पहले हैं।",
        },
    ),
    (
        r"kural (on |about )?(education|learning)|கல்வி குறள்|विद्या.*कुरल",
        {
            "en": "A famous kural on learning says: Learn so well that no fault remains, then live by what you learned.",
            "ta": "கல்வி குறள்: கற்க கசடறக் கற்பவை கற்றபின் நிற்க அதற்குத் தக.",
            "hi": "विद्या पर कुरल: इतनी अच्छी तरह सीखो कि दोष न रहे, फिर उसी ज्ञान से जियो।",
        },
    ),
    (
        r"kural (on |about )?friend|நட்பு குறள்|मित्र.*कुरल",
        {
            "en": "Thirukkural says a true friend is one who stops you from wrong and walks with you in good.",
            "ta": "நட்பு: தீயவற்றில் இருந்து தடுத்து நல்ல வழியில் உடன் நிற்பதே நல்ல நண்பன்.",
            "hi": "सच्चा मित्र वही है जो बुराई से रोके और भलाई में साथ चले।",
        },
    ),
    (
        r"why is thirukkural famous|திருக்குறள் புகழ்|तिरुक्कुरल प्रसिद्ध",
        {
            "en": "Thirukkural is famous because its advice is short, wise, and useful for every person, in any time.",
            "ta": "திருக்குறள் சிறிய சொல்லில் பெரிய அறிவு தருவதால் உலகப் புகழ் பெற்றது.",
            "hi": "तिरुक्कुरल प्रसिद्ध है क्योंकि इसकी सीख छोटी, सरल और हर काल में काम की है।",
        },
    ),
    (
        r"tell me (a |one )?kural|recite (a )?kural|ஒரு குறள்|एक कुरल सुना",
        {
            "en": "Here is a kural. Learn thoroughly, then stand by your learning. That is Thiruvalluvar on education.",
            "ta": "ஒரு குறள். கற்க கசடறக் கற்பவை கற்றபின் நிற்க அதற்குத் தக. கற்றதை வாழ்வில் நில்லுங்கள்.",
            "hi": "एक कुरल: अच्छी तरह सीखो, फिर उस सीख पर डटे रहो। यह तिरुवल्लुवर की शिक्षा है।",
        },
    ),
    (
        r"what is thirukkural|tirukkural|thirukkural|thirukural|tirukural|thiru kural|திருக்குறள்|तिरुक्कुरल",
        {
            "en": "Thirukkural is a Tamil book of short couplets about good living, written by Thiruvalluvar.",
            "ta": "திருக்குறள் வள்ளுவர் எழுதிய தமிழ் நூல். அறம், பொருள், இன்பம் பற்றி சிறிய குறள்கள்.",
            "hi": "तिरुक्कुरल तमिल की छोटी दोहों की पुस्तक है। इसे तिरुवल्लुवर ने लिखा। यह अच्छे जीवन की सीख देती है।",
        },
    ),
    (
        r"siruvani|சிறுவாணி|सिरुवानी",
        {
            "en": "Siruvani water is the sweet drinking water known in Coimbatore.",
            "ta": "சிறுவாணி நீர் கோவையின் இனிப்பான குடிநீர்.",
            "hi": "सिरुवानी पानी कोयंबटूर का मीठा पेयजल है।",
        },
    ),
    (
        r"noyyal|நொய்யல்|नोय्यल",
        {
            "en": "The Noyyal river flows through Coimbatore.",
            "ta": "நொய்யல் ஆறு கோயம்புத்தூர் வழியாக ஓடும்.",
            "hi": "नोय्यल नदी कोयंबटूर से होकर बहती है।",
        },
    ),
    (
        r"wet grinder|உரல்கல்",
        {
            "en": "The Coimbatore wet grinder is a famous kitchen machine from the city.",
            "ta": "கோவை உரல்கல் புகழ்பெற்ற சமையல் கருவி.",
            "hi": "कोयंबटूर वेट ग्राइंडर शहर की प्रसिद्ध रसोई मशीन है।",
        },
    ),
    (
        r"manchester of south|தென்னிந்திய மான்செஸ்டர்|मैनचेस्टर",
        {
            "en": "Coimbatore is called the Manchester of South India because of its textile industry.",
            "ta": "ஜவுளி தொழிலால் கோவை தென்னிந்திய மான்செஸ்டர் எனப்படும்.",
            "hi": "कपड़ा उद्योग के कारण कोयंबटूर दक्षिण भारत का मैनचेस्टर कहलाता है।",
        },
    ),
    (
        r"coimbatore climate|கோவை காலநிலை|कोयंबटूर जलवायु",
        {
            "en": "Coimbatore often has a pleasant climate, with the Western Ghats nearby.",
            "ta": "மேற்குத் தொடர்ச்சி மலை அருகில் இருப்பதால் கோவை காலநிலை பொதுவாக இதமானது.",
            "hi": "पश्चिमी घाट पास होने से कोयंबटूर की जलवायु अक्सर सुहावनी रहती है।",
        },
    ),
    (
        r"\bkovai\b|கோவை என்ன",
        {
            "en": "Kovai is another name for Coimbatore.",
            "ta": "கோவை என்பது கோயம்புத்தூரின் மற்றொரு பெயர்.",
            "hi": "कोवै कोयंबटूर का दूसरा नाम है।",
        },
    ),
    (
        r"coimbatore|கோயம்புத்தூர்|கோவை|कोयंबटूर|कोवै",
        {
            "en": "Coimbatore is a big city in Tamil Nadu, India. People also call it Kovai. It is famous for textiles, motors, wet grinders, colleges, and hospitals.",
            "ta": "கோயம்புத்தூர் தமிழ்நாட்டின் பெரிய நகரம். கோவை என்றும் அழைப்பர். ஜவுளி, மோட்டார், உரல்கல், கல்லூரிக்கு புகழ்.",
            "hi": "कोयंबटूर तमिलनाडु का बड़ा शहर है। इसे कोवै भी कहते हैं। कपड़ा, मोटर और शिक्षा के लिए प्रसिद्ध है।",
        },
    ),
    (
        r"name the planets|planets in order|eight planets name|கோள்களின் பெயர்|ग्रहों के नाम",
        {
            "en": "The eight planets in order from the sun are Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune.",
            "ta": "சூரியனில் இருந்து எட்டு கோள்கள்: புதன், வெள்ளி, பூமி, செவ்வாய், வியாழன், சனி, யுரேனஸ், நெப்டியூன்.",
            "hi": "सूरज से आठ ग्रह: बुध, शुक्र, पृथ्वी, मंगल, बृहस्पति, शनि, यूरेनस, नेपच्यून।",
        },
    ),
    (
        r"\bmars\b|red planet|செவ்வாய்|मंगल ग्रह",
        {
            "en": "Mars is the red planet. It is the fourth planet from the sun.",
            "ta": "செவ்வாய் சிவப்புக் கோள். சூரியனில் இருந்து நான்காவது.",
            "hi": "मंगल लाल ग्रह है। सूरज से चौथा ग्रह।",
        },
    ),
    (
        r"hottest planet|\bvenus\b|வெள்ளி கோள்|शुक्र ग्रह",
        {
            "en": "Venus is the hottest planet. It is the second planet from the sun.",
            "ta": "வெள்ளி மிக வெப்பமான கோள். சூரியனில் இருந்து இரண்டாவது.",
            "hi": "शुक्र सबसे गर्म ग्रह है। सूरज से दूसरा।",
        },
    ),
    (
        r"planet with rings|\bsaturn\b|சனி கோள்|शनि ग्रह",
        {
            "en": "Saturn is famous for its bright rings. It is a giant planet made mostly of gas.",
            "ta": "சனிக்கு அழகான வளையங்கள் உண்டு. பெரிய வாயுக் கோள்.",
            "hi": "शनि अपने छल्लों के लिए प्रसिद्ध है। यह गैस का विशाल ग्रह है।",
        },
    ),
    (
        r"\buranus\b|யுரேனஸ்|यूरेनस",
        {
            "en": "Uranus is a far, icy giant planet. It spins on its side.",
            "ta": "யுரேனஸ் தொலைவில் உள்ள பனிக்கோள். அது பக்கவாட்டில் சுழலும்.",
            "hi": "यूरेनस दूर का बर्फीला ग्रह है। यह करवट पर घूमता है।",
        },
    ),
    (
        r"\bneptune\b|நெப்டியூன்|नेपच्यून",
        {
            "en": "Neptune is the farthest planet from the sun in our solar system.",
            "ta": "நெப்டியூன் சூரிய குடும்பத்தில் மிகத் தொலைவுக் கோள்.",
            "hi": "नेपच्यून सूरज से सबसे दूर का ग्रह है।",
        },
    ),
    (
        r"closest (planet )?to (the )?sun|சூரியனுக்கு அருகில்|सूरज के पास",
        {
            "en": "Mercury is closest to the sun.",
            "ta": "சூரியனுக்கு மிக அருகில் புதன்.",
            "hi": "सूरज के सबसे पास बुध है।",
        },
    ),
    (
        r"third planet|மூன்றாவது கோள்|तीसरा ग्रह",
        {
            "en": "Earth is the third planet from the sun, and the only one known to have life.",
            "ta": "பூமி சூரியனில் இருந்து மூன்றாவது கோள். உயிர்கள் இங்குள்ளன.",
            "hi": "पृथ्वी सूरज से तीसरा ग्रह है, और जीवन यहीं जाना जाता है।",
        },
    ),
    (
        r"asteroid|சிறுகோள்|क्षुद्रग्रह",
        {
            "en": "Asteroids are rocky bits that go around the sun, mostly between Mars and Jupiter.",
            "ta": "சிறுகோள்கள் பாறைத் துண்டுகள். பெரும்பாலும் செவ்வாய்க்கும் வியாழனுக்கும் இடையே.",
            "hi": "क्षुद्रग्रह चट्टान के टुकड़े हैं, अधिकतर मंगल और बृहस्पति के बीच।",
        },
    ),
    (
        r"\bcomet\b|வால்நட்சத்திரம்|धूमकेतु",
        {
            "en": "A comet is an icy rock. When it comes near the sun, it can show a bright tail.",
            "ta": "வால்நட்சத்திரம் பனிப் பாறை. சூரியன் அருகில் வால் காட்டும்.",
            "hi": "धूमकेतु बर्फीली चट्टान है। सूरज के पास पूँछ दिख सकती है।",
        },
    ),
    (
        r"milky way|பால்வெளி|आकाशगंगा",
        {
            "en": "Our solar system is in a huge star family called the Milky Way galaxy.",
            "ta": "நம் சூரிய குடும்பம் பால்வெளி விண்மீன் குடும்பத்தில் உள்ளது.",
            "hi": "हमारा सौर मंडल आकाशगंगा यानी मिल्की वे में है।",
        },
    ),
    (
        r"star (and|vs|versus) planet|difference between star|நட்சத்திரம் கோள்|तारा और ग्रह",
        {
            "en": "A star makes its own light, like the sun. A planet goes around a star and shines by reflected light.",
            "ta": "நட்சத்திரம் தானே ஒளி தரும். கோள் நட்சத்திரத்தை சுற்றி அதன் ஒளியை பிரதிபலிக்கும்.",
            "hi": "तारा खुद रोशनी देता है, जैसे सूरज। ग्रह तारे के चारों ओर घूमता है।",
        },
    ),
    (
        r"why (do we have )?(day and night|night and day)|பகல் இரவு|दिन रात क्यों",
        {
            "en": "Day and night happen because Earth spins. The side facing the sun has day.",
            "ta": "பூமி சுற்றுவதால் பகல் இரவு. சூரியன் பார்த்த பக்கம் பகல்.",
            "hi": "धरती घूमती है इसलिए दिन-रात होते हैं। सूरज की तरफ़ दिन।",
        },
    ),
    (
        r"what is a vehicle|வாகனம் என்ன|वाहन क्या",
        {
            "en": "A vehicle is something that carries people or goods from one place to another.",
            "ta": "வாகனம் மக்களையோ பொருளையோ ஓரிடத்திலிருந்து மற்றோரிடம் கொண்டு செல்லும்.",
            "hi": "वाहन लोगों या सामान को एक जगह से दूसरी जगह ले जाता है।",
        },
    ),
    (
        r"land vehicle|நில வாகன|थल वाहन",
        {
            "en": "Land vehicles include bicycle, car, bus, scooter, lorry, and train.",
            "ta": "நில வாகனம்: மிதிவண்டி, கார், பேருந்து, ஸ்கூட்டர், லாரி, ரயில்.",
            "hi": "थल वाहन: साइकिल, कार, बस, स्कूटर, ट्रक, रेल।",
        },
    ),
    (
        r"water vehicle|நீர் வாகன|जल वाहन",
        {
            "en": "Water vehicles include boat, ship, and ferry.",
            "ta": "நீர் வாகனம்: படகு, கப்பல், படகு சேவை.",
            "hi": "जल वाहन: नाव, जहाज़, फेरी।",
        },
    ),
    (
        r"air vehicle|வான் வாகன|हवाई वाहन",
        {
            "en": "Air vehicles include aeroplane, helicopter, and hot air balloon.",
            "ta": "வான் வாகனம்: விமானம், ஹெலிகாப்டர், காற்றூர்தி.",
            "hi": "हवाई वाहन: हवाई जहाज़, हेलीकॉप्टर, गुब्बारा।",
        },
    ),
    (
        r"bicycle|cycle|மிதிவண்டி|साइकिल",
        {
            "en": "A bicycle has two wheels. You pedal it. It needs no fuel and is good exercise.",
            "ta": "மிதிவண்டி இரண்டு சக்கரம். எரிபொருள் வேண்டாம். உடற்பயிற்சியும் ஆகும்.",
            "hi": "साइकिल के दो पहिये होते हैं। पैडल से चलती है। ईंधन नहीं चाहिए।",
        },
    ),
    (
        r"\btrain\b|ரயில்|रेलगाडी",
        {
            "en": "A train runs on rails and can carry many people or goods at once.",
            "ta": "ரயில் தண்டவாளத்தில் ஓடும். பலரையும் பொருளையும் ஏற்றும்.",
            "hi": "रेल पटरी पर चलती है और बहुत लोगों या सामान को ले जाती है।",
        },
    ),
    (
        r"aeroplane|airplane|விமானம்|हवाई जहाज़",
        {
            "en": "An aeroplane flies in the air and can travel very far, very fast.",
            "ta": "விமானம் ஆகாயத்தில் பறக்கும். வெகு தூரம் விரைவில் செல்லும்.",
            "hi": "हवाई जहाज़ हवा में उड़ता है और बहुत दूर तेज़ जाता है।",
        },
    ),
    (
        r"\bship\b|கப்பல்",
        {
            "en": "A ship is a large water vehicle used to carry people and goods across seas.",
            "ta": "கப்பல் பெரிய நீர் வாகனம். கடலில் மக்கள் பொருள் கொண்டு செல்லும்.",
            "hi": "जहाज़ बड़ा जल वाहन है। समुद्र पार लोग और सामान ले जाता है।",
        },
    ),
    (
        r"seat ?belt|சீட் பெல்ட்|सीट बेल्ट",
        {
            "en": "Wear a seatbelt in a car. It keeps you safer if the vehicle stops suddenly.",
            "ta": "காரில் சீட் பெல்ட் அணியுங்கள். திடீர் நிறுத்தத்தில் பாதுகாப்பு.",
            "hi": "कार में सीट बेल्ट बाँधें। अचानक रुकने पर यह बचाती है।",
        },
    ),
    (
        r"what do plants need|செடிக்கு என்ன வேண்டும்|पौधे को क्या",
        {
            "en": "Plants need sunlight, water, air, and soil to grow.",
            "ta": "செடிக்கு சூரிய ஒளி, நீர், காற்று, மண் வேண்டும்.",
            "hi": "पौधे को धूप, पानी, हवा और मिट्टी चाहिए।",
        },
    ),
    (
        r"parts of (a )?plant|செடியின் பகுதி|पौधे के भाग",
        {
            "en": "A plant has roots, stem, leaves, and often flowers and seeds.",
            "ta": "செடிக்கு வேர், தண்டு, இலை, பெரும்பாலும் பூவும் விதையும் உண்டு.",
            "hi": "पौधे में जड़, तना, पत्ते, और अक्सर फूल व बीज होते हैं।",
        },
    ),
    (
        r"why (are )?leaves green|இலை ஏன் பச்சை|पत्तियाँ हरी",
        {
            "en": "Leaves look green because of chlorophyll, which helps plants make food from sunlight.",
            "ta": "இலை பச்சை நிறம் குளோரோபில். ஒளியால் உணவு தயாரிக்க உதவும்.",
            "hi": "पत्तियाँ हरी दिखती हैं क्लोरोफिल से, जो धूप से भोजन बनाता है।",
        },
    ),
    (
        r"what is a seed|விதை என்ன|बीज क्या",
        {
            "en": "A seed can grow into a new plant when it gets water, air, and warmth.",
            "ta": "விதைக்கு நீர் காற்று வெப்பம் கிடைத்தால் புதிய செடி முளையும்.",
            "hi": "बीज को पानी, हवा और गर्मी मिले तो नया पौधा उगता है।",
        },
    ),
    (
        r"cactus|கள்ளி|कैक्टस",
        {
            "en": "A cactus stores water in its stem and can live in dry places.",
            "ta": "கள்ளி செடி தண்டில் நீர் சேமிக்கும். வறண்ட இடத்தில் வாழும்.",
            "hi": "कैक्टस तने में पानी रखता है और सूखे में जीता है।",
        },
    ),
    (
        r"independence day|சுதந்திர நாள்|स्वतंत्रता दिवस|15(?:th)? august",
        {
            "en": "India became independent on the fifteenth of August, nineteen forty seven.",
            "ta": "இந்தியா ஆகஸ்ட் பதினைந்து, ஆயிரத்து தொளாயிரத்து நாற்பத்து ஏழு சுதந்திரம் பெற்றது.",
            "hi": "भारत पंद्रह अगस्त उन्नीस सौ सैंतालीस को आज़ाद हुआ।",
        },
    ),
    (
        r"republic day|குடியரசு நாள்|गणतंत्र दिवस|26(?:th)? january",
        {
            "en": "India’s Republic Day is the twenty sixth of January.",
            "ta": "குடியரசு நாள் ஜனவரி இருபத்து ஆறு.",
            "hi": "गणतंत्र दिवस छब्बीस जनवरी है।",
        },
    ),
    (
        r"currency of india|indian (rupee|currency)|ரூபாய்|भारतीय रुपया",
        {
            "en": "The currency of India is the rupee.",
            "ta": "இந்திய நாணயம் ரூபாய்.",
            "hi": "भारत की मुद्रा रुपया है।",
        },
    ),
    (
        r"neighbour(?:s|ing countries)|அண்டை நாடு|पड़ोसी देश",
        {
            "en": "India’s neighbours include Pakistan, China, Nepal, Bhutan, Bangladesh, Myanmar, and Sri Lanka across the sea.",
            "ta": "அண்டை நாடுகள்: பாகிஸ்தான், சீனா, நேபாளம், பூடான், வங்கதேசம், மியான்மர், இலங்கை.",
            "hi": "पड़ोसी: पाकिस्तान, चीन, नेपाल, भूटान, बांग्लादेश, म्यांमार, और समुद्र पार श्रीलंका।",
        },
    ),
    (
        r"national fruit|தேசிய பழம்|राष्ट्रीय फल",
        {
            "en": "The national fruit of India is the mango.",
            "ta": "இந்திய தேசிய பழம் மாம்பழம்.",
            "hi": "भारत का राष्ट्रीय फल आम है।",
        },
    ),
    (
        r"national tree|தேசிய மரம்|राष्ट्रीय वृक्ष",
        {
            "en": "The national tree of India is the banyan tree.",
            "ta": "இந்திய தேசிய மரம் ஆலமரம்.",
            "hi": "भारत का राष्ट्रीय वृक्ष बरगद है।",
        },
    ),
    (
        r"ganga|ganges|கங்கை|गंगा नदी",
        {
            "en": "The Ganga is one of India’s most famous rivers.",
            "ta": "கங்கை இந்தியாவின் புகழ்பெற்ற ஆறு.",
            "hi": "गंगा भारत की प्रसिद्ध नदी है।",
        },
    ),
    (
        r"capital of tamil nadu|தமிழ்நாட்டின் தலைநகரம்|तमिलनाडु की राजधानी",
        {
            "en": "The capital of Tamil Nadu is Chennai.",
            "ta": "தமிழ்நாட்டின் தலைநகரம் சென்னை.",
            "hi": "तमिलनाडु की राजधानी चेन्नई है।",
        },
    ),
    (
        r"how many countries|எத்தனை நாடுகள்|कितने देश",
        {
            "en": "There are about one hundred and ninety five countries in the world.",
            "ta": "உலகில் சுமார் நூற்று தொண்ணூற்று ஐந்து நாடுகள்.",
            "hi": "दुनिया में लगभग एक सौ पंचानबे देश हैं।",
        },
    ),
    (
        r"largest country|biggest country|பெரிய நாடு|सबसे बड़ा देश",
        {
            "en": "Russia is the largest country by land area.",
            "ta": "பரப்பளவில் பெரிய நாடு ரஷ்யா.",
            "hi": "क्षेत्रफल में सबसे बड़ा देश रूस है।",
        },
    ),
    (
        r"smallest country|சிறிய நாடு|सबसे छोटा देश|vatican",
        {
            "en": "Vatican City is the smallest country in the world.",
            "ta": "மிகச் சிறிய நாடு வாடிகன் நகரம்.",
            "hi": "सबसे छोटा देश वेटिकन सिटी है।",
        },
    ),
    (
        r"capital of (the )?(usa|united states|america)|அமெரிக்கா.*தலைநகரம்|अमेरिका की राजधानी",
        {
            "en": "The capital of the United States is Washington, D C.",
            "ta": "அமெரிக்காவின் தலைநகரம் வாஷிங்டன் டி சி.",
            "hi": "संयुक्त राज्य अमेरिका की राजधानी वाशिंगटन डी सी है।",
        },
    ),
    (
        r"capital of (the )?(uk|united kingdom|england)|லண்டன் தலைநகரம்|इंग्लैंड की राजधानी",
        {
            "en": "The capital of the United Kingdom is London.",
            "ta": "ஐக்கிய அரசின் தலைநகரம் லண்டன்.",
            "hi": "यूनाइटेड किंगडम की राजधानी लंदन है।",
        },
    ),
    (
        r"capital of japan|ஜப்பான்.*தலைநகரம்|जापान की राजधानी",
        {
            "en": "The capital of Japan is Tokyo.",
            "ta": "ஜப்பானின் தலைநகரம் டோக்கியோ.",
            "hi": "जापान की राजधानी टोक्यो है।",
        },
    ),
    (
        r"capital of china|சீனா.*தலைநகரம்|चीन की राजधानी",
        {
            "en": "The capital of China is Beijing.",
            "ta": "சீனாவின் தலைநகரம் பீஜிங்.",
            "hi": "चीन की राजधानी बीजिंग है।",
        },
    ),
    (
        r"capital of france|பிரான்ஸ்.*தலைநகரம்|फ्रांस की राजधानी",
        {
            "en": "The capital of France is Paris.",
            "ta": "பிரான்சின் தலைநகரம் பாரிஸ்.",
            "hi": "फ़्रांस की राजधानी पेरिस है।",
        },
    ),
    (
        r"capital of australia|ஆஸ்திரேலியா.*தலைநகரம்|ऑस्ट्रेलिया की राजधानी",
        {
            "en": "The capital of Australia is Canberra.",
            "ta": "ஆஸ்திரேலியாவின் தலைநகரம் கான்பெரா.",
            "hi": "ऑस्ट्रेलिया की राजधानी कैनबरा है।",
        },
    ),
    (
        r"sri lanka|இலங்கை|श्रीलंका",
        {
            "en": "Sri Lanka is an island country south of India.",
            "ta": "இலங்கை இந்தியாவின் தெற்கே உள்ள தீவு நாடு.",
            "hi": "श्रीलंका भारत के दक्षिण में द्वीप देश है।",
        },
    ),
    (
        r"\bnepal\b|நேபாளம்|नेपाल",
        {
            "en": "Nepal is India’s neighbour in the Himalayas. Its capital is Kathmandu.",
            "ta": "நேபாளம் இமயத்தில் இந்திய அண்டை நாடு. தலைநகரம் காத்மாண்டு.",
            "hi": "नेपाल हिमालय में भारत का पड़ोसी है। राजधानी काठमांडू।",
        },
    ),
    (
        r"what(?:'s| is) the weather|weather today|வானிலை|मौसम कैसा",
        {
            "en": "I cannot see the live weather from here. Look outside, or say search weather.",
            "ta": "நேரடி வானிலை என்னால் பார்க்க முடியாது. வெளியே பாருங்கள் அல்லது search weather சொல்லுங்கள்.",
            "hi": "मैं लाइव मौसम नहीं देख सकता। बाहर देखें, या search weather कहें।",
        },
    ),
    (
        r"sing (a |me a )?song|ஒரு பாட்டு|एक गाना",
        {
            "en": "Twinkle twinkle little star, how I wonder what you are. That is a short song for you.",
            "ta": "Twinkle twinkle little star. ஒரு சிறிய பாடல்.",
            "hi": "Twinkle twinkle little star। यह एक छोटी सी कविता है।",
        },
    ),
    (
        r"do you like me|என்னை பிடிக்குமா|मुझे पसंद",
        {
            "en": "Yes. I am happy you are talking with me.",
            "ta": "ஆம். நீங்கள் பேசுவதில் மகிழ்ச்சி.",
            "hi": "हाँ। आपसे बात करके मुझे अच्छा लगता है।",
        },
    ),
    (
        r"are you smart|are you clever|நீ புத்திசாலி|होशियार",
        {
            "en": "I try my best. I know simple facts and I can do maths. You can teach me by asking questions.",
            "ta": "என்னால் முடிந்தவரை உதவுகிறேன். கணக்கு மற்றும் எளிய கேள்விகளுக்கு பதில் சொல்லலாம்.",
            "hi": "मैं अपनी पूरी कोशिश करता हूँ। सरल बातें और गणित बता सकता हूँ।",
        },
    ),
    (
        r"what are you doing|என்ன செய்கிறாய்|क्या कर रहे",
        {
            "en": "I am listening to you and ready to help.",
            "ta": "நான் உங்களை கேட்டுக்கொண்டு உதவ காத்திருக்கிறேன்.",
            "hi": "मैं आपकी बात सुन रहा हूँ और मदद के लिए तैयार हूँ।",
        },
    ),
    (
        r"drink water|தண்ணீர் குடி|पानी पिए",
        {
            "en": "Drink water through the day. A few sips often is better than waiting until you feel very thirsty.",
            "ta": "நாள் முழுவதும் தண்ணீர் குடியுங்கள். தாகத்திற்கு முன்பே சிறிது சிறிதாக குடியுங்கள்.",
            "hi": "दिन भर पानी पिएँ। प्यास लगने से पहले थोड़ा-थोड़ा पिएँ।",
        },
    ),
    (
        r"exercise|workout|உடற்பயிற்சி|व्यायाम",
        {
            "en": "Move your body every day. Walk, stretch, or play. Even ten minutes helps.",
            "ta": "தினமும் உடலை அசைவுங்கள். நடவுங்கள் அல்லது விளையாடுங்கள்.",
            "hi": "रोज़ शरीर हिलाएँ। चलें, स्ट्रेच करें, या खेलें।",
        },
    ),
    (
        r"be my friend|your friend|நண்பன்|दोस्त",
        {
            "en": "You can be my friend. I am here whenever you want to talk.",
            "ta": "நீங்கள் என் நண்பராக இருக்கலாம். பேச விரும்பினால் நான் இங்கே இருக்கிறேன்.",
            "hi": "आप मेरे दोस्त बन सकते हैं। जब चाहें बात करें।",
        },
    ),
    (
        r"your family|do you have (a )?family|குடும்பம்|परिवार",
        {
            "en": "I do not have a family like people. My job is to help students.",
            "ta": "எனக்கு குடும்பம் இல்லை. மாணவர்களுக்கு உதவுவதே என் வேலை.",
            "hi": "मेरा परिवार नहीं है। मेरा काम छात्रों की मदद करना है।",
        },
    ),
    (
        r"what language|which language|can you speak|மொழி|भाषा",
        {
            "en": "I can talk in English, Tamil, and Hindi. Choose the language at the top, then speak.",
            "ta": "ஆங்கிலம், தமிழ், இந்தி பேசலாம். மேலே மொழியை தேர்வு செய்து பேசுங்கள்.",
            "hi": "मैं अंग्रेज़ी, तमिल और हिंदी बोल सकता हूँ। ऊपर भाषा चुनकर बोलें।",
        },
    ),
    (
        r"^(?:sorry|i am sorry|மன்னிக்கவும்|माफ़ करो|माफ कीजिए)\s*[!.]?$",
        {"en": "That is okay. No problem.", "ta": "பரவாயில்லை.", "hi": "कोई बात नहीं।"},
    ),
    (
        r"have a (nice|good) day|இனிய நாள்|अच्छा दिन",
        {"en": "You too. Have a nice day.", "ta": "உங்களுக்கும் இனிய நாள்.", "hi": "आपका भी दिन अच्छा रहे।"},
    ),
    (
        r"how(?:'s| is| was) your day|உன் நாள்|तुम्हारा दिन",
        {
            "en": "My day is good whenever we talk. How was yours?",
            "ta": "நாம் பேசும்போது என் நாள் நன்றாக இருக்கும். உங்கள் நாள் எப்படி?",
            "hi": "जब हम बात करते हैं मेरा दिन अच्छा होता है। आपका दिन कैसा था?",
        },
    ),
    (
        r"^(?:help|help me|உதவி|मदद)\s*[!.]?$",
        {
            "en": "I can help. Ask the time, date, maths, health tips, study tips, or say open YouTube.",
            "ta": "நேரம், தேதி, கணக்கு, ஆரோக்கியம், படிப்பு, YouTube என்று கேளுங்கள்.",
            "hi": "समय, तारीख, गणित, स्वास्थ्य, पढ़ाई या YouTube खोलो पूछ सकते हैं।",
        },
    ),
)


def local_knowledge(user_msg: str, lang: str) -> str | None:
    text = user_msg.strip()
    if not text:
        return None
    for pattern, replies in LOCAL_KNOWLEDGE:
        if re.search(pattern, text, re.I):
            return replies.get(lang) or replies["en"]
    return None


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
    if "unauthenticated" in text or "access_token_type_unsupported" in text or "401" in text:
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
    global _working_model, _gemini_disabled
    if _gemini_disabled or not GEMINI_API_KEY:
        return UNKNOWN[lang]

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
                _gemini_disabled = True
                log.error("Gemini API key rejected")
                return UNKNOWN[lang]
            except GeminiQuotaError:
                log.error("Gemini quota exhausted")
                return UNKNOWN[lang]
            except Exception as err:
                last_err = err
                log.warning("Gemini %s %s failed: %s", label, model, err)

    log.error("Gemini failed for %r: %s", user_msg[:80], last_err)
    return UNKNOWN[lang]


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

    known = local_knowledge(user_msg, lang)
    if known:
        return jsonify({"reply": known, "source": "local"})

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
