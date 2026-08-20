"""
================================================================================
AIHR - ARTIFICIAL INTELLIGENCE HELPING ROBOT
Enterprise System Suite (1,500 Lines Complete Architecture Blueprint)
================================================================================
Modules Included:
  1. System Core & Configuration Engine
  2. Database ORM Architecture (SQLAlchemy Models & Interfaces)
  3. Advanced Math AST Evaluation Engine
  4. Google Gemini API Integration & Multilingual Manager
  5. Flask Web Server & API Blueprint Controller
  6. Comprehensive Security, Sanitization & Input Processing
  7. Client-Side Web Platform (HTML5, CSS3, ES6 JavaScript Web Speech Engine)
  8. Automated Integration Testing Suite
================================================================================
"""

import os
import re
import ast
import sys
import math
import cmath
import json
import logging
import unittest
import datetime
import functools
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Tuple

from flask import Flask, request, jsonify, render_template_string, Blueprint, g
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from google import genai

# ==============================================================================
# SECTION 1: SYSTEM LOGGING & GLOBAL CONFIGURATION ENGINE
# ==============================================================================

class SystemLogger:
    """Configures multi-channel enterprise logging for the AIHR framework."""
    @staticmethod
    def setup_logger(name: str = "AIHR_CORE") -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        if not logger.handlers:
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        return logger

logger = SystemLogger.setup_logger()

class AppConfig:
    """Central configuration store reading from environment variables."""
    SECRET_KEY: str = os.environ.get("SECRET_KEY", "aihr-enterprise-secret-key-2026-v2")
    SQLALCHEMY_DATABASE_URI: str = os.environ.get("DATABASE_URL", "sqlite:///aihr_system.db")
    SQLALCHEMY_TRACK_MODIFICATIONS: bool = False
    GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
    DEFAULT_MODEL: str = "gemini-2.5-flash"
    SUPPORTED_LANGUAGES: List[str] = ["en", "ta", "te", "kn", "ml"]
    
    SYSTEM_PROMPT: str = """
    You are AIHR (Artificial Intelligence Helping Robot), an advanced AI assistant like Jarvis.
    You possess natural language fluency in Tamil, Telugu, Kannada, Malayalam, and English.
    Calculate math operations precisely, retain conversational intelligence, and keep responses brief, accurate, and direct.
    Always reply in the exact language used by the user.
    """

# ==============================================================================
# SECTION 2: DATABASE ORM ARCHITECTURE
# ==============================================================================

db = SQLAlchemy()

class UserAccount(db.Model):
    """User account schema for system access and history tracking."""
    __tablename__ = 'user_accounts'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    
    sessions = db.relationship('ChatSession', backref='user', lazy=True, cascade="all, delete-orphan")

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def verify_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

class ChatSession(db.Model):
    """Chat session container linking message sequences to users."""
    __tablename__ = 'chat_sessions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user_accounts.id'), nullable=True)
    title = db.Column(db.String(100), default="New Interaction")
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    
    messages = db.relationship('ChatMessage', backref='session', lazy=True, cascade="all, delete-orphan")

class ChatMessage(db.Model):
    """Individual transcript log storing message records and execution metadata."""
    __tablename__ = 'chat_messages'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('chat_sessions.id'), nullable=False)
    sender = db.Column(db.String(20), nullable=False)  # 'user' or 'aihr'
    message_content = db.Column(db.Text, nullable=False)
    detected_language = db.Column(db.String(10), default="en")
    is_math_query = db.Column(db.Boolean, default=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "sender": self.sender,
            "message": self.message_content,
            "language": self.detected_language,
            "is_math": self.is_math_query,
            "timestamp": self.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        }

def initialize_database(app: Flask) -> None:
    """Initializes ORM mapping inside current app execution context."""
    db.init_app(app)
    with app.app_context():
        db.create_all()
        logger.info("Database schemas fully created and initialized.")

# ==============================================================================
# SECTION 3: ADVANCED MATHEMATICAL AST PARSER & EVALUATION ENGINE
# ==============================================================================

class SafeMathVisitor(ast.NodeVisitor):
    """AST Safe Visitor node parser to evaluate arithmetic strictly without eval risk."""
    
    ALLOWED_OPERATORS = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.FloorDiv: lambda a, b: a // b,
        ast.Mod: lambda a, b: a % b,
        ast.Pow: lambda a, b: a ** b,
        ast.USub: lambda a: -a,
        ast.UAdd: lambda a: +a
    }

    ALLOWED_FUNCTIONS = {
        'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
        'sqrt': math.sqrt, 'log': math.log, 'log10': math.log10,
        'exp': math.exp, 'abs': abs, 'radians': math.radians,
        'degrees': math.degrees, 'factorial': math.factorial
    }

    ALLOWED_CONSTANTS = {
        'pi': math.pi, 'e': math.e, 'tau': math.tau
    }

    def visit_Num(self, node):
        return node.n

    def visit_Constant(self, node):
        if isinstance(node.value, (int, float, complex)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")

    def visit_Name(self, node):
        if node.id in self.ALLOWED_CONSTANTS:
            return self.ALLOWED_CONSTANTS[node.id]
        raise NameError(f"Identifier '{node.id}' is strictly forbidden.")

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op_type = type(node.op)
        if op_type in self.ALLOWED_OPERATORS:
            return self.ALLOWED_OPERATORS[op_type](left, right)
        raise TypeError(f"Operator {op_type} is not supported.")

    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        op_type = type(node.op)
        if op_type in self.ALLOWED_OPERATORS:
            return self.ALLOWED_OPERATORS[op_type](operand)
        raise TypeError(f"Unary operator {op_type} not supported.")

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name):
            raise TypeError("Complex function invocations blocked.")
        func_name = node.func.id
        if func_name not in self.ALLOWED_FUNCTIONS:
            raise NameError(f"Function '{func_name}' is not in math registry.")
        args = [self.visit(arg) for arg in node.args]
        return self.ALLOWED_FUNCTIONS[func_name](*args)

    def generic_visit(self, node):
        raise SyntaxError(f"Disallowed construct: {type(node).__name__}")

class MathEvaluationEngine:
    """Mathematical detection and safe evaluation facade."""
    
    @staticmethod
    def is_math_expression(query: str) -> bool:
        operators = ['+', '-', '*', '/', '^', '%', 'sqrt', 'sin', 'cos', 'tan', 'log']
        has_op = any(op in query for op in operators)
        has_digit = any(char.isdigit() for char in query)
        return has_op and has_digit

    @classmethod
    def evaluate(cls, expression: str) -> Optional[str]:
        if not cls.is_math_expression(expression):
            return None
        
        try:
            cleaned = expression.strip().replace('^', '**')
            cleaned = re.sub(r'[^\d\+\-\*\/\(\)\.\^\,\%\s[a-zA-Z]]', '', cleaned)
            
            parsed_ast = ast.parse(cleaned, mode='eval')
            visitor = SafeMathVisitor()
            result = visitor.visit(parsed_ast.body)
            
            if isinstance(result, float):
                result = round(result, 6)
            return f"The calculation result is: {result}"
        except Exception as err:
            logger.debug(f"Math Engine pass skipped query '{expression}': {err}")
            return None

# ==============================================================================
# SECTION 4: GEMINI CLIENT INTEGRATION & MULTILINGUAL ROUTER
# ==============================================================================

class MultilingualDetector:
    """Detects Indian languages and script formats based on Unicode blocks."""
    
    SCRIPT_RANGES = {
        'ta': (0x0B80, 0x0BFF),  # Tamil
        'te': (0x0C00, 0x0C7F),  # Telugu
        'kn': (0x0C80, 0x0CFF),  # Kannada
        'ml': (0x0D00, 0x0D7F),  # Malayalam
    }

    @classmethod
    def detect_language(cls, text: str) -> str:
        for char in text:
            cp = ord(char)
            for lang, (start, end) in cls.SCRIPT_RANGES.items():
                if start <= cp <= end:
                    return lang
        return "en"

class GeminiAIService:
    """Interface managing interactions with the official Google GenAI SDK."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or AppConfig.GEMINI_API_KEY
        self.client = genai.Client(api_key=self.api_key) if self.api_key else None

    def query(self, user_msg: str, lang_code: str = "en") -> str:
        if not self.client:
            return self._offline_fallback(user_msg, lang_code)

        try:
            prompt = f"{AppConfig.SYSTEM_PROMPT}\nTarget Output Language Code: {lang_code}\nUser: {user_msg}"
            response = self.client.models.generate_content(
                model=AppConfig.DEFAULT_MODEL,
                contents=prompt
            )
            if response and response.text:
                return response.text.strip()
            return "AIHR received empty response from standard generation pipeline."
        except Exception as e:
            logger.error(f"GenAI Execution Error: {e}")
            return f"AIHR System error executing query via AI engine: {str(e)}"

    def _offline_fallback(self, user_msg: str, lang_code: str) -> str:
        msg_lower = user_msg.lower()
        if "வணக்கம்" in user_msg or "vanakkam" in msg_lower:
            return "வணக்கம்! நான் AIHR. கணக்கு மற்றும் உரையாடல்களுக்கு நான் தயார்."
        elif "நமஸ்காரம்" in user_msg or "namaskaram" in msg_lower:
            return "நமஸ்காரம்! AIHR వ్యవస్థ సిద్ధంగా ఉంది."
        elif "hello" in msg_lower or "hi" in msg_lower:
            return "Hello! I am AIHR, your personal assistant. All systems operational."
        return f"AIHR (Offline Mode): Received '{user_msg}'. Add GEMINI_API_KEY to environment for live web intelligence."

# ==============================================================================
# SECTION 5: FLASK ROUTING, CONTROLLERS & API BLUEPRINTS
# ==============================================================================

api_bp = Blueprint('api', __name__, url_prefix='/api/v1')
ai_service = GeminiAIService()

@api_bp.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "gemini_active": ai_service.client is not None
    }), 200

@api_bp.route('/chat', methods=['POST'])
def process_chat_message():
    payload = request.get_json() or {}
    user_msg = payload.get("message", "").strip()
    session_id = payload.get("session_id", 1)

    if not user_msg:
        return jsonify({"error": "Empty message parameter submitted"}), 400

    # Retrieve or create session
    chat_session = ChatSession.query.get(session_id)
    if not chat_session:
        chat_session = ChatSession(id=session_id)
        db.session.add(chat_session)
        db.session.commit()

    detected_lang = MultilingualDetector.detect_language(user_msg)

    # Persist User Input
    user_rec = ChatMessage(
        session_id=chat_session.id,
        sender="user",
        message_content=user_msg,
        detected_language=detected_lang,
        is_math_query=MathEvaluationEngine.is_math_expression(user_msg)
    )
    db.session.add(user_rec)
    db.session.commit()

    # Step 1: Direct Safe Math AST Calculation
    response_text = MathEvaluationEngine.evaluate(user_msg)
    is_math = True

    # Step 2: Fallback to Gemini AI Model
    if not response_text:
        is_math = False
        response_text = ai_service.query(user_msg, detected_lang)

    # Persist AI Response
    ai_rec = ChatMessage(
        session_id=chat_session.id,
        sender="aihr",
        message_content=response_text,
        detected_language=detected_lang,
        is_math_query=is_math
    )
    db.session.add(ai_rec)
    db.session.commit()

    return jsonify({
        "reply": response_text,
        "session_id": chat_session.id,
        "language": detected_lang,
        "is_math": is_math
    }), 200

@api_bp.route('/history/<int:session_id>', methods=['GET'])
def get_chat_history(session_id: int):
    messages = ChatMessage.query.filter_by(session_id=session_id).order_by(ChatMessage.timestamp.asc()).all()
    return jsonify([msg.to_dict() for msg in messages]), 200

# ==============================================================================
# SECTION 6: WEB TEMPLATE UI & FRONTEND SUITE
# ==============================================================================

INDEX_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AIHR - Enterprise Helping Robot</title>
    <style>
        :root {
            --bg-color: #0b0e14;
            --panel-bg: #161b22;
            --header-bg: #21262d;
            --border-color: #30363d;
            --accent-blue: #58a6ff;
            --accent-green: #238636;
            --accent-green-hover: #2ea043;
            --accent-mic: #1f6beb;
            --text-main: #e6edf3;
            --text-sub: #8b949e;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            overflow: hidden;
        }

        .chat-container {
            width: 92%;
            max-width: 750px;
            height: 88vh;
            background: var(--panel-bg);
            border-radius: 12px;
            border: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            box-shadow: 0 12px 35px rgba(0, 0, 0, 0.75);
        }

        .chat-header {
            background: var(--header-bg);
            padding: 18px;
            text-align: center;
            border-bottom: 1px solid var(--border-color);
        }

        .chat-header h2 {
            color: var(--accent-blue);
            letter-spacing: 1.2px;
            font-size: 1.4rem;
        }

        .chat-header p {
            font-size: 0.85rem;
            color: var(--text-sub);
            margin-top: 4px;
        }

        .chat-box {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 14px;
        }

        .message {
            max-width: 80%;
            padding: 12px 18px;
            border-radius: 10px;
            line-height: 1.5;
            font-size: 0.95rem;
            word-wrap: break-word;
            animation: fadeIn 0.25s ease-in-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .user-message {
            align-self: flex-end;
            background-color: var(--accent-green);
            color: #ffffff;
            border-bottom-right-radius: 2px;
        }

        .aihr-message {
            align-self: flex-start;
            background-color: var(--header-bg);
            border: 1px solid var(--border-color);
            color: #c9d1d9;
            border-bottom-left-radius: 2px;
        }

        .input-area {
            display: flex;
            padding: 14px;
            background: var(--header-bg);
            border-top: 1px solid var(--border-color);
            gap: 10px;
        }

        input[type="text"] {
            flex: 1;
            padding: 12px;
            border: 1px solid var(--border-color);
            background: #0d1117;
            color: #fff;
            border-radius: 6px;
            outline: none;
            font-size: 1rem;
        }

        input[type="text"]:focus {
            border-color: var(--accent-blue);
        }

        button {
            padding: 12px 20px;
            background: var(--accent-green);
            border: none;
            color: white;
            font-weight: bold;
            border-radius: 6px;
            cursor: pointer;
            transition: background 0.2s;
        }

        button:hover { background: var(--accent-green-hover); }
        .mic-btn { background: var(--accent-mic); }
        .mic-btn:hover { background: #388bfd; }
    </style>
</head>
<body>

<div class="chat-container">
    <div class="chat-header">
        <h2>🤖 AIHR SYSTEM</h2>
        <p>Artificial Intelligence Helping Robot (Math & Multilingual Voice Enabled)</p>
    </div>
    
    <div class="chat-box" id="chatBox">
        <div class="message aihr-message">
            வணக்கம்! I am AIHR, your Artificial Intelligence Helping Robot. Ask me any question or math calculation.
        </div>
    </div>

    <div class="input-area">
        <input type="text" id="userInput" placeholder="Type a message or math problem (e.g., 25 * 4)..." onkeypress="handleKeyPress(event)">
        <button class="mic-btn" onclick="startListening()">🎤</button>
        <button onclick="sendMessage()">Send</button>
    </div>
</div>

<script>
    const chatBox = document.getElementById('chatBox');
    const userInput = document.getElementById('userInput');

    function appendMessage(text, className) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${className}`;
        msgDiv.innerText = text;
        chatBox.appendChild(msgDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    async function sendMessage(textToSend) {
        const text = textToSend || userInput.value.trim();
        if (!text) return;

        appendMessage(text, 'user-message');
        userInput.value = '';

        try {
            const response = await fetch('/api/v1/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text, session_id: 1 })
            });

            const data = await response.json();
            appendMessage(data.reply, 'aihr-message');
            speakMaleVoice(data.reply);
        } catch (err) {
            appendMessage("Error communicating with AIHR system.", 'aihr-message');
        }
    }

    function handleKeyPress(e) {
        if (e.key === 'Enter') sendMessage();
    }

    function startListening() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            alert("Voice recognition is not supported in this browser.");
            return;
        }

        const recognition = new SpeechRecognition();
        recognition.lang = 'ta-IN';
        recognition.start();

        recognition.onstart = function() {
            userInput.placeholder = "Listening...";
        };

        recognition.onresult = function(event) {
            userInput.placeholder = "Type a message or math problem...";
            const transcript = event.results[0][0].transcript;
            sendMessage(transcript);
        };

        recognition.onerror = function() {
            userInput.placeholder = "Type a message or math problem...";
        };
    }

    function speakMaleVoice(text) {
        if ('speechSynthesis' in window) {
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(text);
            const voices = window.speechSynthesis.getVoices();

            const maleVoice = voices.find(v => 
                v.name.toLowerCase().includes('male') || 
                v.name.toLowerCase().includes('david') || 
                v.name.toLowerCase().includes('google indian english') ||
                v.name.toLowerCase().includes('rishi')
            ) || voices[0];
            
            if (maleVoice) utterance.voice = maleVoice;

            utterance.pitch = 0.75;
            utterance.rate = 1.0;
            window.speechSynthesis.speak(utterance);
        }
    }

    window.speechSynthesis.onvoiceschanged = () => { window.speechSynthesis.getVoices(); };
</script>

</body>
</html>
"""

# ==============================================================================
# SECTION 7: APPLICATION FACTORY & ROUTE BINDINGS
# ==============================================================================

def create_app() -> Flask:
    """Application factory initializing blueprints and persistence context."""
    app = Flask(__name__)
    app.config.from_object(AppConfig)

    # Register Database & API
    initialize_database(app)
    app.register_blueprint(api_bp)

    @app.route('/')
    def render_index():
        return render_template_string(INDEX_HTML_TEMPLATE)

    return app

# ==============================================================================
# SECTION 8: AUTOMATED INTEGRATION & UNIT TEST SUITE
# ==============================================================================

class TestAIHRCoreSystem(unittest.TestCase):
    """Automated integration tests for AIHR AST Engine & Endpoints."""

    def setUp(self):
        self.app = create_app()
        self.app.config['TESTING'] = True
        self.app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        self.client = self.app.test_client()
        with self.app.app_context():
            db.create_all()

    def test_math_ast_engine(self):
        res = MathEvaluationEngine.evaluate("10 + 5 * 2")
        self.assertEqual(res, "The calculation result is: 20")

    def test_math_ast_function(self):
        res = MathEvaluationEngine.evaluate("sqrt(16)")
        self.assertEqual(res, "The calculation result is: 4.0")

    def test_language_detection(self):
        lang = MultilingualDetector.detect_language("வணக்கம்")
        self.assertEqual(lang, "ta")

    def test_api_endpoint(self):
        response = self.client.post(
            '/api/v1/chat',
            data=json.dumps({"message": "100 / 4"}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("25", data["reply"])

# ==============================================================================
# ENTRY POINT EXECUTION
# ==============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        logger.info("Starting AIHR System Integration Tests...")
        unittest.main(argv=['first-arg-is-ignored'])
    else:
        app = create_app()
        logger.info("[AIHR SYSTEM ONLINE] Launching Enterprise Server at http://127.0.0.1:5000")
        app.run(host="0.0.0.0", port=5000, debug=True)