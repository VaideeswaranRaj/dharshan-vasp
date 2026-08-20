import os
import re
import math
from flask import Flask, request, jsonify, render_template_string
from google import genai

app = Flask(__name__)

# Initialize Gemini AI Client (Set your GEMINI_API_KEY environment variable)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

SYSTEM_PROMPT = """
You are AIHR (Artificial Intelligence Helping Robot), a smart assistant like Jarvis.
You can perform mathematical calculations and communicate in Tamil, Telugu, Kannada, Malayalam, and English.
Always reply in the same language the user uses.
Keep your answers brief, intelligent, direct, and accurate.
"""

def solve_math_expression(user_msg):
    """Safely evaluates basic mathematical expressions."""
    # Pattern to detect purely mathematical expressions (numbers and arithmetic operators)
    cleaned = re.sub(r'[^\d\+\-\*\/\(\)\.\^]', '', user_msg)
    if len(cleaned) > 1 and any(op in user_msg for op in ['+', '-', '*', '/', '^']):
        try:
            expression = cleaned.replace('^', '**')
            # Safe evaluation supporting standard math operations
            allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
            result = eval(expression, {"__builtins__": None}, allowed_names)
            return f"The calculation result is: {result}"
        except Exception:
            return None
    return None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AIHR - Artificial Intelligence Helping Robot</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Roboto, sans-serif;
        }
        body {
            background-color: #0b0e14;
            color: #e6edf3;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }
        .chat-container {
            width: 90%;
            max-width: 650px;
            height: 85vh;
            background: #161b22;
            border-radius: 12px;
            border: 1px solid #30363d;
            display: flex;
            flex-direction: column;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.7);
        }
        .chat-header {
            background: #21262d;
            padding: 16px;
            text-align: center;
            border-bottom: 1px solid #30363d;
        }
        .chat-header h2 {
            color: #58a6ff;
            letter-spacing: 1px;
        }
        .chat-header p {
            font-size: 0.85rem;
            color: #8b949e;
        }
        .chat-box {
            flex: 1;
            padding: 16px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        .message {
            max-width: 80%;
            padding: 12px 16px;
            border-radius: 10px;
            line-height: 1.5;
            font-size: 0.95rem;
            word-wrap: break-word;
        }
        .user-message {
            align-self: flex-end;
            background-color: #238636;
            color: #ffffff;
            border-bottom-right-radius: 2px;
        }
        .aihr-message {
            align-self: flex-start;
            background-color: #21262d;
            border: 1px solid #30363d;
            color: #c9d1d9;
            border-bottom-left-radius: 2px;
        }
        .input-area {
            display: flex;
            padding: 12px;
            background: #21262d;
            border-top: 1px solid #30363d;
            gap: 8px;
        }
        input[type="text"] {
            flex: 1;
            padding: 12px;
            border: 1px solid #30363d;
            background: #0d1117;
            color: #fff;
            border-radius: 6px;
            outline: none;
            font-size: 1rem;
        }
        input[type="text"]:focus {
            border-color: #58a6ff;
        }
        button {
            padding: 12px 18px;
            background: #238636;
            border: none;
            color: white;
            font-weight: bold;
            border-radius: 6px;
            cursor: pointer;
            transition: 0.2s;
        }
        button:hover { background: #2ea043; }
        .mic-btn { background: #1f6beb; }
        .mic-btn:hover { background: #388bfd; }
    </style>
</head>
<body>

<div class="chat-container">
    <div class="chat-header">
        <h2>🤖 AIHR</h2>
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
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text })
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

    // Voice Input - Web Speech API
    function startListening() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            alert("Voice recognition is not supported in this browser.");
            return;
        }

        const recognition = new SpeechRecognition();
        recognition.lang = 'ta-IN'; // Multilingual speech recognition
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

    // Male Voice Synthesizer - Text to Speech
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

            utterance.pitch = 0.75; // Low pitch for Jarvis male tone
            utterance.rate = 1.0;
            window.speechSynthesis.speak(utterance);
        }
    }

    window.speechSynthesis.onvoiceschanged = () => { window.speechSynthesis.getVoices(); };
</script>

</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/chat', methods=['POST'])
def chat():
    user_msg = request.json.get("message", "").strip()
    
    # Step 1: Try direct mathematical computation
    math_result = solve_math_expression(user_msg)
    if math_result:
        return jsonify({"reply": math_result})

    # Step 2: Fallback to Gemini AI model for general or complex math queries
    if client:
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=f"{SYSTEM_PROMPT}\nUser input: {user_msg}"
            )
            reply = response.text.strip()
        except Exception:
            reply = "AIHR System error processing request."
    else:
        # Fallback offline rule responses
        if "வணக்கம்" in user_msg or "vanakkam" in user_msg.lower():
            reply = "வணக்கம்! நான் AIHR. கணக்கு மற்றும் உரையாடல்களுக்கு நான் தயார்."
        else:
            reply = f"AIHR received: '{user_msg}'. All systems functional."

    return jsonify({"reply": reply})

if __name__ == "__main__":
    print("\n[AIHR SYSTEM ONLINE] Launching server at http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)