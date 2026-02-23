import requests
import pyttsx3
import traceback
import re
import cv2
import base64
import os
import time
import numpy as np
import subprocess
import json
import platform
import psutil

# =====================
# CONFIG
# =====================

API_KEY = "put_your_openrouter_api_key_here"
MODEL = "stepfun/step-3.5-flash:free"
URL = "https://openrouter.ai/api/v1/chat/completions"

DEBUG = True

# =====================
# HARDWARE DETECTION
# =====================

def detect_gpu():
    """Detect available GPUs and return the best one"""
    try:
        # Check for NVIDIA GPU
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("✅ NVIDIA GPU detected - using CUDA acceleration")
                return "cuda"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # Check for AMD GPU on Windows
        if platform.system() == "Windows":
            try:
                result = subprocess.run(["wmic", "path", "win32_VideoController", "get", "name"], capture_output=True, text=True, timeout=5)
                if "AMD" in result.stdout or "Radeon" in result.stdout:
                    print("✅ AMD GPU detected - using ROCm")
                    return "rocm"
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
        
        # Check for Intel GPU
        if platform.system() == "Windows":
            try:
                result = subprocess.run(["wmic", "path", "win32_VideoController", "get", "name"], capture_output=True, text=True, timeout=5)
                if "Intel" in result.stdout and ("HD Graphics" in result.stdout or "Iris" in result.stdout or "UHD" in result.stdout):
                    print("✅ Intel GPU detected - using OpenCL")
                    return "opencl"
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
        
        print("⚠️ No GPU detected - falling back to CPU")
        return "cpu"
        
    except Exception as e:
        print(f"❌ GPU detection failed: {e}")
        return "cpu"

def detect_camera():
    """Detect available cameras and return the best one"""
    try:
        available_cameras = []
        
        # Test camera indices 0-9
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Get camera info
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                available_cameras.append({
                    'index': i,
                    'resolution': f"{width}x{height}",
                    'fps': fps
                })
                cap.release()
        
        if available_cameras:
            # Choose the camera with highest resolution
            best_camera = max(available_cameras, key=lambda x: int(x['resolution'].split('x')[0]) * int(x['resolution'].split('x')[1]))
            print(f"✅ Camera {best_camera['index']} detected - {best_camera['resolution']} @ {best_camera['fps']}fps")
            return best_camera['index']
        else:
            print("❌ No camera detected")
            return None
            
    except Exception as e:
        print(f"❌ Camera detection failed: {e}")
        return None

# Auto-detect hardware
GPU_TYPE = detect_gpu()
CAMERA_INDEX = detect_camera()

# =====================
# DEBUG
# =====================

def debug(msg):
    if DEBUG:
        print("[DEBUG]", msg)

# =====================
# REMOVE EMOJIS
# =====================

def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols
        "\U0001F680-\U0001F6FF"  # transport
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FAFF"
        "\U00002700-\U000027BF"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub("", text)

# =====================
# VOICE SETUP
# =====================

engine = pyttsx3.init()
engine.setProperty("rate", 170)
engine.setProperty("volume", 1.0)

voices = engine.getProperty("voices")
if len(voices) > 1:
    engine.setProperty("voice", voices[1].id)

def speak(text):
    try:
        clean_text = remove_emojis(text)
        debug("Speaking cleaned text...")
        engine.say(clean_text)
        engine.runAndWait()
    except Exception:
        debug("Speech failed")
        traceback.print_exc()

# =====================
# ASK AI
# =====================

def ask_ai(message):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Voice CMD AI"
    }

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful AI assistant called echo. Always reply briefly and clearly using short sentences. Avoid long explanations."
            },
            {
                "role": "user",
                "content": message
            }
        ]
    }

    try:
        debug("Sending request...")
        r = requests.post(URL, headers=headers, json=payload, timeout=60)

        debug(f"Status code: {r.status_code}")

        if r.status_code != 200:
            print(r.text)
            return "API error."

        data = r.json()
        reply = data["choices"][0]["message"]["content"]

        return reply

    except Exception:
        debug("API crashed")
        traceback.print_exc()
        return "Connection failed."

# =====================
# OLLAMA VISION SETUP
# =====================

def analyze_with_ollama_vision(image_path):
    """Analyze image using Ollama LLaVA model with auto-detected GPU"""
    try:
        if GPU_TYPE == "cpu":
            print("⚠️ Using CPU for LLaVA analysis (no GPU detected)")
        else:
            print(f"🔍 Running LLaVA analysis on {GPU_TYPE.upper()}...")
        
        # Read image and encode to base64
        with open(image_path, "rb") as img:
            image_data = base64.b64encode(img.read()).decode("utf-8")
        
        # Use Ollama API with LLaVA
        ollama_path = r"C:\Users\iceai\AppData\Local\Programs\Ollama\ollama.exe"
        
        # Create a temporary prompt file
        prompt = "What do you see in this image? Be very brief - 1-2 short sentences only. If there's text, identify the language first, then read the text exactly as written."
        
        try:
            # Run Ollama with LLaVA model
            result = subprocess.run([
                ollama_path, "run", "llava",
                prompt,
                image_path
            ], capture_output=True, text=True, timeout=60, encoding='utf-8', errors='ignore')
            
            if result.returncode == 0:
                response = result.stdout.strip()
                if GPU_TYPE == "cpu":
                    print(f"✅ LLaVA CPU result: {response}")
                else:
                    print(f"✅ LLaVA {GPU_TYPE.upper()} result: {response}")
                return response
            else:
                if GPU_TYPE == "cpu":
                    print(f"LLaVA CPU error: {result.stderr}")
                    return "LLaVA CPU analysis failed."
                else:
                    print(f"LLaVA {GPU_TYPE.upper()} error: {result.stderr}")
                    return f"LLaVA {GPU_TYPE.upper()} analysis failed."
                
        except subprocess.TimeoutExpired:
            if GPU_TYPE == "cpu":
                return "LLaVA CPU analysis timed out."
            else:
                return f"LLaVA {GPU_TYPE.upper()} analysis timed out."
        except Exception as e:
            debug(f"LLaVA {GPU_TYPE} error: {e}")
            return f"LLaVA {GPU_TYPE} analysis failed: {str(e)}"
            
    except Exception as e:
        debug(f"LLaVA {GPU_TYPE} vision error: {e}")
        return f"LLaVA {GPU_TYPE} vision failed: {str(e)}"

def analyze_with_ollama_ocr(image_path):
    """Analyze image using Ollama LLaVA for OCR with auto-detected GPU"""
    try:
        if GPU_TYPE == "cpu":
            print("⚠️ Using CPU for LLaVA OCR (no GPU detected)")
        else:
            print(f"🔍 Running LLaVA OCR on {GPU_TYPE.upper()}...")
        
        # Read image
        with open(image_path, "rb") as img:
            image_data = base64.b64encode(img.read()).decode("utf-8")
        
        # Use Ollama API with LLaVA
        ollama_path = r"C:\Users\iceai\AppData\Local\Programs\Ollama\ollama.exe"
        
        # Create OCR prompt
        prompt = "Read all text in this image. First identify the language, then provide the exact text as written. Be brief."
        
        try:
            # Run Ollama with LLaVA model
            result = subprocess.run([
                ollama_path, "run", "llava",
                prompt,
                image_path
            ], capture_output=True, text=True, timeout=60, encoding='utf-8', errors='ignore')
            
            if result.returncode == 0:
                response = result.stdout.strip()
                if GPU_TYPE == "cpu":
                    print(f"✅ LLaVA CPU OCR result: {response}")
                else:
                    print(f"✅ LLaVA {GPU_TYPE.upper()} OCR result: {response}")
                return response
            else:
                if GPU_TYPE == "cpu":
                    print(f"LLaVA CPU OCR error: {result.stderr}")
                    return "LLaVA CPU OCR failed."
                else:
                    print(f"LLaVA {GPU_TYPE.upper()} OCR error: {result.stderr}")
                    return f"LLaVA {GPU_TYPE.upper()} OCR failed."
                
        except subprocess.TimeoutExpired:
            if GPU_TYPE == "cpu":
                return "LLaVA CPU OCR timed out."
            else:
                return f"LLaVA {GPU_TYPE.upper()} OCR timed out."
        except Exception as e:
            debug(f"LLaVA {GPU_TYPE} OCR error: {e}")
            return f"LLaVA {GPU_TYPE} OCR failed: {str(e)}"
            
    except Exception as e:
        debug(f"LLaVA {GPU_TYPE} OCR error: {e}")
        return f"LLaVA {GPU_TYPE} OCR failed: {str(e)}"

def analyze_with_openai_ocr(image_path):
    """Analyze image using OpenAI OCR for text recognition"""
    try:
        print("🔍 Running OCR analysis...")
        
        # Read and encode image
        with open(image_path, "rb") as img:
            b64 = base64.b64encode(img.read()).decode("utf-8")

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "OCR AI"
        }

        payload = {
            "model": "qwen/qwen-2-vl-7b-instruct:free",  # Free vision model
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Read all text in this image. First identify the language, then provide the exact text as written. Be brief."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}"
                            }
                        }
                    ]
                }
            ]
        }

        try:
            print("🧠 Performing OCR...")
            r = requests.post(URL, headers=headers, json=payload, timeout=60)

            if r.status_code != 200:
                print(f"OCR API Error: {r.status_code}")
                return "OCR failed."

            data = r.json()
            response = data["choices"][0]["message"]["content"]
            
            print(f"✅ OCR result: {response}")
            return response

        except Exception as e:
            debug(f"OCR API error: {e}")
            return "OCR failed."
            
    except Exception as e:
        debug(f"OpenAI OCR error: {e}")
        return f"OCR failed: {str(e)}"

def capture_and_analyze_ocr():
    try:
        if CAMERA_INDEX is None:
            print("❌ No camera available")
            return "Camera not detected."
        
        debug(f"Initializing camera {CAMERA_INDEX} for OCR...")
        cap = cv2.VideoCapture(CAMERA_INDEX)
        
        if not cap.isOpened():
            print(f"❌ Camera {CAMERA_INDEX} not accessible")
            return "Camera error."
        
        ret, frame = cap.read()
        if not ret:
            print(f"❌ Failed to capture image from camera {CAMERA_INDEX}")
            cap.release()
            return "Capture failed."
        
        print("✅ Image captured! Performing OCR...")
        
        # Save image temporarily
        temp_path = "temp_capture.jpg"
        cv2.imwrite(temp_path, frame)
        
        # Use LLaVA OCR analysis
        ocr_result = analyze_with_ollama_ocr(temp_path)
        
        # Clean up
        os.remove(temp_path)
        
        cap.release()
        
        return ocr_result
            
    except Exception as e:
        debug(f"OCR Camera error: {e}")
        return "OCR camera initialization failed."

def capture_and_analyze_image():
    try:
        if CAMERA_INDEX is None:
            print("❌ No camera available")
            return "Camera not detected."
        
        debug(f"Initializing camera {CAMERA_INDEX}...")
        cap = cv2.VideoCapture(CAMERA_INDEX)
        
        if not cap.isOpened():
            print(f"❌ Camera {CAMERA_INDEX} not accessible")
            return "Camera error."
        
        ret, frame = cap.read()
        if not ret:
            print(f"❌ Failed to capture image from camera {CAMERA_INDEX}")
            cap.release()
            return "Capture failed."
        
        if GPU_TYPE == "cpu":
            print("✅ Image captured! Analyzing with LLaVA CPU...")
        else:
            print(f"✅ Image captured! Analyzing with LLaVA {GPU_TYPE.upper()}...")
        
        # Save image temporarily
        temp_path = "temp_capture.jpg"
        cv2.imwrite(temp_path, frame)
        
        # Use LLaVA vision analysis
        vision_result = analyze_with_ollama_vision(temp_path)
        
        # Clean up
        os.remove(temp_path)
        
        cap.release()
        
        return vision_result
            
    except Exception as e:
        debug(f"Camera error: {e}")
        return "Camera initialization failed."

# =====================
# MAIN LOOP
# =====================

print("=== ECHO AI READY ===")
print(f"Hardware: {GPU_TYPE.upper()} | Camera: {CAMERA_INDEX if CAMERA_INDEX is not None else 'None'}")
print("Type something (exit to quit)")
print("Type 'what is this' to capture and analyze image")
print("Type 'read this', 'could you read this', 'read this please', or 'please read this' for OCR")
print()

while True:
    try:
        user = input("You: ")

        if user.lower() == "exit":
            break

        if not user.strip():
            continue

        # Check for camera commands
        if "what is this" in user.lower():
            print("📷 Camera mode activated...")
            reply = capture_and_analyze_image()
        elif any(cmd in user.lower() for cmd in ["could you read this", "read this please", "read this", "please read this", "read"]):
            print("📷 OCR mode activated...")
            reply = capture_and_analyze_ocr()
        else:
            print("Thinking...")
            reply = ask_ai(user)

        print("AI:", reply, "\n")

        # Speak WITHOUT emojis
        speak(reply)

    except KeyboardInterrupt:
        break
    except Exception:
        debug("MAIN ERROR")
        traceback.print_exc()
