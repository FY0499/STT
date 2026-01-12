import cv2
import mediapipe as mp
import numpy as np
import os
import google.genai as genai
from google.genai import types
from PIL import Image
from gtts import gTTS
import json
import re
import io
import tempfile
import time
from datetime import datetime

class ASLTranslatorEnhanced:
    def __init__(self, gemini_api_key=None):
        if gemini_api_key is None:
            try:
                with open(r"C:\Users\DELLI511\vc\STT\API.txt", 'r') as f:
                    gemini_api_key = f.read().strip()
            except:
                raise Exception("API key file not found")
        
        self.client = genai.Client(api_key=gemini_api_key)
        self.model_name = 'gemini-2.0-flash-exp'
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.4
        )
        
        self.mp_draw = mp.solutions.drawing_utils
        
        from deep_translator import GoogleTranslator
        self.translator = GoogleTranslator(source='en', target='ar')
        
        self.uploaded_files = []
    
    def cleanup_all_cache(self):
        try:
            for file_name in self.uploaded_files:
                try:
                    self.client.files.delete(name=file_name)
                except:
                    pass
            
            self.uploaded_files.clear()
            
            try:
                all_files = self.client.files.list()
                for file in all_files:
                    try:
                        self.client.files.delete(name=file.name)
                    except:
                        pass
            except:
                pass
        except:
            pass
    
    def create_expert_prompt(self):
        return """You are an expert ASL (American Sign Language) interpreter with deep understanding of linguistic and emotional components.

🎯 CRITICAL TASK:
Analyze this COMPLETE video and identify ALL signs performed in sequence, along with facial expressions and emotions.

⚠️ IMPORTANT CONTEXT:
The signer is a LEARNER practicing ASL signs. They may perform signs at DIFFERENT SPEEDS:
- 🚀 FAST/PROFESSIONAL: Expert-level speed, quick transitions, fluid movements
- 🐢 SLOW/EDUCATIONAL: Deliberate slow movements for learning/teaching
- ⚖️ MEDIUM/NORMAL: Standard comfortable signing speed

**CRITICAL**: You MUST recognize signs at ALL SPEEDS. A sign is still valid whether performed in:
- 0.3 seconds (very fast)
- 1.5 seconds (medium)
- 3+ seconds (very slow/educational)

The SAME sign can be performed at different speeds in the same video!

🎬 SPEED ADAPTATION RULES:

1️⃣ FAST SIGNS (< 0.8 seconds):
   ✅ Accept: Quick, professional movements
   ✅ Look for: Rapid hand shape changes, minimal holds
   ✅ Common in: Greetings (HELLO), simple verbs (YES, NO)
   ⚠️ May blur slightly in video - this is NORMAL
   
2️⃣ MEDIUM SIGNS (0.8 - 2 seconds):
   ✅ Accept: Standard signing speed
   ✅ Look for: Clear movements with brief holds
   ✅ Common in: Most everyday signs
   
3️⃣ SLOW SIGNS (> 2 seconds):
   ✅ Accept: Educational, learning pace
   ✅ Look for: Exaggerated holds, careful positioning
   ✅ Common in: Complex signs, learning practice
   ⚠️ May have long pauses between positions - this is INTENTIONAL

🔍 MULTI-SPEED DETECTION STRATEGY:

**Step 1: IDENTIFY ALL MOVEMENT SEGMENTS**
- Scan the ENTIRE video timeline
- Mark ANY hand movement as potential sign
- Don't ignore movements just because they're fast or slow

**Step 2: CLASSIFY EACH SEGMENT BY SPEED**
For each detected movement:
- FAST: Hand moves and stops within 1 second
- MEDIUM: Movement takes 1-2 seconds
- SLOW: Movement takes 2+ seconds with clear holds

**Step 3: MATCH TO KNOWN SIGNS**
- Compare movement pattern (not duration) to ASL signs
- Same pattern at different speeds = same sign
- Example: HELLO can be:
  * Fast: Quick wave (0.5s)
  * Medium: Normal wave (1.5s)
  * Slow: Deliberate wave (3s)

📋 COMPREHENSIVE ANALYSIS FRAMEWORK:

1️⃣ TEMPORAL SEQUENCE ANALYSIS:
   - Watch the ENTIRE video from START to END
   - Identify ALL signs in chronological order
   - **CRITICAL**: Don't skip fast movements OR slow movements
   - Track timing: Sign 1 (0-0.5s) → Sign 2 (0.5-1.2s) → Sign 3 (1.2-4s)

2️⃣ NON-MANUAL MARKERS (NMMs):
   
   👁️ EYEBROW PATTERNS:
   - Raised eyebrows = YES/NO questions, surprise, emphasis
   - Furrowed eyebrows = WH-questions
   - Neutral = Statement
   
   👄 MOUTH PATTERNS:
   - Open mouth = Emphasis, excitement
   - Tight lips = Concentration, negation
   
   🎭 HEAD MOVEMENTS:
   - Head tilt forward = Question
   - Head nod = Affirmation
   - Head shake = Negation
   
   😊 OVERALL FACIAL EXPRESSION:
   - Happy/Smiling = Positive emotion, greeting
   - Serious/Neutral = Statement
   - Concerned/Worried = Doubt, question

3️⃣ COMMON ASL SIGNS (with ALL speed variations):

   **Greetings & Basics:**
   - HELLO: Wave hand (0.3-3s all valid)
   - GOODBYE: Wave palm out (0.4-4s)
   - THANK YOU: Hand chin→forward (0.5-3s)
   - YES: Fist nods (0.3-2.5s)
   - NO: Finger snap (0.2-2s)

   **Personal Pronouns:**
   - I/ME: Point to chest (0.3-2s)
   - YOU: Point forward (0.3-2s)

   **Common Verbs:**
   - LEARN: Forehead→palm (0.6-5s)
   - SPEAK/SAY: Circles near mouth (0.4-3s)
   - UNDERSTAND: Flick at forehead (0.3-2s)
   - KNOW: Tap temple (0.3-2s)

   **Topics:**
   - SIGN (ASL): Alternating circles (0.8-5s)
   - LANGUAGE: L-shapes from chin (0.6-4s)

4️⃣ MIXED-SPEED PHRASE EXAMPLES:

   "I learn sign language" (can be 4-15 seconds total):
   - I (fast 0.5s) → LEARN (slow 3s) → SIGN (medium 2s) → LANGUAGE (slow 3s)
   - ✅ VALID: Speed varies naturally

5️⃣ SPEED-FLEXIBLE RULES:
   ✅ Accept variations in:
   - Timing (fast/medium/slow)
   - Movement speed
   - Pause duration
   
   ❌ But still require:
   - Recognizable hand shape
   - Clear movement direction
   - Logical sequence

⚠️ COMMON MISTAKES TO AVOID:
   ❌ Ignoring fast movements as "too quick"
   ❌ Dismissing slow movements as "incomplete"
   ❌ Expecting uniform speed
   ❌ Only reporting the last sign

📤 OUTPUT FORMAT (JSON ONLY):
{
  "signs_detected": [
    {
      "sign": "SIGN_NAME",
      "time": "0-0.5s",
      "speed": "fast/medium/slow",
      "duration": "0.5s",
      "confidence": "high/medium/low",
      "notes": "speed observations"
    }
  ],
  "speed_analysis": {
    "overall_pace": "mixed/fast/medium/slow",
    "speed_variation": "high/medium/low",
    "fastest_sign": "SIGN (0.3s)",
    "slowest_sign": "SIGN (3.5s)",
    "speed_consistency": "description"
  },
  "facial_analysis": {
    "eyebrows": "raised/furrowed/neutral",
    "mouth": "open/neutral/tight",
    "head_movement": "nod/shake/tilt/none",
    "overall_expression": "happy/neutral/serious/questioning"
  },
  "sentence_type": "statement/question/exclamation/greeting",
  "emotion": "happy/neutral/serious/excited/concerned",
  "emotion_intensity": "low/medium/high",
  "english_translation": "Complete sentence",
  "confidence": "high/medium/low",
  "can_identify": true/false,
  "needs_repeat": false/true,
  "learner_variations_noted": "including speed variations",
  "interpretation_notes": "with speed observations"
}

REMEMBER: Speed variation is NORMAL in learning! Recognize patterns, not timing.

NOW ANALYZE WITH FULL SPEED FLEXIBILITY:"""
    
    def analyze_speed_profile(self, frames):
        if len(frames) < 3:
            return "insufficient_frames"
        
        try:
            motion_scores = []
            
            for i in range(1, min(len(frames), 30)):
                gray1 = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(gray1, gray2)
                motion_score = np.mean(diff)
                motion_scores.append(motion_score)
            
            if not motion_scores:
                return "unknown"
            
            avg_motion = np.mean(motion_scores)
            max_motion = np.max(motion_scores)
            
            if max_motion > 30 or avg_motion > 15:
                speed_profile = "contains_fast_movements"
            elif avg_motion < 5:
                speed_profile = "contains_slow_movements"
            else:
                speed_profile = "mixed_speed"
            
            return speed_profile
        except:
            return "unknown"
    
    def decode_frames_from_flutter(self, frames_data):
        decoded_frames = []
        for frame_bytes in frames_data:
            try:
                nparr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is not None:
                    decoded_frames.append(frame)
            except:
                continue
        return decoded_frames
    
    def draw_landmarks(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        annotated = frame.copy()
        
        hand_results = self.hands.process(rgb_frame)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    annotated,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)
                )
        
        face_results = self.face_mesh.process(rgb_frame)
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                h, w, _ = annotated.shape
                key_points = [33, 133, 362, 263, 70, 300, 61, 291]
                for idx in key_points:
                    lm = face_landmarks.landmark[idx]
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(annotated, (x, y), 2, (255, 0, 0), -1)
        
        return annotated
    
    def create_video_from_frames(self, frames, fps=15):
        if not frames or len(frames) == 0:
            return None
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_filename = temp_file.name
        temp_file.close()
        
        try:
            height, width, _ = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_filename, fourcc, fps, (width, height))
            
            for frame in frames:
                out.write(frame)
            
            out.release()
            return temp_filename
        except:
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
            return None
    
    def analyze_with_gemini(self, frames):
        speed_profile = self.analyze_speed_profile(frames)
        
        frames_with_landmarks = []
        video_path = None
        
        try:
            for frame in frames:
                annotated = self.draw_landmarks(frame)
                if annotated is not None:
                    frames_with_landmarks.append(annotated)
            
            if not frames_with_landmarks:
                return None
            
            video_path = self.create_video_from_frames(frames_with_landmarks, fps=15)
            
            if video_path is None:
                return None
            
            prompt = self.create_expert_prompt()
            if speed_profile != "unknown":
                speed_note = f"\n\n⚡ SPEED PROFILE: This video {speed_profile}. Be especially attentive to varying speeds."
                prompt += speed_note
            
            with open(video_path, 'rb') as f:
                uploaded_file = self.client.files.upload(
                    file=f,
                    config={'mime_type': 'video/mp4'}
                )
            
            self.uploaded_files.append(uploaded_file.name)
            
            max_wait = 30
            wait_time = 0
            while uploaded_file.state.name == "PROCESSING" and wait_time < max_wait:
                time.sleep(2)
                wait_time += 2
                uploaded_file = self.client.files.get(name=uploaded_file.name)
            
            if uploaded_file.state.name == "FAILED":
                raise ValueError("Video processing failed")
            
            if wait_time >= max_wait:
                raise ValueError("Video processing timeout")
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt, uploaded_file],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    top_p=0.9,
                    max_output_tokens=2000
                )
            )
            
            return self.parse_response(response.text.strip())
        except:
            return None
        finally:
            if video_path and os.path.exists(video_path):
                try:
                    os.unlink(video_path)
                except:
                    pass
    
    def parse_response(self, text):
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0]
        elif '```' in text:
            text = text.split('```')[1].split('```')[0]
        
        try:
            result = json.loads(text.strip())
            
            if result.get('needs_repeat', False) or not result.get('can_identify', True):
                result['english_translation'] = 'Cannot identify - please repeat the sign more clearly'
            
            return result
        except:
            match = re.search(r'"english_translation"\s*:\s*"([^"]+)"', text)
            if match:
                translation = match.group(1)
                if 'cannot' in translation.lower() or 'unclear' in translation.lower():
                    return {
                        'english_translation': 'Cannot identify - please repeat the sign more clearly',
                        'confidence': 'low',
                        'needs_repeat': True,
                        'emotion': 'neutral'
                    }
                return {
                    'english_translation': translation,
                    'confidence': 'medium',
                    'emotion': 'neutral'
                }
            
            return {
                'english_translation': text.strip(),
                'confidence': 'low',
                'emotion': 'neutral'
            }
    
    def translate_to_arabic(self, english_text, emotion='neutral'):
        if 'cannot identify' in english_text.lower() or 'please repeat' in english_text.lower():
            return 'لا يمكن التعرف على الإشارة - يرجى إعادة الإشارة بوضوح أكثر'
        
        try:
            arabic = self.translator.translate(english_text)
            
            if emotion in ['excited', 'enthusiastic', 'happy']:
                if not arabic.endswith('!'):
                    arabic = arabic + '!'
            elif emotion in ['concerned', 'doubtful']:
                if not arabic.endswith('؟'):
                    arabic = arabic + '؟'
            
            return arabic
        except:
            return english_text
    
    def generate_emotional_audio_bytes(self, text, emotion='neutral', intensity='medium'):
        try:
            speed_map = {
                'excited': True,
                'happy': True,
                'enthusiastic': True,
                'neutral': False,
                'serious': False,
                'concerned': False,
                'doubtful': False
            }
            
            slow = not speed_map.get(emotion, False)
            
            tts = gTTS(text=text, lang='ar', slow=slow)
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            
            return audio_buffer.read()
        except:
            return None
    
    def generate_text_file(self, text, emotion='neutral'):
        try:
            
            full_text = text
            
            text_buffer = io.BytesIO()
            text_buffer.write(full_text.encode('utf-8'))
            text_buffer.seek(0)
            return text_buffer.read()
        except:
            return None
    
    # def generate_detailed_analysis_report(self, result, arabic_text, english_text):
    #     try:
            
    #         report = ""
            
    #         if 'facial_analysis' in result:
    #             facial = result['facial_analysis']
    #             report += f"   الحواجب: {facial.get('eyebrows', 'N/A')}\n"
    #             report += f"   الفم: {facial.get('mouth', 'N/A')}\n"
    #             report += f"   حركة الرأس: {facial.get('head_movement', 'N/A')}\n"
    #             report += f"   التعبير: {facial.get('overall_expression', 'N/A')}\n\n"
            
    #         return report.encode('utf-8')
    #     except:
    #         return None
    
    def process_frames_from_flutter(self, frames_data, save_report_locally=True, report_path='./asl_reports'):
        if not frames_data or len(frames_data) == 0:
            return {
                'success': False,
                'error': 'No frames provided',
                'audio_file': None,
                'text_file': None,
                'has_audio': False,
                'has_text': False,
                'needs_repeat': False,
                'emotion': 'neutral'
            }
        
        try:
            frames = self.decode_frames_from_flutter(frames_data)
            
            if not frames:
                return {
                    'success': False,
                    'error': 'Failed to decode frames',
                    'audio_file': None,
                    'text_file': None,
                    'has_audio': False,
                    'has_text': False,
                    'needs_repeat': False,
                    'emotion': 'neutral'
                }
            
            result = self.analyze_with_gemini(frames)
            
            if not result:
                self.cleanup_all_cache()
                return {
                    'success': False,
                    'error': 'Analysis failed',
                    'audio_file': None,
                    'text_file': None,
                    'has_audio': False,
                    'has_text': False,
                    'needs_repeat': True,
                    'emotion': 'neutral'
                }
            
            needs_repeat = result.get('needs_repeat', False) or not result.get('can_identify', True)
            english = result.get('english_translation', 'N/A')
            emotion = result.get('emotion', 'neutral')
            emotion_intensity = result.get('emotion_intensity', 'medium')
            confidence = result.get('confidence', 'unknown')
            
            arabic = self.translate_to_arabic(english, emotion)
            audio_bytes = self.generate_emotional_audio_bytes(arabic, emotion, emotion_intensity)
            text_bytes = self.generate_text_file(arabic, emotion)
            # detailed_report = self.generate_detailed_analysis_report(result, arabic, english)
            
            # report_saved = False
            # if save_report_locally and detailed_report:
            #     try:
            #         os.makedirs(report_path, exist_ok=True)
            #         timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            #         report_filename = os.path.join(report_path, f'asl_analysis_{timestamp}.txt')
                    
            #         with open(report_filename, 'wb') as f:
            #             f.write(detailed_report)
                    
            #         report_saved = True
            #     except:
            #         pass
            
            if not needs_repeat:
                self.cleanup_all_cache()
            
            return {
                'success': True,
                'audio_file': audio_bytes,
                'text_file': text_bytes,
                'has_audio': audio_bytes is not None,
                'has_text': text_bytes is not None,
                # 'report_saved': report_saved,
                'needs_repeat': needs_repeat,
                'english_text': english,
                'arabic_text': arabic,
                'confidence': confidence,
                'emotion': emotion,
                'emotion_intensity': emotion_intensity,
                'sentence_type': result.get('sentence_type', 'statement'),
                'facial_analysis': result.get('facial_analysis', {}),
                'speed_analysis': result.get('speed_analysis', {}),
                'learner_variations': result.get('learner_variations_noted', ''),
                'interpretation_notes': result.get('interpretation_notes', ''),
                'signs_detected': result.get('signs_detected', [])
            }
        except Exception as e:
            self.cleanup_all_cache()
            
            return {
                'success': False,
                'error': str(e),
                'audio_file': None,
                'text_file': None,
                'has_audio': False,
                'has_text': False,
                'needs_repeat': True,
                'emotion': 'neutral'
            }
    
    def __del__(self):
        try:
            self.cleanup_all_cache()
        except:
            pass
