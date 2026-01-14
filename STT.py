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
        self.model_name = 'gemini-2.5-pro'
        
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
                except Exception as e:
                    pass
            
            self.uploaded_files.clear()
        except Exception as e:
            pass
    
    def create_expert_prompt(self):
        return """You are an expert ASL (American Sign Language) interpreter with deep understanding of sign language linguistics.

TASK: Analyze this complete video and identify ALL signs performed in sequence.

CONTEXT:
- The signer may be a LEARNER, so signs might be at different speeds (fast/medium/slow)
- ALL speeds are valid - the same sign can be performed quickly or slowly
- Focus on MOVEMENT PATTERNS, not duration
- Watch the ENTIRE video from start to finish

ANALYSIS GUIDELINES:

1. IDENTIFY ALL SIGNS:
   - Common greetings: HELLO (wave), THANK YOU (chin→forward), GOODBYE
   - Pronouns: I/ME (point to chest), YOU (point forward)
   - Verbs: LEARN (forehead→palm), UNDERSTAND (flick forehead), KNOW (tap temple)
   - Topics: SIGN (alternating circles), LANGUAGE (L-shapes from chin)
   - Questions: Raised eyebrows indicate yes/no questions
   
2. ANALYZE FACIAL EXPRESSIONS:
   - Eyebrows: raised (questions), furrowed (WH-questions), neutral (statements)
   - Mouth: open (emphasis), tight (concentration), smiling (positive)
   - Head: nod (affirmation), shake (negation), tilt (questioning)
   
3. EMOTIONAL CONTEXT:
   - Overall expression: happy, neutral, serious, excited, concerned
   - Intensity: low, medium, high

OUTPUT FORMAT (JSON ONLY):

{
  "signs_detected": [
    {
      "sign": "HELLO",
      "confidence": "high",
      "time_range": "0-1s",
      "notes": "Clear wave gesture"
    }
  ],
  "facial_analysis": {
    "expression": "happy",
    "eyebrows": "neutral",
    "mouth": "smiling",
    "head_movement": "none"
  },
  "english_translation": "Hello",
  "sentence_type": "greeting",
  "emotion": "happy",
  "emotion_intensity": "medium",
  "confidence": "high",
  "can_identify": true,
  "needs_repeat": false
}

IMPORTANT:
- Return ONLY valid JSON, no markdown formatting
- If signs are unclear: set "can_identify": false and "needs_repeat": true
- The "english_translation" must be a natural, complete English sentence
- Focus on identifying ALL signs in sequence, not just the last one

Now analyze the video:"""
    
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
                return "fast_movements"
            elif avg_motion < 5:
                return "slow_movements"
            else:
                return "mixed_speed"
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
    
    # def draw_landmarks_live(self, frame):
    #     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     annotated = frame.copy()
        
    #     hand_results = self.hands.process(rgb_frame)
    #     if hand_results.multi_hand_landmarks:
    #         for hand_landmarks in hand_results.multi_hand_landmarks:
    #             self.mp_draw.draw_landmarks(
    #                 annotated,
    #                 hand_landmarks,
    #                 self.mp_hands.HAND_CONNECTIONS,
    #                 self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
    #                 self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)
    #             )
        
    #     face_results = self.face_mesh.process(rgb_frame)
    #     if face_results.multi_face_landmarks:
    #         for face_landmarks in face_results.multi_face_landmarks:
    #             h, w, _ = annotated.shape
    #             key_points = [33, 133, 362, 263, 70, 300, 61, 291]
    #             for idx in key_points:
    #                 lm = face_landmarks.landmark[idx]
    #                 x, y = int(lm.x * w), int(lm.y * h)
    #                 cv2.circle(annotated, (x, y), 2, (255, 0, 0), -1)
        
    #     return annotated
    
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
        except Exception as e:
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
            return None
    
    def analyze_with_gemini(self, frames):
        speed_profile = self.analyze_speed_profile(frames)
        
        frames_with_landmarks = []
        video_path = None
        
        try:
            for i, frame in enumerate(frames):
                annotated = self.draw_landmarks(frame)
                if annotated is not None:
                    frames_with_landmarks.append(annotated)
            
            if not frames_with_landmarks:
                return None
            
            video_path = self.create_video_from_frames(frames_with_landmarks, fps=15)
            
            if video_path is None:
                return None
            
            video_size = os.path.getsize(video_path)
            
            with open(video_path, 'rb') as f:
                uploaded_file = self.client.files.upload(
                    file=f,
                    config={'mime_type': 'video/mp4'}
                )
            
            self.uploaded_files.append(uploaded_file.name)
            
            max_wait = 60
            wait_time = 0
            while uploaded_file.state.name == "PROCESSING" and wait_time < max_wait:
                time.sleep(2)
                wait_time += 2
                uploaded_file = self.client.files.get(name=uploaded_file.name)
            
            if uploaded_file.state.name == "FAILED":
                return None
            
            if wait_time >= max_wait:
                return None
            
            prompt = self.create_expert_prompt()
            
            if speed_profile != "unknown":
                speed_note = f"\n\nNOTE: This video contains {speed_profile.replace('_', ' ')}. Remember to accept all speed variations."
                prompt += speed_note
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt, uploaded_file],
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    top_p=0.95,
                    max_output_tokens=2048
                )
            )
            
            response_text = response.text.strip()
            
            result = self.parse_response(response_text)
            
            return result
            
        except Exception as e:
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
            
        except json.JSONDecodeError as e:
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
                'english_translation': text.strip()[:200],
                'confidence': 'low',
                'emotion': 'neutral',
                'needs_repeat': True
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
        except Exception as e:
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
        except Exception as e:
            return None
    
    def generate_text_file(self, text, emotion='neutral'):
        try:
            text_buffer = io.BytesIO()
            text_buffer.write(text.encode('utf-8'))
            text_buffer.seek(0)
            return text_buffer.read()
        except Exception as e:
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
                'has_text': False
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
                    'has_text': False
                }
            
            result = self.analyze_with_gemini(frames)
            
            if not result:
                self.cleanup_all_cache()
                return {
                    'success': False,
                    'error': 'Gemini analysis failed - check logs above',
                    'audio_file': None,
                    'text_file': None,
                    'has_audio': False,
                    'has_text': False
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
            
            # if not needs_repeat:
            #     self.cleanup_all_cache()
            
            return {
                'success': True,
                'audio_file': audio_bytes,
                'text_file': text_bytes,
                'has_audio': audio_bytes is not None,
                'has_text': text_bytes is not None
            }
            
        except Exception as e:
            self.cleanup_all_cache()
            
            return {
                'success': False,
                'error': str(e),
                'audio_file': None,
                'text_file': None,
                'has_audio': False,
                'has_text': False
            }
    
    def __del__(self):
        try:
            self.cleanup_all_cache()
        except:
            pass
