"""
Python ML Service - PyTorch 모델들을 위한 별도 서비스
Go 애플리케이션에서 HTTP API로 호출
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
import soundfile as sf
from datetime import datetime
import json
import re
import subprocess
import librosa
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

DOWNLOAD_FOLDER = 'downloads'
GEMINI_API_KEY = "AIzaSyA73IrEjd4T3fkjdDo7CTbJibWclowEmNI"

# 전역 모델 변수
tts_processor = None
tts_model = None
tts_model_type = None
whisper_model = None
whisper_processor = None
blip_processor = None
blip_model = None
text_embedder = None  # 텍스트 임베딩 모델
image_embedder = None  # 이미지 임베딩 모델

def load_tts_model():
    """TTS 모델 로딩 (영어용 - Bark만 사용)"""
    global tts_processor, tts_model, tts_model_type
    
    if tts_model_type == 'bark' and tts_model is not None:
        return tts_processor, tts_model
    
    tts_processor = None
    tts_model = None
    
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    
    try:
        print("Bark TTS 모델 로딩 중...")
        from transformers import BarkModel, AutoProcessor
        
        model_name = "suno/bark-small"
        tts_processor = AutoProcessor.from_pretrained(model_name)
        tts_model = BarkModel.from_pretrained(model_name)
        tts_model = tts_model.to(device)
        tts_model.eval()
        tts_model_type = 'bark'
        print(f"Bark TTS 모델 로딩 완료 ({'GPU' if use_cuda else 'CPU'} 모드)")
    except Exception as e:
        print(f"Bark 모델 로딩 실패: {e}")
        tts_processor = None
        tts_model = None
        tts_model_type = None
    
    return tts_processor, tts_model

def load_whisper_model():
    """Whisper 모델 로딩"""
    global whisper_model, whisper_processor
    
    if whisper_model is not None:
        return whisper_model, whisper_processor
    
    try:
        print("Whisper 모델 로딩 중...")
        try:
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            model_name = "openai/whisper-small"
            whisper_processor = WhisperProcessor.from_pretrained(model_name)
            whisper_model = WhisperForConditionalGeneration.from_pretrained(model_name)
            
            use_cuda = torch.cuda.is_available()
            device = "cuda" if use_cuda else "cpu"
            whisper_model = whisper_model.to(device)
            whisper_model.eval()
            
            print(f"Whisper 모델 로딩 완료 ({'GPU' if use_cuda else 'CPU'} 모드)")
        except ImportError:
            try:
                import whisper
                whisper_model = whisper.load_model("small")
                whisper_processor = None
                print("Whisper 모델 로딩 완료 (openai-whisper)")
            except ImportError:
                print("Whisper 라이브러리를 찾을 수 없습니다.")
                whisper_model = None
                whisper_processor = None
    except Exception as e:
        print(f"Whisper 모델 로딩 실패: {e}")
        whisper_model = None
        whisper_processor = None
    
    return whisper_model, whisper_processor

def load_blip_model():
    """BLIP 비전 모델 로딩"""
    global blip_processor, blip_model
    
    if blip_model is not None:
        return blip_processor, blip_model
    
    try:
        print("BLIP 비전 모델 로딩 중...")
        from transformers import BlipProcessor, BlipForConditionalGeneration
        from PIL import Image
        
        model_name = "Salesforce/blip-image-captioning-base"
        blip_processor = BlipProcessor.from_pretrained(model_name)
        blip_model = BlipForConditionalGeneration.from_pretrained(model_name)
        
        use_cuda = torch.cuda.is_available()
        device = "cuda" if use_cuda else "cpu"
        blip_model = blip_model.to(device)
        blip_model.eval()
        
        print(f"BLIP 모델 로딩 완료 ({'GPU' if use_cuda else 'CPU'} 모드)")
    except Exception as e:
        print(f"BLIP 모델 로딩 실패: {e}")
        blip_processor = None
        blip_model = None
    
    return blip_processor, blip_model

def detect_language(text):
    """텍스트 언어 감지"""
    korean_chars = sum(1 for char in text if '\uAC00' <= char <= '\uD7A3')
    total_chars = len([c for c in text if c.isalpha()])
    if total_chars > 0:
        korean_ratio = korean_chars / total_chars
        return 'ko' if korean_ratio > 0.3 else 'en'
    return 'en'

def split_text_for_tts(text, max_length=400):
    """TTS를 위한 텍스트 분할"""
    sentences = re.split(r'([.!?]\s+)', text)
    
    chunks = []
    current_chunk = ""
    
    for i in range(0, len(sentences), 2):
        sentence = sentences[i] + (sentences[i+1] if i+1 < len(sentences) else "")
        test_chunk = current_chunk + sentence if current_chunk else sentence
        estimated_length = len(test_chunk.split()) * 1.3
        
        if estimated_length > max_length and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk = test_chunk
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    if not chunks:
        chunks = [text]
    
    return chunks

@app.route('/tts', methods=['POST'])
def tts():
    """TTS 생성 API"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        language = data.get('language', 'auto')
        
        if not text:
            return jsonify({'success': False, 'error': 'Text is required'}), 400
        
        if language == 'auto':
            language = detect_language(text)
        
        # 한국어 TTS
        if language == 'ko':
            try:
                from gtts import gTTS
                
                tts_folder = os.path.join(DOWNLOAD_FOLDER, 'tts')
                os.makedirs(tts_folder, exist_ok=True)
                
                filename = f"tts_ko_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
                output_path = os.path.join(tts_folder, filename)
                
                tts = gTTS(text=text, lang='ko', slow=False)
                tts.save(output_path)
                
                return jsonify({
                    'success': True,
                    'filename': filename,
                    'message': 'Korean TTS generated'
                })
            except ImportError:
                return jsonify({'success': False, 'error': 'gTTS not installed'}), 500
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
        
        # 영어 TTS (Bark)
        if language == 'en':
            processor, model = load_tts_model()
            
            if processor is None or model is None:
                return jsonify({'success': False, 'error': 'Bark model not loaded'}), 500
            
            try:
                use_cuda = torch.cuda.is_available()
                device = "cuda" if use_cuda else "cpu"
                
                if len(text.split()) > 150:
                    text_chunks = split_text_for_tts(text, max_length=150)
                else:
                    text_chunks = [text]
                
                speech_chunks = []
                sample_rate = 24000
                
                for i, chunk in enumerate(text_chunks):
                    try:
                        inputs = processor(chunk, return_tensors="pt")
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        
                        with torch.no_grad():
                            try:
                                audio_array = model.generate(**inputs, pad_token_id=10000, voice_preset="v2/en_speaker_6")
                            except:
                                audio_array = model.generate(**inputs, pad_token_id=10000)
                        
                        if isinstance(audio_array, torch.Tensor):
                            audio_array = audio_array.cpu().numpy()
                        
                        if audio_array.ndim > 1:
                            if audio_array.shape[0] == 1:
                                audio_array = audio_array[0]
                            else:
                                audio_array = audio_array.flatten()
                        
                        if audio_array.dtype != np.float32:
                            audio_array = audio_array.astype(np.float32)
                        
                        max_val = np.abs(audio_array).max()
                        if max_val > 0:
                            audio_array = audio_array / max_val
                        
                        audio_array = np.clip(audio_array, -1.0, 1.0)
                        
                        silence = np.zeros(int(sample_rate * 0.3))
                        speech_chunks.append(audio_array)
                        if i < len(text_chunks) - 1:
                            speech_chunks.append(silence)
                    except Exception as e:
                        print(f"Bark chunk {i+1} error: {e}")
                        continue
                
                if not speech_chunks:
                    return jsonify({'success': False, 'error': 'All chunks failed'}), 500
                
                speech = np.concatenate(speech_chunks)
                speech = np.clip(speech, -1.0, 1.0)
                
                tts_folder = os.path.join(DOWNLOAD_FOLDER, 'tts')
                os.makedirs(tts_folder, exist_ok=True)
                
                filename = f"tts_en_bark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
                output_path = os.path.join(tts_folder, filename)
                sf.write(output_path, speech, sample_rate)
                
                return jsonify({
                    'success': True,
                    'filename': filename,
                    'message': 'English TTS generated (Bark)'
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
        
        return jsonify({'success': False, 'error': 'Unsupported language'}), 400
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def extract_audio_from_video(video_path, output_audio_path):
    """동영상에서 오디오 추출"""
    try:
        command = [
            'ffmpeg', '-y', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
            output_audio_path
        ]
        subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        return True
    except Exception as e:
        print(f"오디오 추출 실패: {e}")
        return False

def transcribe_audio(audio_path):
    """오디오를 텍스트로 변환"""
    model, processor = load_whisper_model()
    
    if model is None:
        return None, "Whisper 모델을 로드할 수 없습니다."
    
    try:
        if processor is not None:
            audio, sr = librosa.load(audio_path, sr=16000)
            audio_duration = len(audio) / sr
            print(f"오디오 길이: {audio_duration:.2f}초")
            
            chunk_duration = 30.0
            chunk_samples = int(chunk_duration * sr)
            all_transcriptions = []
            
            if len(audio) > chunk_samples:
                print(f"긴 오디오 감지: {len(audio) / sr:.2f}초, 청크 단위로 처리합니다.")
                num_chunks = int(np.ceil(len(audio) / chunk_samples))
                
                for i in range(num_chunks):
                    start_idx = i * chunk_samples
                    end_idx = min((i + 1) * chunk_samples, len(audio))
                    audio_chunk = audio[start_idx:end_idx]
                    
                    inputs = processor(audio_chunk, sampling_rate=16000, return_tensors="pt")
                    device = next(model.parameters()).device
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        generated_ids = model.generate(**inputs, max_length=512, max_new_tokens=256)
                    
                    chunk_transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    if chunk_transcription.strip():
                        all_transcriptions.append(chunk_transcription.strip())
                
                transcription = " ".join(all_transcriptions)
            else:
                inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
                device = next(model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_length=512, max_new_tokens=256)
                
                transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            print(f"음성 인식 완료: {len(transcription)}자")
            return transcription, None
        else:
            print("openai-whisper로 전체 오디오 인식 중...")
            result = model.transcribe(audio_path, language=None, verbose=False)
            transcription = result["text"]
            print(f"음성 인식 완료: {len(transcription)}자")
            return transcription, None
    except Exception as e:
        error_msg = str(e)
        print(f"음성 인식 오류: {error_msg}")
        import traceback
        traceback.print_exc()
        return None, error_msg

def extract_key_frames(video_path, num_frames=5):
    """동영상에서 주요 프레임 추출"""
    try:
        probe_cmd = [
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', video_path
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        duration = float(result.stdout.strip())
        
        frame_times = []
        if duration > 0:
            interval = duration / (num_frames + 1)
            for i in range(1, num_frames + 1):
                frame_times.append(i * interval)
        
        frames = []
        temp_dir = os.path.join(DOWNLOAD_FOLDER, 'temp_frames')
        os.makedirs(temp_dir, exist_ok=True)
        
        for idx, time in enumerate(frame_times):
            frame_path = os.path.join(temp_dir, f"frame_{idx}.jpg")
            command = [
                'ffmpeg', '-y', '-ss', str(time),
                '-i', video_path, '-vframes', '1',
                '-q:v', '2', frame_path
            ]
            try:
                subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
                if os.path.exists(frame_path):
                    frames.append(frame_path)
            except:
                continue
        
        return frames
    except Exception as e:
        print(f"프레임 추출 오류: {e}")
        return []

def has_repetition(text, threshold=3):
    """텍스트에 반복이 있는지 확인"""
    if not text:
        return False
    
    words = text.split()
    if len(words) < threshold:
        return False
    
    for i in range(len(words) - threshold + 1):
        if len(set(words[i:i+threshold])) == 1:
            return True
    
    for i in range(len(words) - 4):
        phrase1 = " ".join(words[i:i+2])
        phrase2 = " ".join(words[i+2:i+4])
        if phrase1 == phrase2:
            return True
    
    return False

def analyze_frame_with_blip(image_path):
    """BLIP 모델을 사용하여 프레임 이미지 분석"""
    processor, model = load_blip_model()
    
    if model is None or processor is None:
        return None
    
    try:
        from PIL import Image
        
        image = Image.open(image_path).convert('RGB')
        inputs = processor(image, return_tensors="pt")
        
        use_cuda = torch.cuda.is_available()
        device = "cuda" if use_cuda else "cpu"
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_length=50, 
                num_beams=3,
                repetition_penalty=1.5
            )
        
        description = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        if has_repetition(description):
            words = description.split()
            unique_words = []
            prev_word = None
            for word in words:
                if word != prev_word:
                    unique_words.append(word)
                prev_word = word
            description = " ".join(unique_words[:20])
        
        return description
    except Exception as e:
        print(f"프레임 분석 오류 ({image_path}): {e}")
        return None

def summarize_transcription(transcription, summary_ratio=0.25):
    """음성 인식 텍스트를 Gemini API로 요약"""
    if not transcription or len(transcription.strip()) < 50:
        return transcription
    
    text = transcription.strip()
    original_length = len(text)
    
    print(f"Gemini API로 요약 시작: 원본 {original_length}자")
    
    try:
        import google.generativeai as genai
        
        genai.configure(api_key=GEMINI_API_KEY)
        
        available_models = []
        try:
            print("사용 가능한 Gemini 모델 목록 확인 중...")
            models_list = genai.list_models()
            for m in models_list:
                if hasattr(m, 'supported_generation_methods') and 'generateContent' in m.supported_generation_methods:
                    available_models.append(m.name)
                    print(f"  ✓ {m.name}")
        except Exception as e:
            print(f"모델 목록 조회 실패: {e}")
        
        model = None
        if available_models:
            model_name = available_models[0]
            try:
                model = genai.GenerativeModel(model_name)
                print(f"✅ 모델 로드 성공: {model_name}")
            except Exception as e:
                print(f"모델 {model_name} 로드 실패: {e}")
                model = None
        
        if model is None:
            model_names = [
                'models/gemini-1.5-pro-latest',
                'models/gemini-1.5-flash-latest', 
                'models/gemini-pro',
                'gemini-1.5-pro',
                'gemini-1.5-flash',
                'gemini-pro'
            ]
            
            for model_name in model_names:
                try:
                    model = genai.GenerativeModel(model_name)
                    test_response = model.generate_content("테스트")
                    if test_response and test_response.text:
                        print(f"✅ 모델 API 호출 성공: {model_name}")
                        break
                    model = None
                except Exception as e:
                    model = None
                    continue
        
        if model is None:
            raise Exception("사용 가능한 Gemini 모델을 찾을 수 없습니다.")
        
        prompt = f"""다음은 동영상의 음성 인식 결과입니다. 핵심 내용을 간결하게 요약해주세요.

원본 텍스트:
{text}

요약 요구사항:
- 핵심 내용만 간결하게 요약
- 원본의 약 {int(summary_ratio*100)}% 정도의 길이로 압축
- 자연스러운 한국어로 작성
- 불필요한 반복 제거

요약:"""
        
        response = model.generate_content(prompt)
        
        if response and response.text:
            summary = response.text.strip()
            print(f"✅ Gemini API 요약 성공: {original_length}자 -> {len(summary)}자 ({len(summary)/original_length*100:.1f}%)")
            return summary
        else:
            raise Exception("Gemini API 응답이 비어있습니다.")
    
    except ImportError:
        print("google-generativeai 패키지가 설치되지 않았습니다.")
        sentences = re.split(r'[.!?]\s+', text)
        target_sentences = max(3, int(len(sentences) * summary_ratio))
        return ". ".join(sentences[:target_sentences]) + "."
    
    except Exception as e:
        error_msg = str(e)
        print(f"Gemini API 요약 실패: {error_msg}")
        sentences = re.split(r'[.!?]\s+', text)
        target_sentences = max(3, int(len(sentences) * summary_ratio))
        return ". ".join(sentences[:target_sentences]) + "."

def generate_video_summary_from_frames(frame_descriptions, transcription="", transcription_summary=""):
    """프레임 분석 결과와 음성 인식을 종합하여 동영상 요약 생성"""
    summary_parts = []
    
    if frame_descriptions:
        unique_descriptions = []
        for desc in frame_descriptions:
            if desc and desc.strip() and desc not in unique_descriptions:
                unique_descriptions.append(desc.strip())
        
        if unique_descriptions:
            summary_parts.append("주요 장면: " + ", ".join(unique_descriptions[:3]))
    
    if transcription_summary and transcription_summary.strip():
        summary_parts.append(f"내용 요약: {transcription_summary}")
    elif transcription and transcription.strip():
        transcription_clean = transcription.strip()
        if len(transcription_clean) > 200:
            transcription_clean = transcription_clean[:200] + "..."
        summary_parts.append(f"내용: {transcription_clean}")
    
    if summary_parts:
        return " ".join(summary_parts)
    else:
        return "동영상 내용을 분석할 수 없습니다."

def analyze_video_content(video_path):
    """동영상/오디오 내용 분석 및 요약"""
    results = {
        'transcription': '',
        'summary': '',
        'key_frames': [],
        'frame_descriptions': [],
        'duration': 0,
        'error': None
    }
    
    try:
        # 파일 확장자 확인
        file_ext = os.path.splitext(video_path)[1].lower()
        is_audio = file_ext in ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac']
        
        probe_cmd = [
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', video_path
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        if result.returncode == 0:
            results['duration'] = float(result.stdout.strip())
        
        # 오디오 파일이 아닌 경우에만 프레임 추출
        if not is_audio:
            key_frames = extract_key_frames(video_path, num_frames=5)
            results['key_frames'] = [os.path.basename(f) for f in key_frames]
            
            frame_descriptions = []
            if key_frames:
                print(f"프레임 분석 중... ({len(key_frames)}개 프레임)")
                for frame_path in key_frames:
                    description = analyze_frame_with_blip(frame_path)
                    if description:
                        frame_descriptions.append(description)
                        print(f"  - {os.path.basename(frame_path)}: {description}")
            
            results['frame_descriptions'] = frame_descriptions
        
        # 음성 인식 수행
        temp_audio_path = os.path.join(DOWNLOAD_FOLDER, 'temp_audio.wav')
        transcription = ""
        transcription_summary = ""
        
        if is_audio:
            # 오디오 파일은 직접 음성 인식
            transcription, error = transcribe_audio(video_path)
        else:
            # 동영상 파일은 오디오 추출 후 음성 인식
            if extract_audio_from_video(video_path, temp_audio_path):
                transcription, error = transcribe_audio(temp_audio_path)
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
            else:
                error = "Failed to extract audio"
        
        if transcription:
            results['transcription'] = transcription
            
            print("음성 인식 텍스트 요약 중...")
            transcription_summary = summarize_transcription(transcription)
            results['transcription_summary'] = transcription_summary
            print(f"요약 결과: {transcription_summary}")
        else:
            if error:
                print(f"음성 인식 경고: {error}")
        
        summary = generate_video_summary_from_frames(results.get('frame_descriptions', []), transcription, transcription_summary)
        results['summary'] = summary
        
    except Exception as e:
        results['error'] = str(e)
        print(f"파일 분석 오류: {e}")
        import traceback
        traceback.print_exc()
    
    return results

@app.route('/analyze_video', methods=['POST'])
def analyze_video():
    """동영상/오디오 분석 API"""
    try:
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': 'Video/Audio file is required'}), 400
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # 파일 확장자 확인
        file_ext = os.path.splitext(video_file.filename)[1].lower()
        is_audio = file_ext in ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac']
        
        # 오디오 파일은 tts 폴더에 저장
        if is_audio:
            folder = os.path.join(DOWNLOAD_FOLDER, 'tts')
        else:
            folder = os.path.join(DOWNLOAD_FOLDER, 'video')
        
        os.makedirs(folder, exist_ok=True)
        
        filename = secure_filename(video_file.filename)
        video_path = os.path.join(folder, filename)
        video_file.save(video_path)
        
        # 동영상 분석 수행
        results = analyze_video_content(video_path)
        
        if results['error']:
            return jsonify({
                'success': False,
                'error': f'분석 중 오류 발생: {results["error"]}'
            }), 500
        
        # 결과를 JSON 파일로 저장
        analysis_folder = os.path.join(DOWNLOAD_FOLDER, 'analysis')
        os.makedirs(analysis_folder, exist_ok=True)
        
        result_filename = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        result_path = os.path.join(analysis_folder, result_filename)
        
        result_data = {
            'video_filename': filename,
            'duration': results['duration'],
            'transcription': results['transcription'],
            'transcription_summary': results.get('transcription_summary', ''),
            'summary': results['summary'],
            'key_frames': results['key_frames'],
            'frame_descriptions': results.get('frame_descriptions', []),
            'analyzed_at': datetime.now().isoformat()
        }
        
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        # 임시 프레임 파일들을 analysis 폴더로 이동
        temp_dir = os.path.join(DOWNLOAD_FOLDER, 'temp_frames')
        if os.path.exists(temp_dir):
            import shutil
            for frame_file in os.listdir(temp_dir):
                if frame_file.startswith('frame_'):
                    src = os.path.join(temp_dir, frame_file)
                    dst = os.path.join(analysis_folder, frame_file)
                    try:
                        shutil.move(src, dst)
                    except:
                        pass
        
        return jsonify({
            'success': True,
            'result': result_data,
            'result_file': result_filename
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def load_text_embedder():
    """텍스트 임베딩 모델 로딩"""
    global text_embedder
    
    if text_embedder is not None:
        return text_embedder
    
    try:
        print("텍스트 임베딩 모델 로딩 중...")
        from sentence_transformers import SentenceTransformer
        # 한국어와 영어를 모두 지원하는 모델
        text_embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        print("텍스트 임베딩 모델 로딩 완료")
    except ImportError:
        print("sentence-transformers가 설치되지 않았습니다. pip install sentence-transformers")
        text_embedder = None
    except Exception as e:
        print(f"텍스트 임베딩 모델 로딩 실패: {e}")
        text_embedder = None
    
    return text_embedder

def load_image_embedder():
    """이미지 임베딩 모델 로딩"""
    global image_embedder
    
    if image_embedder is not None:
        return image_embedder
    
    try:
        print("이미지 임베딩 모델 로딩 중...")
        from sentence_transformers import SentenceTransformer
        # CLIP 기반 이미지 임베딩 모델
        image_embedder = SentenceTransformer('clip-ViT-B-32')
        print("이미지 임베딩 모델 로딩 완료")
    except ImportError:
        print("sentence-transformers가 설치되지 않았습니다. pip install sentence-transformers")
        image_embedder = None
    except Exception as e:
        print(f"이미지 임베딩 모델 로딩 실패: {e}")
        image_embedder = None
    
    return image_embedder

@app.route('/generate_embedding', methods=['POST'])
def generate_embedding():
    """임베딩 생성 API"""
    try:
        data = request.get_json()
        file_type = data.get('file_type', '')  # 'video', 'image', 'audio', 'text'
        file_path = data.get('file_path', '')  # 파일 경로 (검색 쿼리인 경우 빈 문자열 가능)
        text = data.get('text', '')  # 음성 인식 텍스트 또는 이미지 설명 또는 검색 쿼리
        image_path = data.get('image_path')  # 이미지 파일 경로
        
        # 텍스트만 있는 경우 (검색 쿼리)는 허용
        if not text and not image_path:
            return jsonify({'success': False, 'error': 'text or image_path is required'}), 400
        
        # 파일이 있는 경우 file_type과 file_path 필요
        if image_path and (not file_type or not file_path):
            return jsonify({'success': False, 'error': 'file_type and file_path are required when image_path is provided'}), 400
        
        result = {
            'file_path': file_path or '',
            'file_type': file_type or 'text',
            'text_embedding': None,
            'image_embedding': None,
        }
        
        # 이미지인 경우 BLIP으로 설명 생성 후 텍스트 임베딩도 생성
        if image_path and os.path.exists(image_path) and not text:
            # BLIP으로 이미지 설명 생성
            processor, model = load_blip_model()
            if model is not None and processor is not None:
                try:
                    from PIL import Image
                    image = Image.open(image_path).convert('RGB')
                    inputs = processor(image, return_tensors="pt")
                    
                    use_cuda = torch.cuda.is_available()
                    device = "cuda" if use_cuda else "cpu"
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        generated_ids = model.generate(
                            **inputs, 
                            max_length=50, 
                            num_beams=3,
                            repetition_penalty=1.5
                        )
                    
                    description = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    text = description  # BLIP 설명을 텍스트로 사용
                    print(f"이미지 설명 생성: {description}")
                except Exception as e:
                    print(f"이미지 설명 생성 실패: {e}")
        
        # 텍스트 임베딩 생성
        if text:
            embedder = load_text_embedder()
            if embedder:
                try:
                    embedding = embedder.encode(text, convert_to_numpy=True)
                    result['text_embedding'] = embedding.tolist()
                except Exception as e:
                    print(f"텍스트 임베딩 생성 실패: {e}")
        
        # 이미지 임베딩 생성
        if image_path and os.path.exists(image_path):
            embedder = load_image_embedder()
            if embedder:
                try:
                    from PIL import Image
                    image = Image.open(image_path).convert('RGB')
                    embedding = embedder.encode(image, convert_to_numpy=True)
                    result['image_embedding'] = embedding.tolist()
                except Exception as e:
                    print(f"이미지 임베딩 생성 실패: {e}")
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("Python ML Service starting on port 5002...")
    app.run(host='0.0.0.0', port=5002, debug=True)

