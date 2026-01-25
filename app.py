import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import subprocess
import yt_dlp
from flask import Flask, flash, redirect, render_template, request, send_from_directory, url_for, jsonify
from werkzeug.utils import secure_filename
import json
import re
from datetime import datetime

# Audio processing imports
import torch
import numpy as np
import soundfile as sf

app = Flask(__name__)
app.config['SECRET_KEY'] = 'a_super_secret_key'
DOWNLOAD_FOLDER = 'downloads'
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER

if not os.path.exists(DOWNLOAD_FOLDER):
    os.makedirs(DOWNLOAD_FOLDER)

def check_ffmpeg_available():
    """ffmpeg가 설치되어 있는지 확인"""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False

# TTS 모델 로딩 (최초 1회만)
tts_processor = None
tts_model = None
tts_model_type = None  # 'bark'

# 동영상 분석 모델 로딩 (최초 1회만)
whisper_model = None
whisper_processor = None
blip_processor = None
blip_model = None

# Gemini API 설정
GEMINI_API_KEY = "AIzaSyA73IrEjd4T3fkjdDo7CTbJibWclowEmNI"

def load_tts_model():
    """TTS 모델 로딩 (영어용 - Bark만 사용)"""
    global tts_processor, tts_model, tts_model_type
    
    if tts_model_type == 'bark' and tts_model is not None:
        return tts_processor, tts_model
    
    # 기존 모델 정리
    tts_processor = None
    tts_model = None
    
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    
    try:
        print("Bark TTS 모델 로딩 중... (고품질)")
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
        import traceback
        traceback.print_exc()
        tts_processor = None
        tts_model = None
        tts_model_type = None
    
    return tts_processor, tts_model

def detect_language(text):
    """텍스트 언어 감지 (간단한 휴리스틱)"""
    korean_chars = sum(1 for char in text if '\uAC00' <= char <= '\uD7A3')
    total_chars = len([c for c in text if c.isalpha()])
    if total_chars > 0:
        korean_ratio = korean_chars / total_chars
        return 'ko' if korean_ratio > 0.3 else 'en'
    return 'en'

def split_text_for_tts(text, max_length=400):
    """TTS를 위한 텍스트 분할 (문장 단위로 분할)"""
    # 문장 끝 구분자로 분할
    sentences = re.split(r'([.!?]\s+)', text)
    
    chunks = []
    current_chunk = ""
    
    for i in range(0, len(sentences), 2):
        sentence = sentences[i] + (sentences[i+1] if i+1 < len(sentences) else "")
        
        # 현재 청크에 문장을 추가했을 때 길이 확인
        test_chunk = current_chunk + sentence if current_chunk else sentence
        
        # 토큰 길이 대략 추정 (공백 포함 단어 수의 1.3배)
        estimated_length = len(test_chunk.split()) * 1.3
        
        if estimated_length > max_length and current_chunk:
            # 현재 청크 저장하고 새 청크 시작
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk = test_chunk
    
    # 마지막 청크 추가
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # 청크가 없으면 원본 텍스트 반환 (너무 짧은 경우)
    if not chunks:
        chunks = [text]
    
    return chunks

def get_downloaded_files():
    return sorted(os.listdir(app.config['DOWNLOAD_FOLDER']), reverse=True)

def get_tts_files():
    """TTS 관련 파일만 필터링"""
    files = get_downloaded_files()
    return [f for f in files if f.startswith('tts_') and (f.endswith('.wav') or f.endswith('.mp3'))]

def get_video_files():
    """동영상 파일만 필터링"""
    files = get_downloaded_files()
    return [f for f in files if f.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.mpg', '.mpeg'))]

def load_whisper_model():
    """Whisper 모델 로딩 (음성 인식용)"""
    global whisper_model, whisper_processor
    
    if whisper_model is not None:
        return whisper_model, whisper_processor
    
    try:
        print("Whisper 모델 로딩 중...")
        try:
            # transformers의 Whisper 사용 시도
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            model_name = "openai/whisper-small"  # 또는 "openai/whisper-base", "openai/whisper-medium"
            whisper_processor = WhisperProcessor.from_pretrained(model_name)
            whisper_model = WhisperForConditionalGeneration.from_pretrained(model_name)
            
            use_cuda = torch.cuda.is_available()
            device = "cuda" if use_cuda else "cpu"
            whisper_model = whisper_model.to(device)
            whisper_model.eval()
            
            print(f"Whisper 모델 로딩 완료 ({'GPU' if use_cuda else 'CPU'} 모드)")
        except ImportError:
            # openai-whisper 패키지 사용 시도
            try:
                import whisper
                whisper_model = whisper.load_model("small")  # base, small, medium, large
                whisper_processor = None
                print("Whisper 모델 로딩 완료 (openai-whisper)")
            except ImportError:
                print("Whisper 라이브러리를 찾을 수 없습니다. transformers 또는 openai-whisper를 설치하세요.")
                whisper_model = None
                whisper_processor = None
    except Exception as e:
        print(f"Whisper 모델 로딩 실패: {e}")
        import traceback
        traceback.print_exc()
        whisper_model = None
        whisper_processor = None
    
    return whisper_model, whisper_processor

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
    """오디오를 텍스트로 변환 (전체 오디오 인식)"""
    model, processor = load_whisper_model()
    
    if model is None:
        return None, "Whisper 모델을 로드할 수 없습니다."
    
    try:
        if processor is not None:
            # transformers Whisper 사용
            import librosa
            
            # 오디오 길이 확인
            audio, sr = librosa.load(audio_path, sr=16000)
            audio_duration = len(audio) / sr
            print(f"오디오 길이: {audio_duration:.2f}초")
            
            # 긴 오디오는 청크로 나눠서 처리 (30초 단위)
            chunk_duration = 30.0  # 30초 청크
            chunk_samples = int(chunk_duration * sr)
            
            all_transcriptions = []
            
            if len(audio) > chunk_samples:
                # 긴 오디오: 청크로 나눠서 처리
                print(f"긴 오디오 감지: {len(audio) / sr:.2f}초, 청크 단위로 처리합니다.")
                num_chunks = int(np.ceil(len(audio) / chunk_samples))
                
                for i in range(num_chunks):
                    start_idx = i * chunk_samples
                    end_idx = min((i + 1) * chunk_samples, len(audio))
                    audio_chunk = audio[start_idx:end_idx]
                    
                    print(f"청크 {i+1}/{num_chunks} 처리 중... ({start_idx/sr:.1f}초 ~ {end_idx/sr:.1f}초)")
                    
                    inputs = processor(audio_chunk, sampling_rate=16000, return_tensors="pt")
                    device = next(model.parameters()).device
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        # max_length 제한 제거 또는 충분히 크게 설정
                        generated_ids = model.generate(**inputs, max_length=512, max_new_tokens=256)
                    
                    chunk_transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    if chunk_transcription.strip():
                        all_transcriptions.append(chunk_transcription.strip())
                
                transcription = " ".join(all_transcriptions)
            else:
                # 짧은 오디오: 전체 처리
                inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
                device = next(model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    # max_length 제한 제거 또는 충분히 크게 설정
                    generated_ids = model.generate(**inputs, max_length=512, max_new_tokens=256)
                
                transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            print(f"음성 인식 완료: {len(transcription)}자")
            return transcription, None
        else:
            # openai-whisper 사용 (더 나은 긴 오디오 처리)
            print("openai-whisper로 전체 오디오 인식 중...")
            result = model.transcribe(audio_path, language=None, verbose=False)  # 자동 언어 감지
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
        # 동영상 길이 확인
        probe_cmd = [
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', video_path
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        duration = float(result.stdout.strip())
        
        # 균등하게 프레임 추출
        frame_times = []
        if duration > 0:
            interval = duration / (num_frames + 1)
            for i in range(1, num_frames + 1):
                frame_times.append(i * interval)
        
        frames = []
        temp_dir = os.path.join(app.config['DOWNLOAD_FOLDER'], 'temp_frames')
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

def load_blip_model():
    """BLIP 비전 모델 로딩 (이미지 분석용)"""
    global blip_processor, blip_model
    
    if blip_model is not None:
        return blip_processor, blip_model
    
    try:
        print("BLIP 비전 모델 로딩 중...")
        from transformers import BlipProcessor, BlipForConditionalGeneration
        from PIL import Image
        
        model_name = "Salesforce/blip-image-captioning-base"  # 또는 "Salesforce/blip-image-captioning-large"
        blip_processor = BlipProcessor.from_pretrained(model_name)
        blip_model = BlipForConditionalGeneration.from_pretrained(model_name)
        
        use_cuda = torch.cuda.is_available()
        device = "cuda" if use_cuda else "cpu"
        blip_model = blip_model.to(device)
        blip_model.eval()
        
        print(f"BLIP 모델 로딩 완료 ({'GPU' if use_cuda else 'CPU'} 모드)")
    except Exception as e:
        print(f"BLIP 모델 로딩 실패: {e}")
        import traceback
        traceback.print_exc()
        blip_processor = None
        blip_model = None
    
    return blip_processor, blip_model

def analyze_frame_with_blip(image_path):
    """BLIP 모델을 사용하여 프레임 이미지 분석 및 설명 생성"""
    processor, model = load_blip_model()
    
    if model is None or processor is None:
        return None
    
    try:
        from PIL import Image
        
        # 이미지 로드
        image = Image.open(image_path).convert('RGB')
        
        # BLIP으로 이미지 설명 생성
        inputs = processor(image, return_tensors="pt")
        
        use_cuda = torch.cuda.is_available()
        device = "cuda" if use_cuda else "cpu"
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_length=50, 
                num_beams=3,
                repetition_penalty=1.5  # 반복 억제
            )
        
        description = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # 반복 체크 및 제거
        if has_repetition(description):
            # 반복이 있으면 간단하게 처리
            words = description.split()
            unique_words = []
            prev_word = None
            for word in words:
                if word != prev_word:
                    unique_words.append(word)
                prev_word = word
            description = " ".join(unique_words[:20])  # 최대 20단어로 제한
        
        return description
    except Exception as e:
        print(f"프레임 분석 오류 ({image_path}): {e}")
        return None

def has_repetition(text, threshold=3):
    """텍스트에 반복이 있는지 확인 (프레임 분석용)"""
    if not text:
        return False
    
    words = text.split()
    if len(words) < threshold:
        return False
    
    # 같은 단어가 연속으로 3번 이상 반복되는지 확인
    for i in range(len(words) - threshold + 1):
        if len(set(words[i:i+threshold])) == 1:
            return True
    
    # 같은 구문이 반복되는지 확인 (2-3단어)
    for i in range(len(words) - 4):
        phrase1 = " ".join(words[i:i+2])
        phrase2 = " ".join(words[i+2:i+4])
        if phrase1 == phrase2:
            return True
    
    return False

def summarize_transcription(transcription, summary_ratio=0.25):
    """음성 인식 텍스트를 Gemini API로 요약"""
    if not transcription or len(transcription.strip()) < 50:
        return transcription
    
    text = transcription.strip()
    original_length = len(text)
    
    print(f"Gemini API로 요약 시작: 원본 {original_length}자")
    
    try:
        import google.generativeai as genai
        
        # Gemini API 설정
        genai.configure(api_key=GEMINI_API_KEY)
        
        # 먼저 사용 가능한 모델 목록 확인
        available_models = []
        try:
            print("사용 가능한 Gemini 모델 목록 확인 중...")
            models_list = genai.list_models()
            for m in models_list:
                if hasattr(m, 'supported_generation_methods') and 'generateContent' in m.supported_generation_methods:
                    available_models.append(m.name)
                    print(f"  ✓ {m.name}")
            if not available_models:
                print("  (사용 가능한 모델을 찾을 수 없음)")
        except Exception as e:
            print(f"모델 목록 조회 실패: {e}")
            import traceback
            traceback.print_exc()
        
        # 사용 가능한 모델이 있으면 첫 번째 사용, 없으면 기본 모델명 시도
        model = None
        if available_models:
            # 사용 가능한 모델 중 첫 번째 사용
            model_name = available_models[0]
            print(f"사용 가능한 모델 발견: {model_name}")
            try:
                model = genai.GenerativeModel(model_name)
                print(f"✅ 모델 로드 성공: {model_name}")
            except Exception as e:
                print(f"모델 {model_name} 로드 실패: {e}")
                model = None
        
        # 사용 가능한 모델이 없거나 로드 실패 시 기본 모델명들 시도
        if model is None:
            model_names = [
                'models/gemini-1.5-pro-latest',
                'models/gemini-1.5-flash-latest', 
                'models/gemini-pro',
                'gemini-1.5-pro',
                'gemini-1.5-flash',
                'gemini-pro'
            ]
            
            print("기본 모델명으로 시도 중...")
            for model_name in model_names:
                try:
                    model = genai.GenerativeModel(model_name)
                    print(f"✅ 모델 객체 생성 성공: {model_name}")
                    # 실제 API 호출 테스트 (짧은 프롬프트로)
                    test_response = model.generate_content("테스트")
                    if test_response and test_response.text:
                        print(f"✅ 모델 API 호출 성공: {model_name}")
                        break
                    else:
                        print(f"⚠️ 모델 {model_name} API 응답이 비어있음")
                        model = None
                except Exception as e:
                    error_msg = str(e)
                    print(f"  ✗ {model_name}: {error_msg[:150]}")
                    model = None
                    continue
        
        if model is None:
            raise Exception("사용 가능한 Gemini 모델을 찾을 수 없습니다. API 키와 모델 접근 권한을 확인하세요.")
        
        # 요약 프롬프트 생성
        prompt = f"""다음은 동영상의 음성 인식 결과입니다. 핵심 내용을 간결하게 요약해주세요.

원본 텍스트:
{text}

요약 요구사항:
- 핵심 내용만 간결하게 요약
- 원본의 약 {int(summary_ratio*100)}% 정도의 길이로 압축
- 자연스러운 한국어로 작성
- 불필요한 반복 제거

요약:"""
        
        # Gemini API 호출
        response = model.generate_content(prompt)
        
        if response and response.text:
            summary = response.text.strip()
            print(f"✅ Gemini API 요약 성공: {original_length}자 -> {len(summary)}자 ({len(summary)/original_length*100:.1f}%)")
            return summary
        else:
            print("⚠️ Gemini API 응답이 비어있습니다.")
            raise Exception("Gemini API 응답이 비어있습니다.")
    
    except ImportError:
        print("google-generativeai 패키지가 설치되지 않았습니다. pip install google-generativeai를 실행하세요.")
        # 폴백: 간단한 요약
        sentences = re.split(r'[.!?]\s+', text)
        target_sentences = max(3, int(len(sentences) * summary_ratio))
        return ". ".join(sentences[:target_sentences]) + "."
    
    except Exception as e:
        error_msg = str(e)
        print(f"Gemini API 요약 실패: {error_msg}")
        import traceback
        traceback.print_exc()
        # 폴백: 간단한 요약
        sentences = re.split(r'[.!?]\s+', text)
        target_sentences = max(3, int(len(sentences) * summary_ratio))
        return ". ".join(sentences[:target_sentences]) + "."


def generate_video_summary_from_frames(frame_descriptions, transcription="", transcription_summary=""):
    """프레임 분석 결과와 음성 인식을 종합하여 동영상 요약 생성"""
    summary_parts = []
    
    # 프레임 설명이 있으면 포함
    if frame_descriptions:
        unique_descriptions = []
        for desc in frame_descriptions:
            if desc and desc.strip() and desc not in unique_descriptions:
                unique_descriptions.append(desc.strip())
        
        if unique_descriptions:
            summary_parts.append("주요 장면: " + ", ".join(unique_descriptions[:3]))
    
    # 음성 인식 요약 결과가 있으면 포함
    if transcription_summary and transcription_summary.strip():
        summary_parts.append(f"내용 요약: {transcription_summary}")
    elif transcription and transcription.strip():
        # 요약이 없으면 원본의 첫 부분 사용
        transcription_clean = transcription.strip()
        if len(transcription_clean) > 200:
            transcription_clean = transcription_clean[:200] + "..."
        summary_parts.append(f"내용: {transcription_clean}")
    
    if summary_parts:
        return " ".join(summary_parts)
    else:
        return "동영상 내용을 분석할 수 없습니다."

def analyze_video_content(video_path):
    """동영상 내용 분석 및 요약 (비전 모델 사용)"""
    results = {
        'transcription': '',
        'summary': '',
        'key_frames': [],
        'frame_descriptions': [],
        'duration': 0,
        'error': None
    }
    
    try:
        # 동영상 길이 확인
        probe_cmd = [
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', video_path
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        if result.returncode == 0:
            results['duration'] = float(result.stdout.strip())
        
        # 주요 프레임 추출 (더 많은 프레임 추출)
        key_frames = extract_key_frames(video_path, num_frames=5)
        results['key_frames'] = key_frames
        
        # 각 프레임을 BLIP 모델로 분석
        frame_descriptions = []
        if key_frames:
            print(f"프레임 분석 중... ({len(key_frames)}개 프레임)")
            for frame_path in key_frames:
                description = analyze_frame_with_blip(frame_path)
                if description:
                    frame_descriptions.append(description)
                    print(f"  - {os.path.basename(frame_path)}: {description}")
        
        results['frame_descriptions'] = frame_descriptions
        
        # 오디오 추출 및 음성 인식
        temp_audio_path = os.path.join(app.config['DOWNLOAD_FOLDER'], 'temp_audio.wav')
        transcription = ""
        transcription_summary = ""
        if extract_audio_from_video(video_path, temp_audio_path):
            transcription, error = transcribe_audio(temp_audio_path)
            if transcription:
                results['transcription'] = transcription
                
                # 음성 인식 텍스트를 AI 모델로 요약
                print("음성 인식 텍스트 요약 중...")
                transcription_summary = summarize_transcription(transcription)
                results['transcription_summary'] = transcription_summary
                print(f"요약 결과: {transcription_summary}")
            else:
                if error:
                    print(f"음성 인식 경고: {error}")
            
            # 임시 오디오 파일 삭제
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
        
        # 비전 모델 분석 결과와 음성 인식 요약을 종합하여 최종 요약 생성
        summary = generate_video_summary_from_frames(frame_descriptions, transcription, transcription_summary)
        results['summary'] = summary
        
    except Exception as e:
        results['error'] = str(e)
        print(f"동영상 분석 오류: {e}")
        import traceback
        traceback.print_exc()
    
    return results

@app.route('/')
def index():
    files = get_downloaded_files()
    tts_files = get_tts_files()
    video_files = get_video_files()
    return render_template('index.html', files=files, tts_files=tts_files, video_files=video_files)

@app.route('/download', methods=['POST'])
def download():
    url = request.form.get('url')
    if not url:
        flash('URL is required!', 'danger')
        return redirect(url_for('index'))

    # 동영상 저장 폴더 설정
    video_folder = os.path.join(app.config['DOWNLOAD_FOLDER'], 'video')
    os.makedirs(video_folder, exist_ok=True)  # 폴더가 없으면 생성

    # ffmpeg 사용 가능 여부 확인
    ffmpeg_available = check_ffmpeg_available()
    
    format_options = (
        ['bestvideo+bestaudio/best', 'best', 'worst'] if ffmpeg_available
        else ['best', 'best[height<=720]', 'best[height<=480]', 'worst']
    )
    
    last_error = None
    for format_str in format_options:
        if not ffmpeg_available and '+' in format_str:
            continue
        
        try:
            ydl_opts = {
                'outtmpl': os.path.join(video_folder, '%(title)s.%(ext)s'),
                'format': format_str,
                'noplaylist': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            flash('Download successful!', 'success')
            return redirect(url_for('index'))
        except Exception as e:
            last_error = str(e)
            error_str = str(e).lower()
            if any(kw in error_str for kw in ['format is not available', 'requested format', 'format', 'ffmpeg', 'merging', 'network', 'timeout', 'unavailable', 'connection']):
                continue
            break
    
    error_msg = last_error or 'Unknown error'
    if 'format' in error_msg.lower():
        error_msg += ' (해당 동영상에 사용 가능한 형식이 없습니다. YouTube 제한일 수 있습니다.)'
    elif 'ffmpeg' in error_msg.lower() or 'merging' in error_msg.lower():
        error_msg += ' (ffmpeg가 설치되어 있지 않거나 PATH에 없습니다.)'
    flash(f'An error occurred: {error_msg}', 'danger')

    return redirect(url_for('index'))

@app.route('/edit', methods=['POST'])
def edit():
    filename = request.form.get('filename')
    start_time = request.form.get('start_time')
    end_time = request.form.get('end_time')

    if not all([filename, start_time, end_time]):
        flash('All fields are required for editing!', 'danger')
        return redirect(url_for('index'))
    
    input_path = os.path.join(app.config['DOWNLOAD_FOLDER'], filename)
    
    name, ext = os.path.splitext(filename)
    output_filename = f"{name}_trimmed{ext}"
    output_path = os.path.join(app.config['DOWNLOAD_FOLDER'], output_filename)

    command = [
        'ffmpeg',
        '-y',
        '-i', input_path,
        '-ss', start_time,
        '-to', end_time,
        '-c', 'copy',
        output_path
    ]

    try:
        subprocess.run(command, check=True, capture_output=True)
        flash(f'Successfully trimmed video to {output_filename}!', 'success')
    except subprocess.CalledProcessError as e:
        error_message = e.stderr.decode('utf-8', errors='ignore') if e.stderr else str(e)
        flash(f'Failed to trim video: {error_message}', 'danger')
    except FileNotFoundError:
        flash('Error: ffmpeg is not installed or not in your PATH.', 'danger')

    return redirect(url_for('index'))

@app.route('/add_tts_to_video', methods=['POST'])
def add_tts_to_video():
    """동영상에 TTS 음성 추가/교체"""
    video_filename = request.form.get('video_filename')
    tts_filename = request.form.get('tts_filename')
    audio_mode = request.form.get('audio_mode', 'replace')  # 'replace' 또는 'add'
    output_filename = request.form.get('output_filename', '').strip()
    
    if not video_filename or not tts_filename:
        flash('동영상 파일과 TTS 파일을 모두 선택해주세요.', 'danger')
        return redirect(url_for('index'))
    
    video_path = os.path.join(app.config['DOWNLOAD_FOLDER'], video_filename)
    tts_path = os.path.join(app.config['DOWNLOAD_FOLDER'], tts_filename)
    
    if not os.path.exists(video_path):
        flash(f'동영상 파일을 찾을 수 없습니다: {video_filename}', 'danger')
        return redirect(url_for('index'))
    
    if not os.path.exists(tts_path):
        flash(f'TTS 파일을 찾을 수 없습니다: {tts_filename}', 'danger')
        return redirect(url_for('index'))
    
    # 출력 파일명 생성 (사용자 지정 또는 기본값)
    if not output_filename:
        # 기본값: 원본 파일명_with_tts
        name, ext = os.path.splitext(video_filename)
        output_filename = f"{name}_with_tts{ext}"
    else:
        # 사용자가 지정한 파일명 사용
        # 확장자가 없으면 원본 동영상의 확장자 추가
        if '.' not in output_filename:
            _, ext = os.path.splitext(video_filename)
            output_filename = output_filename + ext
        # 안전한 파일명으로 변환
        output_filename = secure_filename(output_filename)
    
    output_path = os.path.join(app.config['DOWNLOAD_FOLDER'], output_filename)
    
    try:
        # 동영상에 오디오 스트림이 있는지 확인
        check_audio_cmd = [
            'ffmpeg',
            '-i', video_path,
            '-hide_banner'
        ]
        check_result = subprocess.run(check_audio_cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        # ffmpeg는 정보를 stderr로 출력하므로 stderr 확인
        # stderr가 None일 수 있으므로 안전하게 처리
        stderr_output = check_result.stderr or ''
        stdout_output = check_result.stdout or ''
        has_audio = 'Audio:' in stderr_output or 'Stream #0:1' in stderr_output or 'Audio:' in stdout_output
        
        if audio_mode == 'replace':
            # 기존 오디오를 TTS로 교체 (오디오가 없어도 TTS 추가)
            command = [
                'ffmpeg',
                '-y',
                '-i', video_path,
                '-i', tts_path,
                '-c:v', 'copy',  # 비디오 코덱 복사
                '-c:a', 'aac',   # 오디오 코덱 변환
                '-map', '0:v:0',  # 첫 번째 입력의 비디오 스트림
                '-map', '1:a:0',  # 두 번째 입력의 오디오 스트림
                '-shortest',      # 가장 짧은 스트림에 맞춤
                output_path
            ]
        else:  # add (믹스)
            if has_audio:
                # 기존 오디오와 TTS를 믹스
                command = [
                    'ffmpeg',
                    '-y',
                    '-i', video_path,
                    '-i', tts_path,
                    '-filter_complex', '[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=2[a]',
                    '-map', '0:v:0',
                    '-map', '[a]',
                    '-c:v', 'copy',
                    '-c:a', 'aac',
                    output_path
                ]
            else:
                # 오디오가 없으면 TTS만 추가
                flash('동영상에 오디오가 없어서 TTS만 추가합니다.', 'info')
                command = [
                    'ffmpeg',
                    '-y',
                    '-i', video_path,
                    '-i', tts_path,
                    '-c:v', 'copy',
                    '-c:a', 'aac',
                    '-map', '0:v:0',
                    '-map', '1:a:0',
                    '-shortest',
                    output_path
                ]
        
        result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        flash(f'TTS 음성이 적용되었습니다: {output_filename}', 'success')
        
    except subprocess.CalledProcessError as e:
        error_message = e.stderr.decode('utf-8', errors='ignore') if e.stderr else str(e)
        print(f"ffmpeg 오류: {error_message}")
        flash(f'TTS 적용 실패: {error_message[:200]}', 'danger')
    except FileNotFoundError:
        flash('오류: ffmpeg가 설치되어 있지 않거나 PATH에 없습니다.', 'danger')
    except Exception as e:
        flash(f'TTS 적용 중 오류 발생: {str(e)}', 'danger')

    return redirect(url_for('index'))

@app.route('/extract_frame', methods=['POST'])
def extract_frame():
    """동영상에서 특정 시간의 프레임을 이미지로 추출"""
    filename = request.form.get('filename')
    frame_time = request.form.get('frame_time')
    
    if not filename or not frame_time:
        flash('파일명과 시간을 입력하세요.', 'danger')
        return redirect(url_for('index'))
    
    input_path = os.path.join(app.config['DOWNLOAD_FOLDER'], filename)
    
    if not os.path.exists(input_path):
        flash('파일을 찾을 수 없습니다.', 'danger')
        return redirect(url_for('index'))
    
    name, ext = os.path.splitext(filename)
    time_str = frame_time.replace(':', '-')
    output_filename = f"{name}_frame_{time_str}.jpg"
    output_path = os.path.join(app.config['DOWNLOAD_FOLDER'], output_filename)
    
    command = [
        'ffmpeg', '-y', '-ss', frame_time, '-i', input_path,
        '-vframes', '1', '-q:v', '2', output_path
    ]
    
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        flash(f'프레임 추출 완료: {output_filename}', 'success')
    except subprocess.CalledProcessError as e:
        error_message = e.stderr.decode('utf-8', errors='ignore') if e.stderr else str(e)
        flash(f'프레임 추출 실패: {error_message[:200]}', 'danger')
    except FileNotFoundError:
        flash('Error: ffmpeg is not installed or not in your PATH.', 'danger')
    
    return redirect(url_for('index'))

@app.route('/downloads/<path:filename>')
def download_file(filename):
    return send_from_directory(app.config['DOWNLOAD_FOLDER'], filename, as_attachment=True)

@app.route('/tts', methods=['POST'])
def text_to_speech():
    """텍스트를 음성으로 변환 (한국어/영어 자동 감지)"""
    try:
        text = request.form.get('text', '').strip()
        language = request.form.get('language', 'auto')  # 'auto', 'ko', 'en'
        
        if not text:
            flash('텍스트를 입력해주세요.', 'danger')
        return redirect(url_for('index'))
    
        # 언어 자동 감지
        if language == 'auto':
            detected_lang = detect_language(text)
        else:
            detected_lang = language
        
        # 한국어 TTS 시도
        if detected_lang == 'ko':
            try:
                # gTTS 사용 (간단하고 한국어 지원 좋음)
                from gtts import gTTS
                
                # TTS 저장 폴더 설정
                tts_folder = os.path.join(DOWNLOAD_FOLDER, 'tts')
                os.makedirs(tts_folder, exist_ok=True)  # 폴더가 없으면 생성
                
                filename = f"tts_ko_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
                output_path = os.path.join(tts_folder, filename)
                
                tts = gTTS(text=text, lang='ko', slow=False)
                tts.save(output_path)
                
                flash(f'한국어 TTS 생성 완료: {filename}', 'success')
                return redirect(url_for('index'))
            except ImportError:
                flash('한국어 TTS를 사용하려면 gTTS를 설치하세요: pip install gtts', 'warning')
            except Exception as e:
                print(f"한국어 TTS 오류: {e}, 영어 모델로 폴백")
                detected_lang = 'en'
        
        # 영어 TTS (Bark만 사용)
        if detected_lang == 'en':
            processor, model = load_tts_model()
            
            if processor is None or model is None:
                flash('Bark TTS 모델을 로드할 수 없습니다. 모델이 설치되어 있는지 확인해주세요.', 'danger')
                return redirect(url_for('index'))
            
            if tts_model_type != 'bark':
                flash('TTS 모델이 올바르게 로드되지 않았습니다.', 'danger')
                return redirect(url_for('index'))
            
            try:
                print("Bark 모델로 음성 생성 중...")
                use_cuda = torch.cuda.is_available()
                device = "cuda" if use_cuda else "cpu"
                
                # Bark는 긴 텍스트를 자동으로 처리하므로 분할 불필요
                # 하지만 너무 길면 분할
                if len(text.split()) > 150:
                    text_chunks = split_text_for_tts(text, max_length=150)
                else:
                    text_chunks = [text]
                
                speech_chunks = []
                sample_rate = 24000  # Bark는 24kHz 샘플레이트
                
                for i, chunk in enumerate(text_chunks):
                    try:
                        print(f"청크 {i+1}/{len(text_chunks)} 처리 중: {chunk[:50]}...")
                        
                        # Bark 입력 처리
                        inputs = processor(chunk, return_tensors="pt")
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        
                        with torch.no_grad():
                            # Bark는 generate 메서드로 오디오 생성
                            # voice_preset은 선택사항 (더 자연스러운 음성)
                            try:
                                audio_array = model.generate(**inputs, pad_token_id=10000, voice_preset="v2/en_speaker_6")
                            except:
                                # voice_preset이 지원되지 않으면 기본 생성
                                audio_array = model.generate(**inputs, pad_token_id=10000)
                        
                        # Bark 출력 처리 (오디오 배열 추출)
                        if isinstance(audio_array, torch.Tensor):
                            audio_array = audio_array.cpu().numpy()
                        
                        # 다차원 배열인 경우 평탄화
                        if audio_array.ndim > 1:
                            # Bark는 보통 (batch, samples) 형태
                            if audio_array.shape[0] == 1:
                                audio_array = audio_array[0]
                            else:
                                # 여러 샘플이 있으면 첫 번째 사용
                                audio_array = audio_array.flatten()
                        
                        # 정규화 (-1.0 ~ 1.0 범위로)
                        if audio_array.dtype != np.float32:
                            audio_array = audio_array.astype(np.float32)
                        
                        # 값 범위 확인 및 정규화
                        max_val = np.abs(audio_array).max()
                        if max_val > 0:
                            audio_array = audio_array / max_val
                        
                        audio_array = np.clip(audio_array, -1.0, 1.0)
                        
                        # 청크 간 침묵 추가
                        silence = np.zeros(int(sample_rate * 0.3))
                        speech_chunks.append(audio_array)
                        if i < len(text_chunks) - 1:
                            speech_chunks.append(silence)
                        
                    except Exception as e:
                        print(f"Bark 청크 {i+1} 처리 오류: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                
                if not speech_chunks:
                    flash('TTS 생성 실패: 모든 청크 처리에 실패했습니다.', 'danger')
                    return redirect(url_for('index'))
                
                # 모든 청크 합치기
                speech = np.concatenate(speech_chunks)
                speech = np.clip(speech, -1.0, 1.0)
                
                # TTS 저장 폴더 설정
                tts_folder = os.path.join(DOWNLOAD_FOLDER, 'tts')
                os.makedirs(tts_folder, exist_ok=True)  # 폴더가 없으면 생성
                
                filename = f"tts_en_bark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
                output_path = os.path.join(tts_folder, filename)
                sf.write(output_path, speech, sample_rate)
                
                flash(f'영어 TTS 생성 완료 (Bark): {filename}', 'success')
                return redirect(url_for('index'))
                
            except Exception as e:
                error_msg = str(e)
                print(f"Bark 모델 오류: {error_msg}")
                import traceback
                traceback.print_exc()
                flash(f'TTS 생성 실패: {error_msg[:200]}', 'danger')
                return redirect(url_for('index'))

    except Exception as e:
        error_msg = str(e)
        print(f"TTS 생성 오류: {error_msg}")
        import traceback
        traceback.print_exc()
        flash(f'TTS 생성 실패: {error_msg[:200]}', 'danger')
        return redirect(url_for('index'))

@app.route('/analyze_video', methods=['POST'])
def analyze_video():
    """동영상 내용 분석 및 요약 - 항상 JSON 반환"""
    filename = request.form.get('filename')
    
    if not filename:
        return jsonify({'success': False, 'error': '동영상 파일을 선택해주세요.'}), 400
    
    video_path = os.path.join(app.config['DOWNLOAD_FOLDER'], filename)
    
    if not os.path.exists(video_path):
        return jsonify({'success': False, 'error': '동영상 파일을 찾을 수 없습니다.'}), 404
    
    try:
        # 동영상 분석
        results = analyze_video_content(video_path)
        
        if results['error']:
            return jsonify({
                'success': False,
                'error': f'분석 중 오류 발생: {results["error"]}'
            }), 500
        
        # 결과를 JSON 파일로 저장 (downloads/analysis 폴더에 저장)
        analysis_folder = os.path.join(app.config['DOWNLOAD_FOLDER'], 'analysis')
        os.makedirs(analysis_folder, exist_ok=True)  # 폴더가 없으면 생성
        
        result_filename = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        result_path = os.path.join(analysis_folder, result_filename)
        
        # 프레임 경로를 상대 경로로 변환
        result_data = {
            'video_filename': filename,
            'duration': results['duration'],
            'transcription': results['transcription'],
            'transcription_summary': results.get('transcription_summary', ''),
            'summary': results['summary'],
            'key_frames': [os.path.basename(f) for f in results['key_frames']],
            'frame_descriptions': results.get('frame_descriptions', []),
            'analyzed_at': datetime.now().isoformat()
        }
        
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        # 임시 프레임 파일들을 analysis 폴더로 이동
        temp_dir = os.path.join(app.config['DOWNLOAD_FOLDER'], 'temp_frames')
        if os.path.exists(temp_dir):
            for frame_file in results['key_frames']:
                if os.path.exists(frame_file):
                    frame_name = os.path.basename(frame_file)
                    new_frame_path = os.path.join(analysis_folder, frame_name)
                    try:
                        import shutil
                        shutil.move(frame_file, new_frame_path)
                    except:
                        pass
        
        # 항상 JSON 반환
        return jsonify({
            'success': True,
            'result': result_data,
            'result_file': result_filename
        })
        
    except Exception as e:
        error_msg = str(e)
        print(f"동영상 분석 오류: {error_msg}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': f'동영상 분석 실패: {error_msg[:200]}'
        }), 500

@app.route('/video_analysis/<path:filename>')
def get_video_analysis(filename):
    """저장된 분석 결과 조회"""
    result_path = os.path.join(app.config['DOWNLOAD_FOLDER'], filename)
    
    if not os.path.exists(result_path):
        return jsonify({'error': '분석 결과를 찾을 수 없습니다.'}), 404
    
    try:
        with open(result_path, 'r', encoding='utf-8') as f:
            result_data = json.load(f)
        return jsonify(result_data)
    except Exception as e:
        return jsonify({'error': f'결과 읽기 오류: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True) 