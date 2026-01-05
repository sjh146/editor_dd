import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import matplotlib
matplotlib.use('Agg')
import subprocess
import yt_dlp
from flask import Flask, flash, redirect, render_template, request, send_from_directory, url_for, jsonify
from typing import cast
from werkzeug.utils import secure_filename
import json
import re
from datetime import datetime

# Audio processing imports
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torchaudio
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

app = Flask(__name__)
app.config['SECRET_KEY'] = 'a_super_secret_key'
DOWNLOAD_FOLDER = 'downloads'
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER
KNOWLEDGE_BASE_FILE = 'chatbot_knowledge.json'
LEARNING_DATA_FILE = 'chatbot_learning_data.json'

if not os.path.exists(DOWNLOAD_FOLDER):
    os.makedirs(DOWNLOAD_FOLDER)

# 학습 데이터 및 지식 베이스 로딩
def load_knowledge_base():
    """지식 베이스 로딩"""
    if os.path.exists(KNOWLEDGE_BASE_FILE):
        try:
            with open(KNOWLEDGE_BASE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {'qa_pairs': [], 'examples': []}
    return {'qa_pairs': [], 'examples': []}

def save_knowledge_base(kb_data):
    """지식 베이스 저장"""
    with open(KNOWLEDGE_BASE_FILE, 'w', encoding='utf-8') as f:
        json.dump(kb_data, f, ensure_ascii=False, indent=2)

def load_learning_data():
    """학습 데이터 로딩"""
    if os.path.exists(LEARNING_DATA_FILE):
        try:
            with open(LEARNING_DATA_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {'conversations': [], 'feedback': []}
    return {'conversations': [], 'feedback': []}

def save_learning_data(learning_data):
    """학습 데이터 저장"""
    with open(LEARNING_DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(learning_data, f, ensure_ascii=False, indent=2)

# 전역 변수로 지식 베이스와 학습 데이터 로드
knowledge_base = load_knowledge_base()
learning_data = load_learning_data()

def check_ffmpeg_available():
    """ffmpeg가 설치되어 있는지 확인"""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False

# 챗봇 모델 로딩 (최초 1회만)
chatbot_tokenizer = None
chatbot_model = None

# TTS 모델 로딩 (최초 1회만)
tts_processor = None
tts_model = None
tts_vocoder = None
tts_model_type = None  # 'speecht5' or 'bark'
korean_tts_available = False

# 한국어 TTS 라이브러리 확인
try:
    from TTS.api import TTS
    korean_tts_available = True
except ImportError:
    try:
        import gtts
        korean_tts_available = True
    except ImportError:
        korean_tts_available = False

def load_chatbot_model():
    """한국어 챗봇 모델 로딩"""
    global chatbot_tokenizer, chatbot_model
    if chatbot_tokenizer is None or chatbot_model is None:
        model_candidates = ["skt/kogpt2-base-v2", "gpt2"]
        
        for model_name in model_candidates:
            try:
                chatbot_tokenizer = AutoTokenizer.from_pretrained(model_name)
                use_cuda = torch.cuda.is_available()
                
                if use_cuda:
                    try:
                        chatbot_model = AutoModelForCausalLM.from_pretrained(
                            model_name, torch_dtype=torch.float16,
                            device_map="auto", low_cpu_mem_usage=True
                        )
                    except Exception:
                        chatbot_model = AutoModelForCausalLM.from_pretrained(
                            model_name, torch_dtype=torch.float16
                        )
                        chatbot_model = chatbot_model.to("cuda").eval()
                else:
                    chatbot_model = AutoModelForCausalLM.from_pretrained(
                        model_name, torch_dtype=torch.float32
                    ).to("cpu").eval()
                
                if chatbot_tokenizer.pad_token is None:
                    chatbot_tokenizer.pad_token = chatbot_tokenizer.eos_token
                
                print(f"챗봇 모델 로딩 완료: {model_name} ({'GPU' if use_cuda else 'CPU'} 모드)")
                break
            except Exception as e:
                print(f"모델 {model_name} 로딩 실패: {e}")
                chatbot_tokenizer = None
                chatbot_model = None
    
    return chatbot_tokenizer, chatbot_model

def load_tts_model():
    """TTS 모델 로딩 (영어용 - Bark만 사용)"""
    global tts_processor, tts_model, tts_vocoder, tts_model_type
    
    if tts_model_type == 'bark' and tts_model is not None:
        return tts_processor, tts_model, tts_vocoder
    
    # 기존 모델 정리
    tts_processor = None
    tts_model = None
    tts_vocoder = None
    
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    
    try:
        print("Bark TTS 모델 로딩 중... (고품질)")
        from transformers import BarkModel, AutoProcessor
        
        model_name = "suno/bark-small"  # 또는 "suno/bark" (더 큰 모델, 더 좋은 품질)
        tts_processor = AutoProcessor.from_pretrained(model_name)
        tts_model = BarkModel.from_pretrained(model_name)
        
        if use_cuda:
            tts_model = tts_model.to(device)
        else:
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
        tts_vocoder = None
        tts_model_type = None
    
    return tts_processor, tts_model, tts_vocoder

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

@app.route('/')
def index():
    files = get_downloaded_files()
    tts_files = get_tts_files()
    return render_template('index.html', files=files, tts_files=tts_files)

@app.route('/download', methods=['POST'])
def download():
    url = request.form.get('url')
    if not url:
        flash('URL is required!', 'danger')
        return redirect(url_for('index'))

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
                'outtmpl': os.path.join(app.config['DOWNLOAD_FOLDER'], '%(title)s.%(ext)s'),
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
        error_message = e.stderr.decode()
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
        check_result = subprocess.run(check_audio_cmd, capture_output=True, text=True)
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
        
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        flash(f'TTS 음성이 적용되었습니다: {output_filename}', 'success')
        
    except subprocess.CalledProcessError as e:
        error_message = e.stderr.decode() if e.stderr else str(e)
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
        subprocess.run(command, check=True, capture_output=True, text=True)
        flash(f'프레임 추출 완료: {output_filename}', 'success')
    except subprocess.CalledProcessError as e:
        error_message = e.stderr.decode() if e.stderr else str(e)
        flash(f'프레임 추출 실패: {error_message[:200]}', 'danger')
    except FileNotFoundError:
        flash('Error: ffmpeg is not installed or not in your PATH.', 'danger')
    
    return redirect(url_for('index'))

@app.route('/downloads/<path:filename>')
def download_file(filename):
    return send_from_directory(app.config['DOWNLOAD_FOLDER'], filename, as_attachment=True)

# 설문조사 자동화 관련 임포트
try:
    from survey_automation import SurveyAutoFill, SurveyAnalyzer
    SURVEY_AUTOMATION_AVAILABLE = True
except ImportError:
    SURVEY_AUTOMATION_AVAILABLE = False

@app.route('/survey/analyze', methods=['POST'])
def analyze_survey():
    """설문조사 웹페이지를 분석합니다"""
    if not SURVEY_AUTOMATION_AVAILABLE:
        return jsonify({'error': '설문조사 자동화 모듈을 사용할 수 없습니다. 필요한 라이브러리를 설치하세요.'}), 500
    
    url = request.json.get('url') if request.is_json else request.form.get('url')
    if not url:
        return jsonify({'error': 'URL이 필요합니다.'}), 400
    
    try:
        analyzer = SurveyAnalyzer(url)
        questions = analyzer.analyze()
        
        if not questions:
            return jsonify({'error': '설문조사를 찾을 수 없습니다. 페이지 구조를 확인하세요.'}), 404
        
        return jsonify({
            'success': True,
            'url': url,
            'questions': questions,
            'total_questions': len(questions)
        })
    except Exception as e:
        import traceback
        return jsonify({'error': f'분석 중 오류 발생: {str(e)}\n{traceback.format_exc()}'}), 500

@app.route('/survey/fill', methods=['POST'])
def fill_survey():
    """설문조사를 자동으로 작성합니다"""
    if not SURVEY_AUTOMATION_AVAILABLE:
        return jsonify({'error': '설문조사 자동화 모듈을 사용할 수 없습니다.'}), 500
    
    data = request.json if request.is_json else request.form.to_dict()
    url = data.get('url')
    
    if not url:
        return jsonify({'error': 'URL이 필요합니다.'}), 400
    
    # boolean 값을 안전하게 처리
    def to_bool(value, default=False):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() == 'true'
        return default
    
    headless = to_bool(data.get('headless'), True)
    paginated = to_bool(data.get('paginated'), False)
    use_api = to_bool(data.get('use_api'), False)
    api_provider = data.get('api_provider') if use_api else None
    api_key = data.get('api_key') if use_api else None
    
    automation = None
    try:
        automation = SurveyAutoFill(url=url, headless=headless, api_provider=api_provider, api_key=api_key)
        
        success = automation.fill_survey(paginated=paginated)
        
        if not success:
            if automation and automation.driver:
                automation.driver.quit()
            return jsonify({'error': '설문조사 작성 중 오류가 발생했습니다. 서버 콘솔을 확인하세요.'}), 500
        
        return jsonify({
            'success': True,
            'url': url,
            'mode': 'paginated' if paginated else 'single_page'
        })
    except Exception as e:
        # automation 정리
        if automation and automation.driver:
            try:
                automation.driver.quit()
            except:
                pass
        
        # 에러 메시지 안전하게 처리
        error_msg = str(e).replace('"', "'").replace('\n', ' ').replace('\r', ' ')[:500]
        return jsonify({
            'error': f'오류 발생: {error_msg}',
            'type': type(e).__name__
        }), 500

def handle_file_operations(user_message: str) -> tuple[str, bool]:
    """파일 작업 처리 (읽기, 쓰기, 생성 등)"""
    user_lower = user_message.lower()
    
    # 파일 읽기 요청
    if any(keyword in user_lower for keyword in ['파일 읽', '파일 보', '파일 내용', 'read file', 'show file', 'file content']):
        # 파일명 추출 시도
        import re
        file_patterns = [
            r'["\']([^"\']+\.[a-zA-Z]+)["\']',
            r'([a-zA-Z0-9_\-]+\.(py|txt|js|html|css|json|md|yml|yaml))',
        ]
        for pattern in file_patterns:
            match = re.search(pattern, user_message)
            if match:
                filename = match.group(1)
                try:
                    if os.path.exists(filename):
                        with open(filename, 'r', encoding='utf-8') as f:
                            content = f.read()
                        return f"파일 '{filename}' 내용:\n\n```\n{content[:1000]}{'...' if len(content) > 1000 else ''}\n```", True
                    else:
                        return f"파일 '{filename}'을 찾을 수 없습니다.", True
                except Exception as e:
                    return f"파일 읽기 오류: {str(e)}", True
    
    # 파일 생성/쓰기 요청
    if any(keyword in user_lower for keyword in ['파일 만들', '파일 생성', '파일 쓰', 'create file', 'write file', 'make file']):
        return "파일 생성/수정 기능을 사용하려면 파일명과 내용을 명확히 알려주세요.\n예: 'test.py 파일을 만들고 print(\"hello\") 코드를 작성해줘'", True
    
    return "", False

def search_knowledge_base(query: str) -> list:
    """지식 베이스에서 관련 정보 검색"""
    global knowledge_base
    results = []
    query_lower = query.lower()
    
    for qa in knowledge_base.get('qa_pairs', []):
        question = qa.get('question', '').lower()
        if any(word in question for word in query_lower.split()):
            results.append(qa)
    
    return results[:3]  # 상위 3개만 반환

def get_chatbot_response(user_message: str, conversation_history: list = None) -> str:
    """로컬 Transformer 모델을 사용한 챗봇 응답 생성"""
    global knowledge_base, learning_data
    
    if conversation_history is None:
        conversation_history = []
    
    # 파일 작업 처리 시도
    file_response, handled = handle_file_operations(user_message)
    if handled:
        return file_response
    
    # 지식 베이스 검색
    kb_results = search_knowledge_base(user_message)
    kb_context = ""
    if kb_results:
        kb_context = "\n\n학습된 지식:\n"
        for qa in kb_results:
            kb_context += f"Q: {qa.get('question')}\nA: {qa.get('answer')}\n\n"
    
    # Few-shot 예제 추가
    examples = ""
    if knowledge_base.get('examples'):
        examples = "\n\n예제 대화:\n"
        for ex in knowledge_base.get('examples', [])[:2]:
            examples += f"사용자: {ex.get('user')}\n조수: {ex.get('assistant')}\n\n"
    
    # 시스템 프롬프트: 편집 작업을 도와주는 조수 역할
    system_prompt = f"""당신은 편집 작업을 도와주는 친절한 한국어 조수입니다. 
다음 기능들을 제공할 수 있습니다:
1. 코드 작성 및 수정: Python, JavaScript, HTML, CSS 등 다양한 언어의 코드 작성
2. 파일 편집: 파일 읽기, 생성, 수정, 삭제
3. 문제 해결: 에러 분석, 버그 수정, 코드 최적화
4. 코드 리뷰: 코드 품질 개선, 스타일 가이드 준수
5. 문서화: 주석 추가, README 작성

사용자가 구체적인 요청을 하면 실용적이고 실행 가능한 코드나 해결책을 제공하세요.
항상 한국어로 친절하고 명확하게 답변하세요.{kb_context}{examples}"""
    
    try:
        # 로컬 Transformer 모델 로딩
        tokenizer, model = load_chatbot_model()
        
        if tokenizer is None or model is None:
            return "죄송합니다. 챗봇 모델을 로드할 수 없습니다. 모델 다운로드 중 오류가 발생했을 수 있습니다."
        
        prompt_parts = [f"편집 조수: {system_prompt}\n"]
        for msg in conversation_history[-3:]:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'user':
                prompt_parts.append(f"사용자: {content}\n")
            elif role == 'assistant':
                prompt_parts.append(f"조수: {content}\n")
        prompt_parts.append(f"사용자: {user_message}\n조수:")
        full_prompt = "".join(prompt_parts)
        
        # 토큰화
        inputs = tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if not next(model.parameters()).is_cuda if device == "cuda" else not next(model.parameters()).is_cpu:
            model = model.to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 생성
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2
            )
        
        # 디코딩
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "조수:" in generated_text:
            response = generated_text.split("조수:")[-1].strip()
            if "\n사용자:" in response:
                response = response.split("\n사용자:")[0].strip()
            if "\n" in response:
                response = response.split("\n")[0].strip()
        else:
            response = generated_text[len(full_prompt):].strip()
            if "\n" in response:
                response = response.split("\n")[0].strip()
        
        if not response or len(response) < 3:
            response = "죄송합니다. 적절한 응답을 생성하지 못했습니다. 다시 질문해주세요."
        if len(response) > 500:
            response = response[:500] + "..."
        
        # 학습 데이터에 대화 저장
        learning_data['conversations'].append({
            'user': user_message,
            'assistant': response,
            'timestamp': datetime.now().isoformat()
        })
        if len(learning_data['conversations']) > 1000:
            learning_data['conversations'] = learning_data['conversations'][-1000:]
        save_learning_data(learning_data)
        
        return response
        
    except Exception as e:
        error_msg = str(e)
        print(f"챗봇 응답 생성 오류: {error_msg}")
        user_lower = user_message.lower()
        
        if any(word in user_lower for word in ['안녕', 'hello', 'hi', '하이']):
            return "안녕하세요! 편집 작업을 도와드리는 조수입니다. 무엇을 도와드릴까요?"
        elif any(word in user_lower for word in ['파일', 'file', '코드', 'code']):
            return "파일이나 코드 작업을 도와드릴 수 있습니다. 구체적으로 어떤 작업이 필요하신가요?"
        elif any(word in user_lower for word in ['도움', 'help', '도와']):
            return "다음과 같은 작업을 도와드릴 수 있습니다:\n- 코드 작성 및 수정\n- 파일 편집\n- 문제 해결\n- 문법 및 스타일 개선\n\n구체적으로 무엇을 도와드릴까요?"
        elif any(word in user_lower for word in ['감사', '고마', 'thanks', 'thank']):
            return "천만에요! 다른 도움이 필요하시면 언제든지 말씀해주세요."
        else:
            return f"죄송합니다. 모델 처리 중 오류가 발생했습니다: {error_msg[:100]}. 다시 시도해주세요."

@app.route('/chatbot', methods=['POST'])
def chatbot():
    """챗봇 API 엔드포인트"""
    try:
        data = request.json if request.is_json else request.form.to_dict()
        user_message = data.get('message', '').strip()
        conversation_history = data.get('history', [])
        
        if not user_message:
            return jsonify({'error': '메시지를 입력해주세요.'}), 400
        
        response = get_chatbot_response(user_message, conversation_history)
        
        return jsonify({
            'success': True,
            'response': response
        })
    except Exception as e:
        return jsonify({
            'error': f'오류 발생: {str(e)}'
        }), 500

@app.route('/chatbot/file/read', methods=['POST'])
def chatbot_read_file():
    """파일 읽기 API"""
    try:
        data = request.json if request.is_json else request.form.to_dict()
        filename = data.get('filename', '').strip()
        
        if not filename:
            return jsonify({'error': '파일명을 입력해주세요.'}), 400
        
        if not os.path.exists(filename):
            return jsonify({'error': f'파일을 찾을 수 없습니다: {filename}'}), 404
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
            return jsonify({
                'success': True,
                'filename': filename,
                'content': content,
                'size': len(content)
            })
        except Exception as e:
            return jsonify({'error': f'파일 읽기 오류: {str(e)}'}), 500
            
    except Exception as e:
        return jsonify({'error': f'오류 발생: {str(e)}'}), 500

@app.route('/chatbot/file/write', methods=['POST'])
def chatbot_write_file():
    """파일 쓰기/생성 API"""
    try:
        data = request.json if request.is_json else request.form.to_dict()
        filename = data.get('filename', '').strip()
        content = data.get('content', '')
        mode = data.get('mode', 'w')  # 'w' (쓰기) 또는 'a' (추가)
        
        if not filename:
            return jsonify({'error': '파일명을 입력해주세요.'}), 400
        
        # 보안: 상위 디렉토리 접근 방지
        if '..' in filename or filename.startswith('/'):
            return jsonify({'error': '잘못된 파일 경로입니다.'}), 400
        
        try:
            with open(filename, mode, encoding='utf-8') as f:
                f.write(content)
            return jsonify({
                'success': True,
                'filename': filename,
                'message': f'파일이 {"생성" if mode == "w" else "수정"}되었습니다.',
                'size': len(content)
            })
        except Exception as e:
            return jsonify({'error': f'파일 쓰기 오류: {str(e)}'}), 500
            
    except Exception as e:
        return jsonify({'error': f'오류 발생: {str(e)}'}), 500

@app.route('/chatbot/file/list', methods=['GET', 'POST'])
def chatbot_list_files():
    """현재 디렉토리 파일 목록 조회"""
    try:
        current_dir = request.args.get('dir', '.') or request.json.get('dir', '.') if request.is_json else '.'
        
        if not os.path.exists(current_dir):
            return jsonify({'error': '디렉토리를 찾을 수 없습니다.'}), 404
        
        files = []
        for item in os.listdir(current_dir):
            item_path = os.path.join(current_dir, item)
            files.append({
                'name': item,
                'type': 'directory' if os.path.isdir(item_path) else 'file',
                'size': os.path.getsize(item_path) if os.path.isfile(item_path) else 0
            })
        
        return jsonify({
            'success': True,
            'directory': current_dir,
            'files': sorted(files, key=lambda x: (x['type'] == 'file', x['name']))
        })
    except Exception as e:
        return jsonify({'error': f'오류 발생: {str(e)}'}), 500

@app.route('/chatbot/learn/qa', methods=['POST'])
def chatbot_learn_qa():
    """Q&A 쌍을 지식 베이스에 추가"""
    global knowledge_base
    try:
        data = request.json if request.is_json else request.form.to_dict()
        question = data.get('question', '').strip()
        answer = data.get('answer', '').strip()
        
        if not question or not answer:
            return jsonify({'error': '질문과 답변을 모두 입력해주세요.'}), 400
        
        knowledge_base['qa_pairs'].append({
            'question': question,
            'answer': answer,
            'timestamp': datetime.now().isoformat()
        })
        save_knowledge_base(knowledge_base)
        
        return jsonify({
            'success': True,
            'message': 'Q&A 쌍이 학습되었습니다.',
            'total_qa': len(knowledge_base['qa_pairs'])
        })
    except Exception as e:
        return jsonify({'error': f'오류 발생: {str(e)}'}), 500

@app.route('/chatbot/learn/example', methods=['POST'])
def chatbot_learn_example():
    """예제 대화를 지식 베이스에 추가"""
    global knowledge_base
    try:
        data = request.json if request.is_json else request.form.to_dict()
        user_msg = data.get('user', '').strip()
        assistant_msg = data.get('assistant', '').strip()
        
        if not user_msg or not assistant_msg:
            return jsonify({'error': '사용자 메시지와 조수 응답을 모두 입력해주세요.'}), 400
        
        knowledge_base['examples'].append({
            'user': user_msg,
            'assistant': assistant_msg,
            'timestamp': datetime.now().isoformat()
        })
        save_knowledge_base(knowledge_base)
        
        return jsonify({
            'success': True,
            'message': '예제 대화가 학습되었습니다.',
            'total_examples': len(knowledge_base['examples'])
        })
    except Exception as e:
        return jsonify({'error': f'오류 발생: {str(e)}'}), 500

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
                import io
                
                filename = f"tts_ko_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
                output_path = os.path.join(DOWNLOAD_FOLDER, filename)
                
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
            processor, model, vocoder = load_tts_model()
            
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
                
                filename = f"tts_en_bark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
                output_path = os.path.join(DOWNLOAD_FOLDER, filename)
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

@app.route('/chatbot/learn/feedback', methods=['POST'])
def chatbot_learn_feedback():
    """사용자 피드백 저장 및 학습"""
    global learning_data
    try:
        data = request.json if request.is_json else request.form.to_dict()
        user_message = data.get('user_message', '').strip()
        assistant_response = data.get('assistant_response', '').strip()
        feedback = data.get('feedback', '').strip()  # 'good', 'bad', 'improve'
        improvement = data.get('improvement', '').strip()  # 개선 제안
        
        learning_data['feedback'].append({
            'user_message': user_message,
            'assistant_response': assistant_response,
            'feedback': feedback,
            'improvement': improvement,
            'timestamp': datetime.now().isoformat()
        })
        
        # 피드백이 'bad'이고 개선 제안이 있으면 지식 베이스에 추가
        if feedback == 'bad' and improvement:
            knowledge_base['qa_pairs'].append({
                'question': user_message,
                'answer': improvement,
                'timestamp': datetime.now().isoformat(),
                'source': 'feedback'
            })
            save_knowledge_base(knowledge_base)
        
        save_learning_data(learning_data)
        
        return jsonify({
            'success': True,
            'message': '피드백이 저장되었습니다.'
        })
    except Exception as e:
        return jsonify({'error': f'오류 발생: {str(e)}'}), 500

@app.route('/chatbot/learn/stats', methods=['GET'])
def chatbot_learn_stats():
    """학습 통계 조회"""
    global knowledge_base, learning_data
    return jsonify({
        'success': True,
        'knowledge_base': {
            'qa_pairs': len(knowledge_base.get('qa_pairs', [])),
            'examples': len(knowledge_base.get('examples', []))
        },
        'learning_data': {
            'conversations': len(learning_data.get('conversations', [])),
            'feedback': len(learning_data.get('feedback', []))
        }
    })

@app.route('/chatbot/learn/export', methods=['GET'])
def chatbot_learn_export():
    """학습 데이터 내보내기"""
    global knowledge_base, learning_data
    return jsonify({
        'success': True,
        'knowledge_base': knowledge_base,
        'learning_data': learning_data
    })

@app.route('/chatbot/learn/import', methods=['POST'])
def chatbot_learn_import():
    """학습 데이터 가져오기"""
    global knowledge_base, learning_data
    try:
        data = request.json if request.is_json else request.form.to_dict()
        
        if 'knowledge_base' in data:
            knowledge_base.update(data['knowledge_base'])
            save_knowledge_base(knowledge_base)
        
        if 'learning_data' in data:
            learning_data.update(data['learning_data'])
            save_learning_data(learning_data)
        
        return jsonify({
            'success': True,
            'message': '학습 데이터가 가져와졌습니다.'
        })
    except Exception as e:
        return jsonify({'error': f'오류 발생: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True) 