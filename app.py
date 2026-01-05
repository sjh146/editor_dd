import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import matplotlib
matplotlib.use('Agg')
import subprocess
import yt_dlp
from flask import Flask, flash, redirect, render_template, request, send_from_directory, url_for, jsonify
from typing import cast
from werkzeug.utils import secure_filename

# MusicGen & audio processing imports
from transformers import MusicgenForConditionalGeneration, MusicgenProcessor
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

if not os.path.exists(DOWNLOAD_FOLDER):
    os.makedirs(DOWNLOAD_FOLDER)

def check_ffmpeg_available():
    """ffmpeg가 설치되어 있는지 확인"""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False

# MusicGen 모델 로딩 (최초 1회만)
musicgen_processor = None
musicgen_model = None

def load_musicgen() -> tuple[MusicgenProcessor, MusicgenForConditionalGeneration]:
    global musicgen_processor, musicgen_model
    if musicgen_processor is None or musicgen_model is None:
        musicgen_processor = cast(MusicgenProcessor, MusicgenProcessor.from_pretrained("facebook/musicgen-small"))
        musicgen_model = cast(MusicgenForConditionalGeneration, MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small"))
    return musicgen_processor, musicgen_model

# 챗봇 모델 로딩 (최초 1회만)
chatbot_tokenizer = None
chatbot_model = None

def load_chatbot_model():
    """한국어 챗봇 모델 로딩"""
    global chatbot_tokenizer, chatbot_model
    if chatbot_tokenizer is None or chatbot_model is None:
        # 한국어 지원 모델 사용 (실제로 존재하는 모델들)
        # 우선순위: 작은 모델부터 시도 (빠른 로딩, 적은 메모리)
        model_candidates = [
            "skt/kogpt2-base-v2",  # SKT의 KoGPT2 (작고 빠름, 한국어 지원)
            "gpt2",  # 기본 GPT2 (영어지만 작고 안정적, 폴백용)
        ]
        
        for model_name in model_candidates:
            print(f"챗봇 모델 로딩 시도: {model_name}...")
            try:
                chatbot_tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                # GPU/CPU 설정 - GPU가 있으면 무조건 GPU 사용
                use_cuda = torch.cuda.is_available()
                if use_cuda:
                    # GPU 사용 시 - 무조건 GPU 모드
                    print(f"GPU 감지됨: {torch.cuda.get_device_name(0)}")
                    try:
                        # accelerate가 있으면 최적화 옵션 사용
                        chatbot_model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            torch_dtype=torch.float16,
                            device_map="auto",
                            low_cpu_mem_usage=True
                        )
                        # device_map="auto"를 사용하면 자동으로 GPU에 할당되지만, 명시적으로 확인
                        if hasattr(chatbot_model, 'device'):
                            print(f"모델이 {chatbot_model.device}에 로드됨")
                        else:
                            # device_map을 사용했지만 확인을 위해 첫 번째 파라미터의 device 확인
                            first_param = next(chatbot_model.parameters())
                            print(f"모델이 {first_param.device}에 로드됨")
                    except Exception as e:
                        print(f"accelerate 최적화 실패, 기본 GPU 모드로 전환: {e}")
                        # accelerate가 없으면 기본 방식으로 GPU에 명시적으로 로드
                        chatbot_model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            torch_dtype=torch.float16
                        )
                        chatbot_model = chatbot_model.to("cuda")
                        chatbot_model.eval()
                        print(f"모델이 cuda에 명시적으로 로드됨")
                else:
                    # GPU가 없으면 에러 메시지와 함께 CPU 사용
                    print("경고: GPU를 감지할 수 없습니다. CPU 모드로 작동합니다.")
                    print("GPU를 사용하려면 CUDA가 설치된 GPU가 필요합니다.")
                    chatbot_model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float32
                    )
                    chatbot_model = chatbot_model.to("cpu")
                    chatbot_model.eval()
                
                # 패딩 토큰 설정
                if chatbot_tokenizer.pad_token is None:
                    chatbot_tokenizer.pad_token = chatbot_tokenizer.eos_token
                
                print(f"챗봇 모델 로딩 완료: {model_name} ({'GPU' if use_cuda else 'CPU'} 모드)")
                break  # 성공하면 루프 종료
                
            except Exception as e:
                print(f"모델 {model_name} 로딩 실패: {e}")
                chatbot_tokenizer = None
                chatbot_model = None
                continue  # 다음 모델 시도
        
        if chatbot_tokenizer is None or chatbot_model is None:
            print("모든 챗봇 모델 로딩 실패. 규칙 기반 응답만 사용 가능합니다.")
    
    return chatbot_tokenizer, chatbot_model

def get_downloaded_files():
    return sorted(os.listdir(app.config['DOWNLOAD_FOLDER']), reverse=True)

def get_musicgen_files():
    """MusicGen 관련 파일만 필터링"""
    files = get_downloaded_files()
    return [f for f in files if f.endswith(('.mp3', '.png'))]

@app.route('/')
def index():
    files = get_downloaded_files()
    musicgen_files = get_musicgen_files()
    return render_template('index.html', files=files, musicgen_files=musicgen_files)

@app.route('/download', methods=['POST'])
def download():
    url = request.form.get('url')
    if not url:
        flash('URL is required!', 'danger')
        return redirect(url_for('index'))

    # ffmpeg 사용 가능 여부 확인
    ffmpeg_available = check_ffmpeg_available()
    
    # 형식 옵션 선택 (더 유연한 선택 우선)
    if ffmpeg_available:
        format_options = [
            'bestvideo+bestaudio/best',  # 가장 유연한 형식 먼저
            'best',  # 단일 최고 품질
            'worst',  # 최후의 수단
        ]
    else:
        format_options = [
            'best',  # 이미 병합된 형식만
            'best[height<=720]',
            'best[height<=480]',
            'worst',
        ]
    
    # 다운로드 시도
    last_error = None
    for format_str in format_options:
        if not ffmpeg_available and '+' in format_str:
            continue
        
        try:
            ydl_opts = {
                'outtmpl': os.path.join(app.config['DOWNLOAD_FOLDER'], '%(title)s.%(ext)s'),
                'format': format_str,
                'noplaylist': True,
                'quiet': False,
                'no_warnings': False,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            flash('Download successful!', 'success')
            return redirect(url_for('index'))
        except Exception as e:
            last_error = str(e)
            error_str = str(e).lower()
            # 형식 관련 오류면 다음 형식 시도
            if any(kw in error_str for kw in ['format is not available', 'requested format', 'format']):
                continue
            # ffmpeg/merging 오류면 다음 형식 시도
            if any(kw in error_str for kw in ['ffmpeg', 'merging']):
                continue
            # 네트워크 오류도 재시도
            if any(kw in error_str for kw in ['network', 'timeout', 'unavailable', 'connection']):
                continue
            # 그 외 오류는 중단
            break
    
    # 모든 시도 실패
    error_msg = last_error or 'Unknown error'
    if 'format is not available' in error_msg.lower() or 'requested format' in error_msg.lower():
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
    
    # 이미지 파일명 생성
    name, ext = os.path.splitext(filename)
    # 시간을 파일명에 사용 가능한 형식으로 변환 (콜론 제거)
    time_str = frame_time.replace(':', '-')
    output_filename = f"{name}_frame_{time_str}.jpg"
    output_path = os.path.join(app.config['DOWNLOAD_FOLDER'], output_filename)
    
    # ffmpeg를 사용하여 프레임 추출
    command = [
        'ffmpeg',
        '-y',
        '-ss', frame_time,
        '-i', input_path,
        '-vframes', '1',
        '-q:v', '2',  # 고품질
        output_path
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

@app.route('/musicgen', methods=['POST'])
def musicgen():
    prompt = request.form.get('prompt')
    duration = request.form.get('duration', '30')  # 기본값 30초
    
    if not prompt:
        flash('프롬프트를 입력하세요.', 'danger')
        return redirect(url_for('index'))
    
    # duration을 초 단위로 변환하고 토큰 수 계산
    try:
        duration_seconds = int(duration)
    except ValueError:
        duration_seconds = 30
    
    # MusicGen은 대략 5-10초당 256 토큰 정도 생성 가능
    # 모델의 최대 위치 임베딩을 고려하여 안전한 값으로 제한
    # MusicGen-small 모델의 경우 일반적으로 2048 토큰이 안전한 최대값
    MAX_SAFE_TOKENS = 1536  # 안전한 최대 토큰 수 (약 2분)
    
    if duration_seconds <= 15:
        max_new_tokens = 256
    elif duration_seconds <= 30:
        max_new_tokens = 512
    elif duration_seconds <= 60:
        max_new_tokens = 1024
    elif duration_seconds <= 120:
        max_new_tokens = 1536  # 안전한 최대값
    else:
        max_new_tokens = MAX_SAFE_TOKENS  # 최대값 제한
        flash(f'요청하신 길이({duration_seconds}초)가 너무 깁니다. 최대 {MAX_SAFE_TOKENS} 토큰(약 2분)으로 제한됩니다.', 'warning')
    
    processor, model = load_musicgen()
    # MusicGen inference
    inputs = processor(text=[prompt], padding=True, return_tensors="pt")
    
    # 모델의 최대 위치 임베딩 길이 확인
    max_position_embeddings = getattr(model.config, 'max_position_embeddings', None)
    if max_position_embeddings is None:
        # 기본값 사용 (MusicGen-small의 경우 일반적으로 2048)
        max_position_embeddings = 2048
    
    # 입력 시퀀스 길이 확인 및 제한
    input_ids = inputs.get('input_ids', None)
    if input_ids is not None:
        input_length = input_ids.shape[1]
        # 입력 길이 + 생성할 토큰 수가 최대값을 초과하지 않도록 조정
        if input_length + max_new_tokens > max_position_embeddings:
            max_new_tokens = max(256, max_position_embeddings - input_length - 50)  # 여유 공간 확보
            flash(f'입력 길이를 고려하여 max_new_tokens를 {max_new_tokens}로 조정했습니다.', 'info')
    
    try:
        with torch.no_grad():
            audio_values = model.generate(**inputs, max_new_tokens=max_new_tokens)
    except IndexError as e:
        # IndexError 발생 시 더 작은 값으로 재시도
        flash(f'토큰 수가 너무 큽니다. 더 작은 값으로 재시도합니다.', 'warning')
        max_new_tokens = min(512, max_new_tokens // 2)
        if input_ids is not None:
            input_length = input_ids.shape[1]
            if input_length + max_new_tokens > max_position_embeddings:
                max_new_tokens = max(256, max_position_embeddings - input_length - 50)
        with torch.no_grad():
            audio_values = model.generate(**inputs, max_new_tokens=max_new_tokens)
    # 오디오 저장 (MP3 또는 WAV)
    mp3_filename = f"musicgen_{prompt.replace(' ', '_')}.mp3"
    mp3_path = os.path.join(DOWNLOAD_FOLDER, mp3_filename)
    
    # torchaudio로 저장 시도, 실패하면 soundfile 사용
    audio_data = audio_values[0].cpu().numpy()
    if audio_data.ndim > 1:
        audio_data = audio_data.squeeze()
    
    # 오디오 정규화: 클리핑 방지를 위해 진폭을 -1.0~1.0 범위로 제한
    # 약간의 헤드룸(-0.95~0.95)을 남겨서 안전하게 저장
    max_amplitude = np.max(np.abs(audio_data))
    if max_amplitude > 0:
        # 최대 진폭이 0.95를 넘지 않도록 스케일링
        target_max = 0.95
        if max_amplitude > target_max:
            audio_data = audio_data * (target_max / max_amplitude)
    # 추가 안전장치: -1.0~1.0 범위를 벗어나는 값은 클리핑
    audio_data = np.clip(audio_data, -0.95, 0.95)
    
    try:
        # soundfile로 WAV 저장 후 필요시 MP3로 변환 (더 안정적)
        wav_path = mp3_path.replace('.mp3', '.wav')
        if SOUNDFILE_AVAILABLE:
            sf.write(wav_path, audio_data, 32000)
            # pydub를 사용하여 WAV를 MP3로 변환
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_wav(wav_path)
                audio.export(mp3_path, format="mp3")
                os.remove(wav_path)  # 임시 WAV 파일 삭제
            except:
                # MP3 변환 실패 시 WAV 파일명 사용
                mp3_filename = wav_path
                mp3_path = wav_path
        else:
            # soundfile 없으면 torchaudio로 WAV 저장 시도 (정규화된 데이터 사용)
            audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)
            torchaudio.save(wav_path, audio_tensor, 32000, format="wav")
            mp3_filename = wav_path
            mp3_path = wav_path
    except Exception as e:
        # 모든 저장 방법 실패 시 torchaudio WAV로 fallback (정규화된 데이터 사용)
        try:
            wav_path = mp3_path.replace('.mp3', '.wav')
            audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)
            torchaudio.save(wav_path, audio_tensor, 32000, format="wav")
            mp3_filename = wav_path
            mp3_path = wav_path
        except:
            flash(f'오디오 저장 실패: {str(e)}', 'danger')
            return redirect(url_for('index'))
    # Waveform 이미지 생성 (WAV나 MP3 모두 로드 가능)
    y, sr = librosa.load(mp3_path, sr=None)
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform')
    waveform_path = os.path.join(DOWNLOAD_FOLDER, f"{mp3_filename}_waveform.png")
    plt.savefig(waveform_path)
    plt.close()
    # Spectrogram 이미지 생성
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    plt.figure(figsize=(10, 3))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    spectrogram_path = os.path.join(DOWNLOAD_FOLDER, f"{mp3_filename}_spectrogram.png")
    plt.savefig(spectrogram_path)
    plt.close()
    flash(f'MusicGen 결과가 생성되었습니다: {mp3_filename}', 'success')
    return redirect(url_for('index'))

@app.route('/extract_embedding', methods=['POST'])
def extract_embedding():
    try:
        file = request.files.get('mp3file')
        if not file or not file.filename or not file.filename.endswith('.mp3'):
            flash('MP3 파일만 업로드 가능합니다.', 'danger')
            return redirect(url_for('index'))
        filename = secure_filename(file.filename)
        save_path = os.path.join(DOWNLOAD_FOLDER, filename)
        file.save(save_path)

        y, sr = librosa.load(save_path, sr=32000, mono=True)
        if y.ndim > 1:
            y = librosa.to_mono(y)
        if y.dtype != np.float32:
            y = y.astype(np.float32)
        y = y[:32000*30]
        audio_input = y

        processor, model = load_musicgen()
        inputs = processor(audio=audio_input, sampling_rate=32000, return_tensors='pt')
        input_values = inputs.get('input_values', None)
        if input_values is None:
            flash('임베딩 추출 실패: 입력값 오류', 'danger')
            return redirect(url_for('index'))
        with torch.no_grad():
            if hasattr(model, 'encodec'):
                wav = input_values.unsqueeze(1)
                frame_embeddings = model.encodec.encoder(wav)  # type: ignore
                if hasattr(frame_embeddings, 'audio_embeds'):
                    embedding_tensor = frame_embeddings.audio_embeds
                elif isinstance(frame_embeddings, dict) and 'audio_embeds' in frame_embeddings:
                    embedding_tensor = frame_embeddings['audio_embeds']
                elif isinstance(frame_embeddings, torch.Tensor):
                    embedding_tensor = frame_embeddings
                else:
                    flash('임베딩 추출 실패: encodec 결과 오류', 'danger')
                    return redirect(url_for('index'))
            elif hasattr(model, 'audio_encoder'):
                wav = input_values
                try:
                    frame_embeddings = model.audio_encoder(wav)  # type: ignore
                except Exception as e:
                    flash(f'임베딩 추출 실패: audio_encoder 호출 예외: {str(e)}', 'danger')
                    return redirect(url_for('index'))
                # audio_codes를 임베딩으로 사용
                if hasattr(frame_embeddings, 'audio_codes'):
                    embedding_tensor = frame_embeddings.audio_codes
                elif isinstance(frame_embeddings, dict) and 'audio_codes' in frame_embeddings:
                    embedding_tensor = frame_embeddings['audio_codes']
                elif isinstance(frame_embeddings, torch.Tensor):
                    embedding_tensor = frame_embeddings
                else:
                    flash(f'임베딩 추출 실패: audio_encoder 결과 오류, 반환값 타입: {type(frame_embeddings)}', 'danger')
                    return redirect(url_for('index'))
            else:
                flash('임베딩 추출 실패: 모델 구조 오류', 'danger')
                return redirect(url_for('index'))

            embedding = embedding_tensor.mean(dim=tuple(range(1, embedding_tensor.ndim))).squeeze().cpu().numpy()

        embedding_filename = f"{filename}_embedding.npy"
        embedding_path = os.path.join(DOWNLOAD_FOLDER, embedding_filename)
        np.save(embedding_path, embedding)
        flash(f'임베딩 추출 완료: {embedding_filename}', 'success')
        return redirect(url_for('index'))
    except Exception as e:
        flash(f'임베딩 추출 실패: {str(e)}', 'danger')
        return redirect(url_for('index'))

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

def get_chatbot_response(user_message: str, conversation_history: list = None) -> str:
    """로컬 Transformer 모델을 사용한 챗봇 응답 생성"""
    if conversation_history is None:
        conversation_history = []
    
    # 시스템 프롬프트: 편집 작업을 도와주는 조수 역할
    system_prompt = """당신은 편집 작업을 도와주는 친절한 한국어 조수입니다. 
사용자가 코드 작성, 파일 편집, 문제 해결 등을 도와달라고 요청하면 도움을 제공하세요.
항상 한국어로 친절하고 명확하게 답변하세요."""
    
    try:
        # 로컬 Transformer 모델 로딩
        tokenizer, model = load_chatbot_model()
        
        if tokenizer is None or model is None:
            return "죄송합니다. 챗봇 모델을 로드할 수 없습니다. 모델 다운로드 중 오류가 발생했을 수 있습니다."
        
        # 대화 히스토리와 현재 메시지를 프롬프트로 구성
        # KoGPT2는 일반 텍스트 생성 모델이므로 간단한 형식 사용
        prompt_parts = []
        
        # 시스템 프롬프트 (간단하게)
        prompt_parts.append(f"편집 조수: {system_prompt}\n")
        
        # 최근 대화 히스토리 추가 (최대 3개, 너무 길면 메모리 부족)
        for msg in conversation_history[-3:]:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'user':
                prompt_parts.append(f"사용자: {content}\n")
            elif role == 'assistant':
                prompt_parts.append(f"조수: {content}\n")
        
        # 현재 사용자 메시지 추가
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
        
        # GPU/CPU 설정 - GPU가 있으면 무조건 GPU 사용
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            device = "cuda"
            # 모델이 이미 GPU에 있는지 확인하고, 없으면 이동
            if not next(model.parameters()).is_cuda:
                model = model.to(device)
        else:
            device = "cpu"
            # GPU가 없으면 CPU 사용
            if not next(model.parameters()).is_cpu:
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
        
        # 조수 응답 부분만 추출
        if "조수:" in generated_text:
            # 마지막 "조수:" 이후의 텍스트 추출
            response = generated_text.split("조수:")[-1].strip()
            # 다음 "사용자:" 또는 줄바꿈 전까지
            if "\n사용자:" in response:
                response = response.split("\n사용자:")[0].strip()
            if "\n" in response:
                # 첫 번째 문장만 사용 (더 자연스러운 응답)
                response = response.split("\n")[0].strip()
        else:
            # 프롬프트 제거
            response = generated_text[len(full_prompt):].strip()
            # 줄바꿈이나 특수 문자 제거
            if "\n" in response:
                response = response.split("\n")[0].strip()
        
        # 빈 응답 처리
        if not response or len(response) < 3:
            response = "죄송합니다. 적절한 응답을 생성하지 못했습니다. 다시 질문해주세요."
        
        # 응답 정리 (너무 길면 자르기)
        if len(response) > 500:
            response = response[:500] + "..."
        
        return response
        
    except Exception as e:
        error_msg = str(e)
        print(f"챗봇 응답 생성 오류: {error_msg}")
        # 폴백: 규칙 기반 응답
        user_lower = user_message.lower()
        
        if any(word in user_lower for word in ['안녕', 'hello', 'hi', '하이']):
            return "안녕하세요! 편집 작업을 도와드리는 조수입니다. 무엇을 도와드릴까요? (모델 오류로 간단한 응답만 가능합니다)"
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True) 