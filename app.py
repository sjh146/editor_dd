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
    # 2분(120초)을 위해 약 1500-2000 토큰이 필요
    # 더 긴 음악을 위해 최대값 설정
    if duration_seconds <= 15:
        max_new_tokens = 256
    elif duration_seconds <= 30:
        max_new_tokens = 512
    elif duration_seconds <= 60:
        max_new_tokens = 1024
    elif duration_seconds <= 120:
        max_new_tokens = 2048
    else:
        max_new_tokens = 3072  # 최대 약 3-4분
    
    processor, model = load_musicgen()
    # MusicGen inference
    inputs = processor(text=[prompt], padding=True, return_tensors="pt")
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True) 