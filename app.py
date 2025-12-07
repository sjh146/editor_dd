import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import matplotlib
matplotlib.use('Agg')
import subprocess
import yt_dlp
from flask import (Flask, flash, redirect, render_template, request,
                   send_from_directory, url_for, jsonify)
from typing import cast
from werkzeug.utils import secure_filename
import json
import threading

# MusicGen & audio processing imports
from transformers import MusicgenForConditionalGeneration, MusicgenProcessor
import torch
import torchaudio
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'a_super_secret_key'
DOWNLOAD_FOLDER = 'downloads'
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER

if not os.path.exists(DOWNLOAD_FOLDER):
    os.makedirs(DOWNLOAD_FOLDER)

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
    # mp3, waveform, spectrogram만 필터링
    files = get_downloaded_files()
    return [f for f in files if f.endswith('.mp3') or f.endswith('.png')]

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

    ydl_opts = {
        'outtmpl': os.path.join(app.config['DOWNLOAD_FOLDER'], '%(title)s.%(ext)s'),
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        flash('Download successful!', 'success')
    except Exception as e:
        flash(f'An error occurred: {str(e)}', 'danger')

    return redirect(url_for('index'))

@app.route('/edit', methods=['POST'])
def edit():
    filename = request.form.get('filename')
    start_time = request.form.get('start_time')
    end_time = request.form.get('end_time')

    if not all([filename, start_time, end_time]):
        flash('All fields are required for editing!', 'danger')
        return redirect(url_for('index'))
    
    assert filename is not None 

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

@app.route('/downloads/<path:filename>')
def download_file(filename):
    return send_from_directory(app.config['DOWNLOAD_FOLDER'], filename, as_attachment=True)

@app.route('/musicgen', methods=['POST'])
def musicgen():
    prompt = request.form.get('prompt')
    if not prompt:
        flash('프롬프트를 입력하세요.', 'danger')
        return redirect(url_for('index'))
    processor, model = load_musicgen()
    # MusicGen inference
    inputs = processor(text=[prompt], padding=True, return_tensors="pt")
    with torch.no_grad():
        audio_values = model.generate(**inputs, max_new_tokens=256)
    # torchaudio로 mp3 저장
    mp3_filename = f"musicgen_{prompt.replace(' ', '_')}.mp3"
    mp3_path = os.path.join(DOWNLOAD_FOLDER, mp3_filename)
    torchaudio.save(mp3_path, audio_values[0].cpu(), 32000, format="mp3")
    # Waveform 이미지 생성
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
                    print('audio_encoder 반환값 타입:', type(frame_embeddings))
                    print('audio_encoder 반환값:', frame_embeddings)
                    import sys; sys.stdout.flush()
                except Exception as e:
                    flash(f'임베딩 추출 실패: audio_encoder 호출 예외: {str(e)}', 'danger')
                    print('audio_encoder 호출 예외:', e)
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
        print('임베딩 추출 실패:', e)
        return redirect(url_for('index'))

# 설문조사 자동화 관련 임포트
try:
    from survey_automation import SurveyAutomation, SurveyAnalyzer
    SURVEY_AUTOMATION_AVAILABLE = True
except ImportError as e:
    print(f"설문조사 자동화 모듈 임포트 실패: {e}")
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
        questions = analyzer.analyze_survey()
        
        if not questions:
            return jsonify({'error': '설문조사를 찾을 수 없습니다. 페이지 구조를 확인하세요.'}), 404
        
        # JSON 직렬화 가능한 형태로 변환
        questions_json = []
        for q in questions:
            q_json = {
                'type': q.get('type'),
                'name': q.get('name'),
                'question': q.get('question'),
                'options': []
            }
            for opt in q.get('options', []):
                q_json['options'].append({
                    'label': opt.get('label', ''),
                    'value': opt.get('value', '')
                })
            questions_json.append(q_json)
        
        return jsonify({
            'success': True,
            'url': url,
            'questions': questions_json,
            'total_questions': len(questions_json)
        })
    except Exception as e:
        return jsonify({'error': f'분석 중 오류 발생: {str(e)}'}), 500

@app.route('/survey/fill', methods=['POST'])
def fill_survey():
    """설문조사를 자동으로 작성합니다"""
    if not SURVEY_AUTOMATION_AVAILABLE:
        return jsonify({'error': '설문조사 자동화 모듈을 사용할 수 없습니다.'}), 500
    
    data = request.json if request.is_json else request.form.to_dict()
    url = data.get('url')
    
    # boolean 값을 안전하게 처리하는 헬퍼 함수
    def to_bool(value, default=False):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() == 'true'
        return default
    
    def to_str(value, default=''):
        if isinstance(value, str):
            return value.lower()
        return str(value).lower() if value else default
    
    use_openai = to_bool(data.get('use_openai'), False)
    openai_api_key = data.get('openai_api_key', os.environ.get('OPENAI_API_KEY'))
    auto_submit = to_bool(data.get('auto_submit'), False)
    headless = to_bool(data.get('headless'), True)
    paginated = to_bool(data.get('paginated'), False)  # 페이지네이션 모드
    browser_type = to_str(data.get('browser_type'), 'edge')  # 'edge' or 'chrome', 기본값은 'edge'
    
    if not url:
        return jsonify({'error': 'URL이 필요합니다.'}), 400
    
    automation = None
    try:
        automation = SurveyAutomation(
            url=url,
            headless=headless,
            use_openai=use_openai,
            openai_api_key=openai_api_key,
            browser_type=browser_type
        )
        
        if paginated:
            # 페이지네이션 모드 (SurveyMonkey 등)
            results = automation.fill_survey_paginated()
            
            if results is False:
                if automation:
                    automation.close()
                return jsonify({'error': '설문조사 작성 중 오류가 발생했습니다. 서버 콘솔을 확인하세요.'}), 500
            
            if automation:
                automation.close()
            
            # results가 None일 수 있으므로 체크
            if results is None:
                results = []
            
            return jsonify({
                'success': True,
                'url': url,
                'mode': 'paginated',
                'filled_questions': len([r for r in results if isinstance(r, dict) and r.get('status') == 'success']),
                'total_pages': max([r.get('page', 1) for r in results if isinstance(r, dict)], default=1),
                'results': results
            })
        else:
            # 기존 모드 (모든 질문을 한 페이지에서 분석)
            questions = automation.analyze_survey()
            if not questions:
                if automation:
                    automation.close()
                return jsonify({'error': '설문조사를 찾을 수 없습니다. 페이지네이션 모드를 시도해보세요.'}), 404
            
            # 자동 작성
            results = automation.fill_survey(questions)
            
            if results is False:
                if automation:
                    automation.close()
                return jsonify({'error': '설문조사 작성 중 오류가 발생했습니다.'}), 500
            
            # 자동 제출 (요청된 경우)
            submitted = False
            if auto_submit:
                submitted = automation.submit_survey()
            
            if automation:
                automation.close()
            
            if results is None:
                results = []
            
            return jsonify({
                'success': True,
                'url': url,
                'mode': 'single_page',
                'total_questions': len(questions) if questions else 0,
                'filled_questions': len([r for r in results if isinstance(r, dict) and r.get('status') == 'success']),
                'results': results if isinstance(results, list) else [],
                'submitted': submitted
            })
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"오류 상세:\n{error_details}")
        
        # automation 정리
        if automation:
            try:
                automation.close()
            except:
                pass
        
        # JSON 응답 반환 보장
        try:
            error_msg = str(e)
            # JSON에 문제가 될 수 있는 문자 이스케이프
            error_msg = error_msg.replace('"', "'").replace('\n', ' ').replace('\r', ' ')
            return jsonify({
                'error': f'오류 발생: {error_msg}',
                'type': type(e).__name__
            }), 500
        except Exception as json_error:
            # JSON 직렬화 실패 시 최소한의 텍스트 반환
            from flask import Response
            try:
                error_msg = str(e).replace('"', "'")[:200]  # 길이 제한
            except:
                error_msg = "알 수 없는 오류"
            return Response(
                f'{{"error": "{error_msg}"}}',
                status=500,
                mimetype='application/json'
            )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True) 