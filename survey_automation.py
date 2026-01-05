"""
설문조사 자동화 모듈 (AI 기반)
Edge 드라이버로 페이지를 파싱하고 AI가 질문, 문항, 버튼, 입력란을 찾아 자동으로 답변합니다.
OpenAI 및 DeepSeek API 지원
"""
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import NoSuchWindowException, InvalidSessionIdException, WebDriverException
import re
import time
import os
import platform
import subprocess
import traceback
import random

# Transformer 모델 임포트
try:
    from transformers import pipeline
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False

# OpenAI API 임포트
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class SurveyTransformer:
    """AI 기반 질문 해석 및 답변 생성"""
    
    def __init__(self, api_provider: str = None, api_key: str = None):
        self.classifier = None
        self.api_client = None
        self.api_provider = api_provider  # 'openai' or 'deepseek'
        self.api_error_occurred = False  # API 오류 발생 플래그
        self._load_model()
        self._init_api_client(api_key)
    
    def _load_model(self):
        """Transformer 모델 로드"""
        if not TRANSFORMER_AVAILABLE:
            return
        try:
            self.classifier = pipeline(
                "text-classification",
                model="monologg/koelectra-base-v3-discriminator",
                device=-1
            )
        except Exception as e:
            print(f"모델 로딩 실패: {e}")
    
    def _init_api_client(self, api_key: str = None):
        """OpenAI/DeepSeek API 클라이언트 초기화"""
        if not self.api_provider or not OPENAI_AVAILABLE:
            return
        
        # API 키 가져오기
        if not api_key:
            if self.api_provider == 'openai':
                api_key = os.getenv('OPENAI_API_KEY')
            elif self.api_provider == 'deepseek':
                api_key = os.getenv('DEEPSEEK_API_KEY')
        
        if not api_key:
            print(f"{self.api_provider.upper()} API 키를 찾을 수 없습니다.")
            return
        
        try:
            # DeepSeek은 base_url을 변경해야 함
            if self.api_provider == 'deepseek':
                self.api_client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
            else:
                self.api_client = OpenAI(api_key=api_key)
            print(f"{self.api_provider.upper()} API 클라이언트 초기화 완료")
        except Exception as e:
            print(f"{self.api_provider.upper()} API 클라이언트 초기화 실패: {e}")
    
    def _is_question_with_api(self, text: str) -> bool:
        """API를 사용하여 질문인지 판단"""
        if not self.api_client or self.api_error_occurred:
            return None
        
        try:
            prompt = f"""다음 텍스트가 설문조사의 질문인지 판단하세요. 질문이면 'yes', 아니면 'no'로만 답변하세요.

텍스트: {text[:200]}

답변:"""

            response = self.api_client.chat.completions.create(
                model="gpt-3.5-turbo" if self.api_provider == 'openai' else "deepseek-chat",
                messages=[
                    {"role": "system", "content": "당신은 설문조사의 질문을 판단하는 전문가입니다. 질문이면 'yes', 아니면 'no'로만 답변하세요."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=5,
                temperature=0.1
            )
            
            answer = response.choices[0].message.content.strip().lower()
            return 'yes' in answer or '예' in answer
        except Exception as e:
            # API 오류가 발생하면 플래그 설정하고 더 이상 API 사용 안 함
            if not self.api_error_occurred:
                error_msg = str(e)
                if '401' in error_msg or 'invalid_api_key' in error_msg or 'Incorrect API key' in error_msg:
                    print(f"⚠️ API 키 오류가 발생했습니다. API 기능을 비활성화하고 기본 로직을 사용합니다.")
                else:
                    print(f"⚠️ API 오류 발생: {error_msg[:100]}. API 기능을 비활성화하고 기본 로직을 사용합니다.")
                self.api_error_occurred = True
            return None
    
    def understand_question(self, question_text: str, options: list) -> int:
        """질문을 이해하고 답변 인덱스 반환 (DeepSeek API 활용)"""
        if not question_text or not options:
            return random.randint(0, len(options) - 1) if options else 0
        
        # DeepSeek API가 있으면 먼저 시도
        if self.api_client and self.api_provider == 'deepseek':
            ai_result = self._analyze_question_with_deepseek(question_text, options)
            if ai_result is not None:
                return ai_result
        
        # O, X 기호 기반 판단 (fallback)
        o_x_result = self._interpret_o_x_symbols(options)
        if o_x_result is not None:
            return o_x_result
        
        # 모르는 문자/그림 감지
        if self._has_unreadable_content(question_text, options):
            return random.randint(0, len(options) - 1)
        
        # 키워드 기반 판단 (fallback)
        question_lower = question_text.lower()
        positive_keywords = ['만족', '좋', '긍정', '동의', '예', 'yes', '좋다', '좋아', 'satisfied', 'good', 'positive', '추천']
        negative_keywords = ['불만', '나쁜', '부정', '비동의', '아니요', 'no', '나쁘다', 'disappointed', 'bad', 'negative']
        
        if any(kw in question_lower for kw in positive_keywords):
            return len(options) - 1
        elif any(kw in question_lower for kw in negative_keywords):
            return 0
        else:
            return random.randint(0, len(options) - 1)
    
    def _determine_question_type_with_deepseek(self, question_text: str, options: list = None, inputs: list = None) -> str:
        """DeepSeek API를 사용하여 질문 타입 판단"""
        if not self.api_client or self.api_provider != 'deepseek' or self.api_error_occurred:
            return None
        
        try:
            options_text = ""
            if options:
                options_text = "\n".join([f"- {opt.get('label', '옵션') if isinstance(opt, dict) else str(opt)}" 
                                         for opt in options])
            
            inputs_text = ""
            if inputs:
                inputs_text = "\n".join([f"- {inp.get('type', 'text')}: {inp.get('label', '입력란')}" 
                                        for inp in inputs])
            
            prompt = f"""다음 설문조사 질문을 분석하여 질문 타입을 판단하세요.

질문: {question_text}
{f'선택지: {options_text}' if options_text else ''}
{f'입력란: {inputs_text}' if inputs_text else ''}

질문 타입을 다음 중 하나로 판단하세요:
1. "객관식_라디오" - 라디오 버튼으로 하나만 선택하는 객관식 질문
2. "객관식_체크박스" - 체크박스로 여러 개 선택하는 객관식 질문
3. "주관식_텍스트" - 텍스트 입력란에 자유롭게 입력하는 주관식 질문
4. "주관식_숫자" - 숫자만 입력하는 주관식 질문
5. "주관식_이메일" - 이메일 주소를 입력하는 주관식 질문
6. "주관식_전화번호" - 전화번호를 입력하는 주관식 질문
7. "주관식_날짜" - 날짜를 입력하는 주관식 질문
8. "버튼" - 버튼을 클릭하는 질문

답변은 타입만 제공하세요. 예: 객관식_라디오"""

            response = self.api_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "당신은 설문조사의 질문 타입을 판단하는 전문가입니다. 질문과 선택지/입력란을 분석하여 정확한 타입을 판단하세요."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=20,
                temperature=0.1
            )
            
            answer_text = response.choices[0].message.content.strip()
            print(f"    DeepSeek 타입 판단 결과: {answer_text}")
            return answer_text
        except Exception as e:
            # API 오류가 발생하면 플래그 설정하고 더 이상 API 사용 안 함
            if not self.api_error_occurred:
                error_msg = str(e)
                if '401' in error_msg or 'invalid_api_key' in error_msg or 'Incorrect API key' in error_msg:
                    print(f"    ⚠️ API 키 오류가 발생했습니다. API 기능을 비활성화하고 기본 로직을 사용합니다.")
                else:
                    print(f"    ⚠️ DeepSeek 타입 판단 오류: {error_msg[:100]}")
                self.api_error_occurred = True
            return None
    
    def _analyze_question_with_deepseek(self, question_text: str, options: list) -> int:
        """DeepSeek API를 사용하여 질문과 옵션을 분석하고 적절한 답변 선택"""
        if not self.api_client or not options or self.api_error_occurred:
            return None
        
        try:
            # 옵션 리스트 생성
            options_text = "\n".join([f"{idx + 1}. {opt.get('label', '옵션') if isinstance(opt, dict) else str(opt)}" 
                                     for idx, opt in enumerate(options)])
            
            prompt = f"""다음 설문조사 질문과 선택지를 분석하여 가장 적절한 답변을 선택하세요.

질문: {question_text}

선택지:
{options_text}

지시사항:
1. 질문의 의도를 정확히 파악하세요.
2. 각 선택지의 의미를 이해하세요.
3. 일반적으로 긍정적이고 적절한 답변을 선택하세요.
4. 만족도 조사인 경우 보통 중간 이상의 긍정적인 답변을 선택하세요.
5. O/X 질문인 경우 O(긍정)를 선택하세요.
6. 숫자나 등급이 있는 경우 중간 이상의 값을 선택하세요.

답변은 반드시 선택지 번호(1부터 시작)만 숫자로 답변하세요. 예: 1, 2, 3 등"""

            response = self.api_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "당신은 설문조사에 답변하는 전문가입니다. 질문을 분석하고 가장 적절한 선택지를 번호로만 답변하세요."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0.3
            )
            
            answer_text = response.choices[0].message.content.strip()
            print(f"    DeepSeek 분석 결과: {answer_text}")
            
            # 숫자 추출
            import re
            numbers = re.findall(r'\d+', answer_text)
            if numbers:
                answer_idx = int(numbers[0]) - 1  # 1-based to 0-based
                if 0 <= answer_idx < len(options):
                    print(f"    ✓ DeepSeek 선택: {answer_idx + 1}번 옵션 ({options[answer_idx].get('label', '옵션') if isinstance(options[answer_idx], dict) else str(options[answer_idx])})")
                    return answer_idx
            
            return None
        except Exception as e:
            # API 오류가 발생하면 플래그 설정하고 더 이상 API 사용 안 함
            if not self.api_error_occurred:
                error_msg = str(e)
                if '401' in error_msg or 'invalid_api_key' in error_msg or 'Incorrect API key' in error_msg:
                    print(f"    ⚠️ API 키 오류가 발생했습니다. API 기능을 비활성화하고 기본 로직을 사용합니다.")
                else:
                    print(f"    ⚠️ DeepSeek 분석 오류: {error_msg[:100]}")
                self.api_error_occurred = True
            return None
    
    def _interpret_o_x_symbols(self, options: list) -> int:
        """O, X 기호 해석"""
        if not options:
            return None
        
        # O, X 기호 패턴 (다양한 형태)
        o_symbols = ['o', 'O', '○', '⭕', '✓', '✔', '✅', '◯', '〇', 'YES', 'yes', '예', '동의', '긍정']
        x_symbols = ['x', 'X', '×', '✗', '✘', '❌', '❎', 'NO', 'no', '아니오', '비동의', '부정']
        
        for idx, opt in enumerate(options):
            label = opt.get('label', '') or opt.get('value', '') if isinstance(opt, dict) else str(opt)
            if not label:
                continue
            
            label_clean = label.strip()
            
            # O 기호가 있으면 해당 옵션 선택 (긍정)
            if any(symbol in label_clean for symbol in o_symbols):
                print(f"  O 기호 감지: {label_clean} → 긍정 선택")
                return idx
            
            # X 기호가 있으면 해당 옵션 선택 (부정)
            if any(symbol in label_clean for symbol in x_symbols):
                print(f"  X 기호 감지: {label_clean} → 부정 선택")
                return idx
        
        # O, X 기호가 없으면 None 반환 (다른 로직 사용)
        return None
    
    def _has_unreadable_content(self, question_text: str, options: list) -> bool:
        """모르는 문자/그림 감지 (O, X 기호는 읽을 수 있는 것으로 간주)"""
        try:
            if question_text:
                special_chars = re.findall(r'[^\w\s가-힣a-zA-Z0-9.,!?()\-]', question_text)
                if len(special_chars) > len(question_text) * 0.3:
                    return True
                readable_chars = re.findall(r'[가-힣a-zA-Z0-9]', question_text)
                if len(readable_chars) < len(question_text) * 0.2:
                    return True
            
            unreadable_count = 0
            o_x_found = False  # O, X 기호가 있는지 확인
            
            for opt in options:
                label = opt.get('label', '') or opt.get('value', '') if isinstance(opt, dict) else str(opt)
                if not label:
                    unreadable_count += 1
                    continue
                
                label_clean = label.strip()
                
                # O, X 기호 체크
                o_x_symbols = ['o', 'O', 'x', 'X', '○', '×', '⭕', '❌', '✓', '✗', '✔', '✘', '◯', '〇', '✅', '❎']
                if any(symbol in label_clean for symbol in o_x_symbols):
                    o_x_found = True
                    continue  # O, X 기호가 있으면 읽을 수 있는 것으로 간주
                
                if label_clean in ['옵션', 'option', '선택', 'select']:
                    unreadable_count += 1
                    continue
                
                special_chars = re.findall(r'[^\w\s가-힣a-zA-Z0-9.,!?()\-]', label)
                if len(special_chars) > len(label) * 0.4:
                    unreadable_count += 1
            
            # O, X 기호가 있으면 읽을 수 있는 것으로 간주
            if o_x_found:
                return False
            
            return unreadable_count >= len(options) * 0.5 if options else False
        except:
            return True
    
    def generate_text_answer(self, question_text: str, input_type: str = 'text') -> str:
        """텍스트 입력용 답변 생성 (입력 타입 고려)"""
        if not question_text:
            question_text = "입력"
        
        # 숫자 입력란 처리
        if input_type == 'number':
            # API 클라이언트가 있으면 먼저 시도
            if self.api_client:
                api_answer = self._generate_text_with_api(question_text, input_type)
                if api_answer:
                    # 숫자만 추출
                    import re
                    numbers = re.findall(r'\d+', api_answer)
                    if numbers:
                        return numbers[0]
            
            # 키워드 기반 숫자 답변 생성
            question_lower = question_text.lower()
            if any(kw in question_lower for kw in ['나이', 'age', '연령']):
                return str(random.randint(20, 50))
            elif any(kw in question_lower for kw in ['인원', '명', '사람', 'people', 'count']):
                return str(random.randint(1, 10))
            elif any(kw in question_lower for kw in ['점수', '점', 'score', 'rating']):
                return str(random.randint(3, 5))
            elif any(kw in question_lower for kw in ['년', 'year']):
                return str(random.randint(2000, 2024))
            else:
                # 기본적으로 1-100 사이의 숫자
                return str(random.randint(1, 100))
        
        # API 클라이언트가 있으면 먼저 시도
        if self.api_client:
            api_answer = self._generate_text_with_api(question_text, input_type)
            if api_answer:
                return api_answer
        
        question_lower = question_text.lower()
        if '이름' in question_text or 'name' in question_lower:
            return "홍길동"
        elif '이메일' in question_text or 'email' in question_lower or input_type == 'email':
            return "test@example.com"
        elif '전화' in question_text or 'phone' in question_lower or input_type == 'tel':
            return "010-1234-5678"
        elif '주소' in question_text or 'address' in question_lower:
            return "서울특별시 강남구 테헤란로 123"
        elif '생년월일' in question_text or 'birth' in question_lower or input_type == 'date':
            return "2000-01-01"
        elif '의견' in question_text or 'opinion' in question_lower:
            return "전반적으로 만족스럽고 좋은 서비스라고 생각합니다."
        else:
            return "답변" if len(question_text) < 20 else "설문조사에 참여하게 되어 기쁩니다."
    
    def _generate_text_with_api(self, question_text: str, input_type: str = 'text') -> str:
        """API를 사용하여 텍스트 답변 생성 (입력 타입 고려)"""
        if not self.api_client or self.api_error_occurred:
            return None
        
        try:
            # 입력 타입에 따른 프롬프트 조정
            if input_type == 'number':
                prompt = f"""다음 설문조사 질문에 적절한 숫자만 답변하세요. 숫자만 입력하세요 (예: 25, 100).

질문: {question_text}

답변 (숫자만):"""
                system_content = "당신은 설문조사에 답변하는 사용자입니다. 질문에 맞는 적절한 숫자만 답변하세요. 예를 들어, 나이면 20-50 사이의 숫자, 점수면 1-5 사이의 숫자를 답변하세요."
            elif input_type == 'email':
                prompt = f"""다음 설문조사 질문에 적절한 이메일 주소만 답변하세요.

질문: {question_text}

답변 (이메일 주소만):"""
                system_content = "당신은 설문조사에 답변하는 사용자입니다. 질문에 맞는 적절한 이메일 주소만 답변하세요 (예: test@example.com)."
            elif input_type == 'tel':
                prompt = f"""다음 설문조사 질문에 적절한 전화번호만 답변하세요.

질문: {question_text}

답변 (전화번호만):"""
                system_content = "당신은 설문조사에 답변하는 사용자입니다. 질문에 맞는 적절한 전화번호만 답변하세요 (예: 010-1234-5678)."
            elif input_type == 'date':
                prompt = f"""다음 설문조사 질문에 적절한 날짜만 답변하세요 (YYYY-MM-DD 형식).

질문: {question_text}

답변 (날짜만):"""
                system_content = "당신은 설문조사에 답변하는 사용자입니다. 질문에 맞는 적절한 날짜만 YYYY-MM-DD 형식으로 답변하세요."
            else:
                prompt = f"""다음 설문조사 질문에 적절한 한 문장 답변을 작성하세요. 간결하고 자연스러운 답변을 작성하세요.

질문: {question_text}

답변:"""
                system_content = "당신은 설문조사에 답변하는 사용자입니다. 질문에 맞는 간결하고 자연스러운 답변을 작성하세요."

            response = self.api_client.chat.completions.create(
                model="gpt-3.5-turbo" if self.api_provider == 'openai' else "deepseek-chat",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50 if input_type == 'number' else 100,
                temperature=0.3 if input_type == 'number' else 0.7
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            # API 오류가 발생하면 플래그 설정하고 더 이상 API 사용 안 함
            if not self.api_error_occurred:
                error_msg = str(e)
                if '401' in error_msg or 'invalid_api_key' in error_msg or 'Incorrect API key' in error_msg:
                    print(f"    ⚠️ API 키 오류가 발생했습니다. API 기능을 비활성화하고 기본 로직을 사용합니다.")
                else:
                    print(f"    ⚠️ API 텍스트 생성 오류: {error_msg[:100]}")
                self.api_error_occurred = True
            return None


class SurveyAutoFill:
    """설문조사 자동 작성"""
    
    def __init__(self, url: str, headless: bool = True, api_provider: str = None, api_key: str = None):
        self.url = url
        self.headless = headless
        self.driver = None
        self.transformer = SurveyTransformer(api_provider=api_provider, api_key=api_key)
    
    def _init_driver(self):
        """Edge 드라이버 초기화"""
        try:
            edge_path = self._find_edge_executable()
            options = EdgeOptions()
            if self.headless:
                options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-blink-features=AutomationControlled')
            options.add_experimental_option('excludeSwitches', ['enable-logging'])
            options.add_experimental_option('useAutomationExtension', False)
            if edge_path:
                options.binary_location = edge_path
            
            self.driver = webdriver.Edge(options=options)
            # 브라우저 창 최대화 시도
            try:
                self.driver.maximize_window()
            except:
                pass
            return True
        except Exception as e:
            print(f"Edge 드라이버 초기화 실패: {e}")
            return False
    
    def _is_driver_valid(self):
        """드라이버 세션 유효성 확인"""
        try:
            if not self.driver:
                return False
            # 간단한 명령으로 세션 확인
            _ = self.driver.current_url
            return True
        except (NoSuchWindowException, InvalidSessionIdException, WebDriverException):
            return False
        except:
            return False
    
    def _find_edge_executable(self):
        """Edge 실행 파일 찾기"""
        if platform.system() != 'Windows':
            return None
        
        paths = [
            r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
            r'C:\Program Files\Microsoft\Edge\Application\msedge.exe',
            os.path.expanduser(r'~\AppData\Local\Microsoft\Edge\Application\msedge.exe'),
        ]
        
        for path in paths:
            if os.path.exists(path):
                return path
        
        try:
            ps_cmd = "Get-ItemProperty 'HKLM:\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\App Paths\\msedge.exe' -ErrorAction SilentlyContinue | Select-Object -ExpandProperty '(default)'"
            result = subprocess.run(['powershell', '-Command', ps_cmd], capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                path = result.stdout.strip()
                if os.path.exists(path):
                    return path
        except:
            pass
        return None
    
    def _analyze_page_structure_with_deepseek(self, question_texts: list, radio_groups: dict, checkbox_groups: dict, text_inputs: list) -> dict:
        """DeepSeek API를 사용하여 페이지 구조 분석 (질문 타입 판단)"""
        if not self.transformer.api_client or self.transformer.api_provider != 'deepseek' or self.transformer.api_error_occurred:
            return None
        
        try:
            # 질문 텍스트와 발견된 요소들을 정리
            questions_info = []
            for q_text in question_texts[:10]:  # 최대 10개 질문만
                questions_info.append(f"- {q_text[:100]}")
            
            radio_info = []
            for name, radios in list(radio_groups.items())[:5]:  # 최대 5개 그룹만
                labels = [r.get('label', '옵션') for r in radios[:5]]
                radio_info.append(f"  그룹 '{name}': {', '.join(labels)}")
            
            checkbox_info = []
            for name, cbs in list(checkbox_groups.items())[:5]:
                labels = [cb.get('label', '옵션') for cb in cbs[:5]]
                checkbox_info.append(f"  그룹 '{name}': {', '.join(labels)}")
            
            prompt = f"""다음 설문조사 페이지 정보를 분석하여 각 질문의 타입을 판단하세요.

발견된 질문들:
{chr(10).join(questions_info) if questions_info else '질문 없음'}

발견된 라디오 버튼 그룹:
{chr(10).join(radio_info) if radio_info else '라디오 버튼 없음'}

발견된 체크박스 그룹:
{chr(10).join(checkbox_info) if checkbox_info else '체크박스 없음'}

각 질문에 대해 다음 정보를 JSON 형식으로 제공하세요:
1. 질문 텍스트
2. 질문 타입: "객관식_라디오", "객관식_체크박스", "주관식_텍스트", "주관식_숫자", "주관식_이메일", "주관식_전화번호", "주관식_날짜", "버튼"
3. 선택지 (객관식인 경우)

JSON 형식:
{{
  "questions": [
    {{
      "question": "질문 텍스트",
      "type": "객관식_라디오",
      "options": ["옵션1", "옵션2"]
    }}
  ]
}}

답변은 JSON만 제공하세요."""

            response = self.transformer.api_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "당신은 설문조사 페이지를 분석하는 전문가입니다. 질문과 발견된 요소들을 분석하여 각 질문의 타입을 정확히 판단하세요."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.1
            )
            
            result_text = response.choices[0].message.content.strip()
            print(f"  DeepSeek 페이지 분석 결과: {result_text[:200]}...")
            
            # JSON 추출
            import json
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result
            
            return None
        except Exception as e:
            # API 오류가 발생하면 플래그 설정하고 더 이상 API 사용 안 함
            if not self.transformer.api_error_occurred:
                error_msg = str(e)
                if '401' in error_msg or 'invalid_api_key' in error_msg or 'Incorrect API key' in error_msg:
                    print(f"  ⚠️ API 키 오류가 발생했습니다. API 기능을 비활성화하고 기본 로직을 사용합니다.")
                else:
                    print(f"  ⚠️ DeepSeek 페이지 분석 오류: {error_msg[:100]}")
                self.transformer.api_error_occurred = True
            return None
    
    def _parse_page(self):
        """AI 기반 페이지 파싱 - 질문, 문항, 버튼, 입력란 찾기"""
        try:
            WebDriverWait(self.driver, 15).until(
                lambda d: d.execute_script('return document.readyState') == 'complete'
            )
            time.sleep(2)
            
            # DeepSeek으로 페이지 구조 분석 시도 (질문을 먼저 찾은 후)
            # 질문을 먼저 찾고 나서 DeepSeek 분석을 수행하는 것이 더 효과적
            
            questions = []
            seen_questions = set()
            
            # 1. 질문 찾기
            question_elements = []
            selectors = ["h1, h2, h3, h4, h5, h6", "legend", "label", "p", "div[class*='question']"]
            for selector in selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    for elem in elements:
                        try:
                            text = elem.text.strip()
                            if text and 5 < len(text) < 500 and self._is_question_text(text):
                                question_elements.append((elem, text))
                        except:
                            continue
                except:
                    continue
            
            # 2. 라디오/체크박스 찾기 (다양한 방법 시도)
            radio_inputs = []
            checkbox_inputs = []
            
            # 방법 1: 표준 input[type='radio']
            try:
                radio_inputs.extend(self.driver.find_elements(By.CSS_SELECTOR, "input[type='radio']"))
            except:
                pass
            
            # 방법 2: role='radio' 속성을 가진 요소
            try:
                radio_elements = self.driver.find_elements(By.CSS_SELECTOR, "[role='radio']")
                for elem in radio_elements:
                    # 실제 input 요소를 찾거나 클릭 가능한 요소로 사용
                    try:
                        input_elem = elem.find_element(By.XPATH, ".//input[@type='radio']")
                        if input_elem not in radio_inputs:
                            radio_inputs.append(input_elem)
                    except:
                        # input이 없으면 요소 자체를 사용 (커스텀 라디오 버튼)
                        if elem not in radio_inputs:
                            radio_inputs.append(elem)
            except:
                pass
            
            # 방법 3: aria-checked 속성을 가진 요소 (라디오/체크박스)
            try:
                aria_elements = self.driver.find_elements(By.CSS_SELECTOR, "[aria-checked]")
                for elem in aria_elements:
                    role = elem.get_attribute('role') or ''
                    elem_class = elem.get_attribute('class') or ''
                    if role == 'radio' or 'radio' in elem_class:
                        if elem not in radio_inputs:
                            radio_inputs.append(elem)
                    elif role == 'checkbox' or 'checkbox' in elem_class:
                        if elem not in checkbox_inputs:
                            checkbox_inputs.append(elem)
            except:
                pass
            
            # 방법 4: label과 연결된 input 찾기
            try:
                labels = self.driver.find_elements(By.TAG_NAME, "label")
                for label in labels:
                    try:
                        label_for = label.get_attribute('for')
                        if label_for:
                            input_elem = self.driver.find_element(By.ID, label_for)
                            input_type = input_elem.get_attribute('type')
                            if input_type == 'radio' and input_elem not in radio_inputs:
                                radio_inputs.append(input_elem)
                            elif input_type == 'checkbox' and input_elem not in checkbox_inputs:
                                checkbox_inputs.append(input_elem)
                    except:
                        # label 내부의 input 찾기
                        try:
                            input_elem = label.find_element(By.CSS_SELECTOR, "input[type='radio'], input[type='checkbox']")
                            input_type = input_elem.get_attribute('type')
                            if input_type == 'radio' and input_elem not in radio_inputs:
                                radio_inputs.append(input_elem)
                            elif input_type == 'checkbox' and input_elem not in checkbox_inputs:
                                checkbox_inputs.append(input_elem)
                        except:
                            pass
            except:
                pass
            
            # 방법 5: 표준 input[type='checkbox']
            try:
                checkbox_inputs.extend(self.driver.find_elements(By.CSS_SELECTOR, "input[type='checkbox']"))
            except:
                pass
            
            # 중복 제거
            radio_inputs = list(dict.fromkeys(radio_inputs))  # 순서 유지하면서 중복 제거
            checkbox_inputs = list(dict.fromkeys(checkbox_inputs))
            
            print(f"  발견된 라디오 버튼 수: {len(radio_inputs)}")
            print(f"  발견된 체크박스 수: {len(checkbox_inputs)}")
            
            # 디버깅: 라디오 버튼이 없으면 페이지 구조 확인
            if len(radio_inputs) == 0 and len(checkbox_inputs) == 0:
                print(f"  ⚠️ 라디오/체크박스를 찾지 못했습니다. 페이지 구조 확인 중...")
                try:
                    # 다른 선택 가능한 요소 찾기
                    clickable_elements = self.driver.find_elements(By.CSS_SELECTOR, 
                        "[onclick], [data-value], [data-option], .option, .choice, .select-item")
                    print(f"    클릭 가능한 요소 수: {len(clickable_elements)}")
                    
                    # select 요소 찾기
                    select_elements = self.driver.find_elements(By.TAG_NAME, "select")
                    print(f"    select 요소 수: {len(select_elements)}")
                    
                    # 버튼이나 클릭 가능한 요소 찾기
                    clickable_buttons = self.driver.find_elements(By.CSS_SELECTOR, 
                        "button[class*='option'], button[class*='choice'], div[class*='option'][role='button']")
                    print(f"    옵션 버튼 수: {len(clickable_buttons)}")
                except Exception as e:
                    print(f"    페이지 구조 확인 오류: {e}")
            
            radio_groups = {}
            unnamed_radio_index = 0  # name이 없는 라디오 버튼용 인덱스
            
            for radio in radio_inputs:
                try:
                    if not radio.is_displayed():
                        continue
                    
                    # name 속성 가져오기 (input 요소인 경우)
                    name = radio.get_attribute('name') or ''
                    
                    # name이 없으면 고유한 키 생성 (부모 요소 기반)
                    if not name:
                        try:
                            # aria-label이나 data-name 등 다른 속성 확인
                            name = (radio.get_attribute('aria-label') or 
                                   radio.get_attribute('data-name') or 
                                   radio.get_attribute('data-group') or '')
                            
                            # 여전히 name이 없으면 부모 요소 기반으로 생성
                            if not name:
                                parent = radio.find_element(By.XPATH, "./..")
                                parent_id = parent.get_attribute('id') or ''
                                parent_class = parent.get_attribute('class') or ''
                                # 부모의 id나 class를 사용하여 그룹화
                                if parent_id:
                                    name = f"_unnamed_group_{parent_id}"
                                elif parent_class:
                                    name = f"_unnamed_group_{parent_class[:30]}"
                                else:
                                    # 마지막 수단: 인덱스 사용
                                    name = f"_unnamed_radio_{unnamed_radio_index}"
                                    unnamed_radio_index += 1
                        except:
                            name = f"_unnamed_radio_{unnamed_radio_index}"
                            unnamed_radio_index += 1
                    
                    if name not in radio_groups:
                        radio_groups[name] = []
                    label = self._get_label_for_input(radio)
                    radio_groups[name].append({'element': radio, 'label': label})
                    print(f"    라디오 버튼 추가: name={name}, label={label}")
                except Exception as e:
                    print(f"    라디오 버튼 처리 오류: {e}")
                    continue
            
            checkbox_groups = {}
            unnamed_checkbox_index = 0
            
            for cb in checkbox_inputs:
                try:
                    if not cb.is_displayed():
                        continue
                    name = cb.get_attribute('name') or ''
                    # name이 없으면 고유한 키 생성
                    if not name:
                        try:
                            parent = cb.find_element(By.XPATH, "./..")
                            parent_id = parent.get_attribute('id') or ''
                            parent_class = parent.get_attribute('class') or ''
                            if parent_id:
                                name = f"_unnamed_cb_group_{parent_id}"
                            elif parent_class:
                                name = f"_unnamed_cb_group_{parent_class[:30]}"
                            else:
                                name = f"_unnamed_checkbox_{unnamed_checkbox_index}"
                                unnamed_checkbox_index += 1
                        except:
                            name = f"_unnamed_checkbox_{unnamed_checkbox_index}"
                            unnamed_checkbox_index += 1
                    
                    if name not in checkbox_groups:
                        checkbox_groups[name] = []
                    label = self._get_label_for_input(cb)
                    checkbox_groups[name].append({'element': cb, 'label': label})
                except Exception as e:
                    print(f"    체크박스 처리 오류: {e}")
                    continue
            
            print(f"  라디오 그룹 수: {len(radio_groups)}")
            print(f"  체크박스 그룹 수: {len(checkbox_groups)}")
            
            # DeepSeek으로 페이지 구조 분석 시도 (질문과 요소를 찾은 후)
            if self.transformer.api_client and self.transformer.api_provider == 'deepseek' and not self.transformer.api_error_occurred:
                try:
                    question_texts = [q[1] for q in question_elements]
                    deepseek_analysis = self._analyze_page_structure_with_deepseek(
                        question_texts, radio_groups, checkbox_groups, []
                    )
                    if deepseek_analysis and deepseek_analysis.get('questions'):
                        print(f"  DeepSeek 분석 완료: {len(deepseek_analysis.get('questions', []))}개 질문 발견")
                except Exception as e:
                    print(f"  DeepSeek 분석 시도 실패: {e}")
            
            # 3. 텍스트 입력란 찾기
            text_inputs = []
            for input_type in ['text', 'email', 'number', 'tel', 'date', 'textarea']:
                try:
                    if input_type == 'textarea':
                        inputs = self.driver.find_elements(By.TAG_NAME, "textarea")
                    else:
                        inputs = self.driver.find_elements(By.CSS_SELECTOR, f"input[type='{input_type}']")
                    for inp in inputs:
                        try:
                            if inp.is_displayed() and inp.is_enabled():
                                label = self._get_label_for_input(inp)
                                text_inputs.append({'element': inp, 'type': input_type, 'label': label})
                        except:
                            continue
                except:
                    continue
            
            # 4. 버튼 찾기
            buttons = []
            for selector in ["button", "input[type='button']", "input[type='submit']", "a[role='button']"]:
                try:
                    btns = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    for btn in btns:
                        try:
                            if btn.is_displayed() and btn.is_enabled():
                                btn_text = btn.text.strip() or btn.get_attribute('value') or ''
                                if btn_text:
                                    buttons.append({'element': btn, 'label': btn_text})
                        except:
                            continue
                except:
                    continue
            
            # 5. 질문과 요소 매칭 (개선된 로직: 이미 사용된 요소 추적)
            used_elements = set()  # 이미 매칭된 요소 추적
            used_radio_names = set()  # 이미 사용된 라디오 그룹 이름
            used_checkbox_names = set()  # 이미 사용된 체크박스 그룹 이름
            
            for question_elem, question_text in question_elements:
                try:
                    cleaned = self._clean_question_text(question_text)
                    if not cleaned or len(cleaned) < 3:
                        continue
                    
                    question_key = cleaned[:50]
                    if question_key in seen_questions:
                        continue
                    seen_questions.add(question_key)
                    
                    # 질문 요소의 위치 정보 가져오기
                    try:
                        question_rect = question_elem.rect
                        question_y = question_rect['y']
                    except:
                        question_y = 0
                    
                    # 질문 요소의 컨테이너 찾기 (더 넓은 범위로)
                    container = None
                    try:
                        # 부모 컨테이너 찾기 (fieldset, div, form 등)
                        for xpath in ["./ancestor::fieldset[1]", "./ancestor::div[@class*='question'][1]", 
                                     "./ancestor::div[@class*='form'][1]", "./ancestor::div[@class*='survey'][1]",
                                     "./ancestor::div[@class*='item'][1]", "./ancestor::li[1]", 
                                     "./ancestor::div[@role='group'][1]", "./.."]:
                            try:
                                container = question_elem.find_element(By.XPATH, xpath)
                                if container:
                                    break
                            except:
                                continue
                    except:
                        pass
                    
                    # 라디오 버튼 찾기 (개선된 로직)
                    options = []
                    for name, radios in radio_groups.items():
                        if name in used_radio_names:
                            continue
                        try:
                            matched_radios = []
                            
                            # 컨테이너 내에 있는지 확인
                            if container:
                                try:
                                    # name 속성으로 찾기
                                    container_radios = container.find_elements(By.CSS_SELECTOR, f"input[type='radio'][name='{name}']")
                                    if not container_radios:
                                        # name이 없는 경우 다른 방법으로 찾기
                                        container_radios = container.find_elements(By.CSS_SELECTOR, "input[type='radio']")
                                    
                                    if container_radios:
                                        # 컨테이너 내의 라디오 버튼만 필터링
                                        for radio_info in radios:
                                            radio_elem = radio_info['element']
                                            if id(radio_elem) in used_elements:
                                                continue
                                            # 컨테이너 내에 있는지 확인
                                            try:
                                                if radio_elem in container_radios or container.find_elements(By.XPATH, f".//input[@id='{radio_elem.get_attribute('id')}']"):
                                                    matched_radios.append(radio_info)
                                            except:
                                                # 위치 기반으로 확인
                                                try:
                                                    radio_rect = radio_elem.rect
                                                    radio_y = radio_rect['y']
                                                    # 질문 아래 1000px 이내에 있는 것 (범위 확대)
                                                    if radio_y >= question_y - 100 and radio_y <= question_y + 1000:
                                                        matched_radios.append(radio_info)
                                                except:
                                                    matched_radios.append(radio_info)
                                        if matched_radios:
                                            options.extend(matched_radios)
                                            used_radio_names.add(name)
                                            for radio in matched_radios:
                                                used_elements.add(id(radio['element']))
                                except Exception as e:
                                    print(f"    컨테이너 내 라디오 찾기 오류: {e}")
                            
                            # 컨테이너가 없거나 매칭 실패 시 위치 기반으로 찾기
                            if not matched_radios:
                                for radio_info in radios:
                                    radio_elem = radio_info['element']
                                    if id(radio_elem) in used_elements:
                                        continue
                                    try:
                                        radio_rect = radio_elem.rect
                                        radio_y = radio_rect['y']
                                        # 질문 아래 1000px 이내에 있는 것 (범위 확대)
                                        if radio_y >= question_y - 100 and radio_y <= question_y + 1000:
                                            matched_radios.append(radio_info)
                                            used_elements.add(id(radio_elem))
                                    except:
                                        # 위치 정보를 가져올 수 없으면 일단 추가 (마지막 수단)
                                        matched_radios.append(radio_info)
                                        used_elements.add(id(radio_elem))
                                
                                if matched_radios:
                                    options.extend(matched_radios)
                                    used_radio_names.add(name)
                        except Exception as e:
                            print(f"    라디오 버튼 매칭 오류: {e}")
                            pass
                    
                    # 체크박스 찾기 (질문과 가까운 것만)
                    checkbox_options = []
                    for name, checkboxes in checkbox_groups.items():
                        if name in used_checkbox_names:
                            continue
                        try:
                            if container:
                                if container.find_elements(By.CSS_SELECTOR, f"input[name='{name}']"):
                                    first_cb = checkboxes[0]['element']
                                    try:
                                        cb_rect = first_cb.rect
                                        cb_y = cb_rect['y']
                                        if cb_y >= question_y and cb_y <= question_y + 500:
                                            checkbox_options.extend(checkboxes)
                                            used_checkbox_names.add(name)
                                            for cb in checkboxes:
                                                used_elements.add(id(cb['element']))
                                    except:
                                        checkbox_options.extend(checkboxes)
                                        used_checkbox_names.add(name)
                                        for cb in checkboxes:
                                            used_elements.add(id(cb['element']))
                            else:
                                for cb_info in checkboxes:
                                    cb_elem = cb_info['element']
                                    if id(cb_elem) in used_elements:
                                        continue
                                    try:
                                        cb_rect = cb_elem.rect
                                        cb_y = cb_rect['y']
                                        if cb_y >= question_y and cb_y <= question_y + 500:
                                            checkbox_options.append(cb_info)
                                            used_elements.add(id(cb_elem))
                                    except:
                                        pass
                                if checkbox_options:
                                    used_checkbox_names.add(name)
                        except:
                            pass
                    
                    # 텍스트 입력란 찾기 (질문과 가까운 것만)
                    inputs = []
                    for text_inp in text_inputs:
                        inp_elem = text_inp['element']
                        if id(inp_elem) in used_elements:
                            continue
                        try:
                            if container:
                                if container.find_elements(By.CSS_SELECTOR, "input, textarea"):
                                    try:
                                        inp_rect = inp_elem.rect
                                        inp_y = inp_rect['y']
                                        if inp_y >= question_y and inp_y <= question_y + 500:
                                            inputs.append(text_inp)
                                            used_elements.add(id(inp_elem))
                                    except:
                                        inputs.append(text_inp)
                                        used_elements.add(id(inp_elem))
                            else:
                                try:
                                    inp_rect = inp_elem.rect
                                    inp_y = inp_rect['y']
                                    if inp_y >= question_y and inp_y <= question_y + 500:
                                        inputs.append(text_inp)
                                        used_elements.add(id(inp_elem))
                                except:
                                    pass
                        except:
                            pass
                    
                    # 버튼 찾기 (질문과 가까운 것만, 다음/제출 버튼은 제외)
                    question_buttons = []
                    for btn_info in buttons:
                        btn_elem = btn_info['element']
                        if id(btn_elem) in used_elements:
                            continue
                        btn_text = btn_info.get('label', '').lower()
                        # 다음/제출 버튼은 제외
                        if any(kw in btn_text for kw in ['다음', 'next', '제출', 'submit', '확인', 'confirm']):
                            continue
                        try:
                            btn_rect = btn_elem.rect
                            btn_y = btn_rect['y']
                            if btn_y >= question_y and btn_y <= question_y + 500:
                                question_buttons.append(btn_info)
                                used_elements.add(id(btn_elem))
                        except:
                            pass
                    
                    # 질문 타입 결정 (우선순위: radio > checkbox > text > button)
                    if options:
                        questions.append({'type': 'radio', 'question': cleaned, 'options': options})
                        print(f"  ✓ 질문 발견 (라디오): {cleaned[:50]}... ({len(options)}개 옵션)")
                    elif checkbox_options:
                        questions.append({'type': 'checkbox', 'question': cleaned, 'options': checkbox_options})
                        print(f"  ✓ 질문 발견 (체크박스): {cleaned[:50]}... ({len(checkbox_options)}개 옵션)")
                    elif inputs:
                        questions.append({'type': 'text', 'question': cleaned, 'inputs': inputs})
                        print(f"  ✓ 질문 발견 (텍스트 입력): {cleaned[:50]}... ({len(inputs)}개 입력란)")
                    elif question_buttons:
                        questions.append({'type': 'button', 'question': cleaned, 'options': question_buttons})
                        print(f"  ✓ 질문 발견 (버튼): {cleaned[:50]}... ({len(question_buttons)}개 버튼)")
                    else:
                        print(f"  ⚠ 질문 발견했지만 매칭된 요소 없음: {cleaned[:50]}...")
                except Exception as e:
                    print(f"질문 매칭 오류: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # 질문이 없지만 요소가 있는 경우 (사용되지 않은 요소만)
            if not questions:
                # 사용되지 않은 라디오 버튼 찾기
                unused_radio_groups = []
                for name, radios in radio_groups.items():
                    if name not in used_radio_names:
                        unused_radios = [r for r in radios if id(r['element']) not in used_elements]
                        if unused_radios:
                            unused_radio_groups.extend(unused_radios)
                
                # 사용되지 않은 체크박스 찾기
                unused_checkbox_groups = []
                for name, checkboxes in checkbox_groups.items():
                    if name not in used_checkbox_names:
                        unused_cbs = [cb for cb in checkboxes if id(cb['element']) not in used_elements]
                        if unused_cbs:
                            unused_checkbox_groups.extend(unused_cbs)
                
                unused_text_inputs = [inp for inp in text_inputs if id(inp['element']) not in used_elements]
                unused_buttons = [btn for btn in buttons if id(btn['element']) not in used_elements]
                
                # 우선순위: 라디오 > 체크박스 > 텍스트 입력 > 버튼
                if unused_radio_groups:
                    # 라디오 그룹별로 묶기
                    grouped_radios = {}
                    for radio_info in unused_radio_groups:
                        radio_elem = radio_info['element']
                        radio_name = radio_elem.get_attribute('name') or f"_ungrouped_{id(radio_elem)}"
                        if radio_name not in grouped_radios:
                            grouped_radios[radio_name] = []
                        grouped_radios[radio_name].append(radio_info)
                    
                    # 각 그룹을 별도의 질문으로 추가
                    for name, radios in grouped_radios.items():
                        if radios:
                            questions.append({'type': 'radio', 'question': f'선택 ({len(radios)}개 옵션)', 'options': radios})
                            print(f"  ✓ 라디오 버튼 발견 (질문 없음): {len(radios)}개 옵션")
                elif unused_checkbox_groups:
                    grouped_checkboxes = {}
                    for cb_info in unused_checkbox_groups:
                        cb_elem = cb_info['element']
                        cb_name = cb_elem.get_attribute('name') or f"_ungrouped_{id(cb_elem)}"
                        if cb_name not in grouped_checkboxes:
                            grouped_checkboxes[cb_name] = []
                        grouped_checkboxes[cb_name].append(cb_info)
                    
                    for name, cbs in grouped_checkboxes.items():
                        if cbs:
                            questions.append({'type': 'checkbox', 'question': f'선택 ({len(cbs)}개 옵션)', 'options': cbs})
                            print(f"  ✓ 체크박스 발견 (질문 없음): {len(cbs)}개 옵션")
                elif unused_text_inputs:
                    questions.append({'type': 'text', 'question': '텍스트 입력', 'inputs': unused_text_inputs})
                    print(f"  ✓ 텍스트 입력란 발견 (질문 없음): {len(unused_text_inputs)}개")
                elif unused_buttons:
                    # 다음/제출 버튼은 제외
                    filtered_buttons = [btn for btn in unused_buttons 
                                       if not any(kw in btn.get('label', '').lower() 
                                                 for kw in ['다음', 'next', '제출', 'submit', '확인', 'confirm'])]
                    if filtered_buttons:
                        questions.append({'type': 'button', 'question': '버튼 클릭', 'options': filtered_buttons})
                        print(f"  ✓ 버튼 발견 (질문 없음): {len(filtered_buttons)}개")
            
            return questions
        except Exception as e:
            print(f"파싱 오류: {e}")
            return []
    
    def _is_question_text(self, text: str) -> bool:
        """질문인지 판단 (AI 활용 가능)"""
        if not text or len(text) < 3:
            return False
        
        # AI 클라이언트가 있으면 AI로 판단
        if self.transformer.api_client:
            ai_result = self.transformer._is_question_with_api(text)
            if ai_result is not None:
                return ai_result
        
        # 규칙 기반 판단 (fallback)
        text_lower = text.lower()
        markers = ['?', '질문', '입력', '선택', '답변', '어디', '무엇', '언제', '누구', '어떻게', '얼마', '의견', '생각']
        question_patterns = [
            r'[A-Z]?Q\d+\.',  # Q1., Q2. 등
            r'^\d+[\.\)]\s*',  # 1. 2) 등
            r'.*\?',  # 물음표로 끝나는 경우
        ]
        has_question_marker = any(marker in text_lower for marker in markers)
        has_question_pattern = any(re.search(pattern, text, re.IGNORECASE) for pattern in question_patterns)
        
        return has_question_marker or has_question_pattern
    
    def _clean_question_text(self, text: str) -> str:
        """질문 텍스트 정리"""
        if not text:
            return ""
        text = re.sub(r'%[A-F0-9]{2}', '', text)  # URL 인코딩 제거
        text = re.sub(r'Powered by SurveyMonkey', '', text, flags=re.IGNORECASE)
        text = re.sub(r'이전/다음', '', text)
        text = re.sub(r'^\d+[\.\)]\s*', '', text)  # 번호 제거
        text = text.strip()
        # 질문 마커에서 자르기
        for marker in ['?', ':', '.']:
            idx = text.find(marker)
            if idx > 10:
                text = text[:idx+1]
        return text.strip()
    
    def _get_label_for_input(self, input_elem):
        """입력 필드의 라벨 찾기 (라디오/체크박스 포함)"""
        try:
            # 1. id 속성으로 label[for] 찾기
            inp_id = input_elem.get_attribute('id')
            if inp_id:
                try:
                    label = self.driver.find_element(By.CSS_SELECTOR, f"label[for='{inp_id}']")
                    text = label.text.strip()
                    if text and text != '옵션' and len(text) > 0:
                        return text
                except:
                    pass
            
            # 2. 부모 요소에서 라벨 찾기 (라디오/체크박스용)
            try:
                parent = input_elem.find_element(By.XPATH, "./..")
                
                # 부모가 label인 경우
                if parent.tag_name.lower() == 'label':
                    text = parent.text.strip()
                    if text and text != '옵션' and len(text) > 0:
                        return text
                
                # 부모 내에서 label 요소 찾기
                try:
                    labels = parent.find_elements(By.CSS_SELECTOR, "label")
                    for label in labels:
                        # 라디오 버튼과 연결된 label인지 확인
                        label_for = label.get_attribute('for')
                        if label_for == inp_id or not label_for:
                            text = label.text.strip()
                            if text and text != '옵션' and len(text) > 0 and len(text) < 100:
                                return text
                except:
                    pass
                
                # 부모 내에서 span, div 등에서 텍스트 찾기
                try:
                    text_elements = parent.find_elements(By.CSS_SELECTOR, "span, div, p, td")
                    for elem in text_elements:
                        text = elem.text.strip()
                        # input 요소가 아닌 텍스트만 있는 요소 찾기
                        if text and text != '옵션' and len(text) > 0 and len(text) < 100:
                            # input 요소를 포함하지 않는지 확인
                            if not elem.find_elements(By.CSS_SELECTOR, "input, select, textarea"):
                                return text
                except:
                    pass
                
                # 부모의 직접 텍스트 (다른 요소 제외)
                try:
                    parent_text = parent.text.strip()
                    # 자식 요소의 텍스트를 제외한 순수 텍스트 찾기
                    child_texts = []
                    for child in parent.find_elements(By.XPATH, "./*"):
                        child_text = child.text.strip()
                        if child_text:
                            child_texts.append(child_text)
                    
                    if parent_text:
                        # 자식 텍스트를 제외한 텍스트
                        for child_text in child_texts:
                            parent_text = parent_text.replace(child_text, '').strip()
                        if parent_text and len(parent_text) > 0 and len(parent_text) < 100:
                            return parent_text
                except:
                    pass
            except:
                pass
            
            # 3. aria-label 속성
            aria_label = input_elem.get_attribute('aria-label')
            if aria_label and aria_label != '옵션' and len(aria_label) > 0:
                return aria_label
            
            # 4. value 속성 (라디오 버튼의 경우)
            value = input_elem.get_attribute('value')
            if value and value != '옵션' and len(value) > 0 and len(value) < 100:
                return value
            
            # 5. 다음 형제 요소에서 텍스트 찾기
            try:
                following_sibling = input_elem.find_element(By.XPATH, "./following-sibling::*[1]")
                if following_sibling:
                    text = following_sibling.text.strip()
                    if text and text != '옵션' and len(text) > 0 and len(text) < 100:
                        return text
            except:
                pass
            
        except Exception as e:
            print(f"    라벨 찾기 오류: {e}")
            pass
        
        return '옵션'
    
    def _click_element(self, element):
        """요소 클릭"""
        try:
            self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
            time.sleep(0.3)
            self.driver.execute_script("arguments[0].click();", element)
            time.sleep(0.5)
            return True
        except:
            try:
                element.click()
                return True
            except:
                try:
                    ActionChains(self.driver).move_to_element(element).click().perform()
                    return True
                except:
                    return False
    
    def _fill_input(self, element, value, input_type='text'):
        """입력 필드에 값 입력 (입력 타입에 맞게 처리)"""
        try:
            self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
            time.sleep(0.3)
            
            if input_type == 'select':
                Select(element).select_by_index(0)
            else:
                # 숫자 입력란인 경우 숫자만 추출
                if input_type == 'number':
                    import re
                    numbers = re.findall(r'\d+', str(value))
                    if numbers:
                        value = numbers[0]
                    else:
                        # 숫자가 없으면 기본값 사용
                        value = "0"
                
                element.clear()
                
                # JavaScript로 먼저 시도 (특히 number 타입의 경우)
                try:
                    self.driver.execute_script(f"arguments[0].value = '{value}';", element)
                    # change 이벤트 발생
                    self.driver.execute_script("arguments[0].dispatchEvent(new Event('input', {{ bubbles: true }}));", element)
                    self.driver.execute_script("arguments[0].dispatchEvent(new Event('change', {{ bubbles: true }}));", element)
                    time.sleep(0.2)
                except:
                    # JavaScript 실패 시 일반 입력 시도
                    element.send_keys(value)
                    time.sleep(0.3)
            
            return True
        except Exception as e:
            print(f"입력 필드 채우기 실패: {e}")
            return False
    
    def _detect_page_change(self, before_url, before_source_hash):
        """페이지 변경 감지"""
        try:
            current_url = self.driver.current_url
            current_source = self.driver.page_source[:1000]  # 처음 1000자만 해시
            current_hash = hash(current_source)
            
            # URL이 변경되었거나 페이지 소스가 변경되었으면 페이지 변경으로 간주
            if current_url != before_url:
                return True
            if current_hash != before_source_hash:
                return True
            return False
        except:
            return False
    
    def _process_questions_on_current_page(self):
        """현재 페이지를 다시 파싱하고 AI 분석하여 질문 처리"""
        try:
            print(f"  현재 페이지 재파싱 및 AI 분석 시작...")
            
            # 페이지 다시 파싱
            questions = self._parse_page()
            print(f"  재파싱된 질문 수: {len(questions)}")
            
            if not questions:
                # 질문이 없으면 텍스트 입력란 찾기
                text_inputs = self.driver.find_elements(By.CSS_SELECTOR, 
                    "input[type='text'], input[type='email'], input[type='number'], input[type='tel'], textarea")
                for inp in text_inputs:
                    try:
                        if inp.is_displayed() and inp.is_enabled():
                            label = self._get_label_for_input(inp)
                            input_type = inp.get_attribute('type') or 'text'
                            if inp.tag_name == 'textarea':
                                input_type = 'textarea'
                            answer = self.transformer.generate_text_answer(label, input_type)
                            if self._fill_input(inp, answer, input_type):
                                print(f"    텍스트 입력 ({input_type}): {answer[:30]}")
                    except:
                        continue
                return True
            
            # 각 질문 처리
            answered_count = 0
            for q in questions:
                try:
                    question_text = q.get('question', '')
                    q_type = q.get('type', '')
                    
                    print(f"    질문: {question_text[:50]}...")
                    
                    if q_type == 'radio' or q_type == 'checkbox':
                        # DeepSeek으로 질문 타입 확인 및 답변 선택
                        options = q.get('options', [])
                        if options:
                            # DeepSeek으로 질문과 옵션 분석
                            if self.transformer.api_client and self.transformer.api_provider == 'deepseek':
                                print(f"      DeepSeek으로 질문 분석 중...")
                            
                            answer_idx = self.transformer.understand_question(question_text, options)
                            if answer_idx < len(options):
                                element = options[answer_idx].get('element')
                                if element:
                                    if self._click_element(element):
                                        answered_count += 1
                                        option_label = options[answer_idx].get('label', '옵션')
                                        print(f"      ✓ 선택: {option_label} (인덱스: {answer_idx + 1}/{len(options)})")
                    
                    elif q_type == 'text':
                        # 텍스트 입력 (DeepSeek으로 질문 타입 확인)
                        inputs = q.get('inputs', [])
                        for inp_info in inputs:
                            try:
                                element = inp_info.get('element')
                                input_type = inp_info.get('type', 'text')
                                label = inp_info.get('label', question_text)
                                
                                # DeepSeek으로 질문 타입 확인 및 답변 생성
                                if self.transformer.api_client and self.transformer.api_provider == 'deepseek':
                                    print(f"      DeepSeek으로 입력 타입 분석 중... ({input_type})")
                                
                                # 질문 텍스트와 라벨을 조합하여 더 정확한 답변 생성
                                combined_text = f"{question_text} {label}".strip()
                                answer = self.transformer.generate_text_answer(combined_text, input_type)
                                if self._fill_input(element, answer, input_type):
                                    answered_count += 1
                                    print(f"      ✓ 입력 ({input_type}): {answer[:30]}")
                            except Exception as e:
                                print(f"      입력 실패: {e}")
                                continue
                    
                    elif q_type == 'button':
                        # 버튼 클릭
                        buttons = q.get('options', [])
                        if buttons:
                            element = buttons[0].get('element')
                            if element:
                                if self._click_element(element):
                                    answered_count += 1
                                    print(f"      ✓ 클릭: {buttons[0].get('label', '버튼')}")
                    
                    time.sleep(0.5)
                except Exception as e:
                    print(f"      질문 처리 오류: {e}")
                    continue
            
            print(f"    처리 완료: {answered_count}개 답변")
            return answered_count > 0
        except Exception as e:
            print(f"  페이지 재처리 오류: {e}")
            return False
    
    def _click_next_button(self, max_retries=3):
        """다음 버튼 클릭 (페이지 변경 확인 포함, 재시도 및 재파싱 로직)"""
        try:
            # 현재 페이지 상태 저장
            before_url = self.driver.current_url
            before_source = self.driver.page_source[:1000]
            before_hash = hash(before_source)
            
            # 모든 버튼 찾기
            buttons = self.driver.find_elements(By.CSS_SELECTOR, 
                "button, input[type='button'], input[type='submit'], a[role='button'], "
                "a[class*='next'], a[class*='Next'], a[class*='continue'], "
                "button[class*='next'], button[class*='Next'], button[class*='continue']")
            
            # 우선순위 키워드
            keywords = ['다음', 'next', '제출', 'submit', '확인', 'confirm', 'continue', '계속', '진행']
            
            # 재시도 루프
            for retry in range(max_retries):
                if retry > 0:
                    print(f"  재시도 {retry}/{max_retries - 1}...")
                
                # 1단계: 키워드가 있는 버튼 찾기
                for btn in buttons:
                    try:
                        if not btn.is_displayed() or not btn.is_enabled():
                            continue
                        btn_text = (btn.text.strip() or btn.get_attribute('value') or btn.get_attribute('aria-label') or '').lower()
                        if any(kw in btn_text for kw in keywords):
                            print(f"  다음 버튼 발견: {btn_text}")
                            if self._click_element(btn):
                                time.sleep(2)
                                # 페이지 변경 확인
                                if self._detect_page_change(before_url, before_hash):
                                    print(f"  ✓ 페이지 변경 확인됨")
                                    return True
                                else:
                                    print(f"  ⚠️ 페이지 변경 없음")
                                    # 페이지 변경이 없으면 12초 대기 후 재파싱 및 재시도
                                    if retry < max_retries - 1:
                                        print(f"  12초 대기 후 재파싱 및 재시도...")
                                        time.sleep(12)
                                        
                                        # 현재 페이지를 다시 파싱하고 AI 분석하여 처리
                                        self._process_questions_on_current_page()
                                        
                                        # 버튼 다시 찾기 (페이지가 동적으로 변경될 수 있음)
                                        buttons = self.driver.find_elements(By.CSS_SELECTOR, 
                                            "button, input[type='button'], input[type='submit'], a[role='button'], "
                                            "a[class*='next'], a[class*='Next'], a[class*='continue'], "
                                            "button[class*='next'], button[class*='Next'], button[class*='continue']")
                                        before_url = self.driver.current_url
                                        before_source = self.driver.page_source[:1000]
                                        before_hash = hash(before_source)
                                        break  # 외부 루프로 돌아가서 재시도
                    except Exception as e:
                        print(f"  버튼 클릭 오류: {e}")
                        continue
                
                # 2단계: 모든 버튼 시도
                if retry == 0:
                    print(f"  키워드 버튼 실패, 모든 버튼 시도 중...")
                for btn in buttons:
                    try:
                        if btn.is_displayed() and btn.is_enabled():
                            btn_text = (btn.text.strip() or btn.get_attribute('value') or '').lower()
                            print(f"  버튼 시도: {btn_text[:30]}")
                            if self._click_element(btn):
                                time.sleep(2)
                                # 페이지 변경 확인
                                if self._detect_page_change(before_url, before_hash):
                                    print(f"  ✓ 페이지 변경 확인됨")
                                    return True
                                else:
                                    print(f"  ⚠️ 페이지 변경 없음")
                                    # 페이지 변경이 없으면 12초 대기 후 재파싱 및 재시도
                                    if retry < max_retries - 1:
                                        print(f"  12초 대기 후 재파싱 및 재시도...")
                                        time.sleep(12)
                                        
                                        # 현재 페이지를 다시 파싱하고 AI 분석하여 처리
                                        self._process_questions_on_current_page()
                                        
                                        # 버튼 다시 찾기
                                        buttons = self.driver.find_elements(By.CSS_SELECTOR, 
                                            "button, input[type='button'], input[type='submit'], a[role='button'], "
                                            "a[class*='next'], a[class*='Next'], a[class*='continue'], "
                                            "button[class*='next'], button[class*='Next'], button[class*='continue']")
                                        before_url = self.driver.current_url
                                        before_source = self.driver.page_source[:1000]
                                        before_hash = hash(before_source)
                                        break  # 외부 루프로 돌아가서 재시도
                    except:
                        continue
                
                # 재시도 전에 12초 대기 및 재파싱 (마지막 시도가 아니면)
                if retry < max_retries - 1:
                    print(f"  12초 대기 후 재파싱 및 재시도...")
                    time.sleep(12)
                    
                    # 현재 페이지를 다시 파싱하고 AI 분석하여 처리
                    self._process_questions_on_current_page()
                    
                    # 버튼 다시 찾기
                    buttons = self.driver.find_elements(By.CSS_SELECTOR, 
                        "button, input[type='button'], input[type='submit'], a[role='button'], "
                        "a[class*='next'], a[class*='Next'], a[class*='continue'], "
                        "button[class*='next'], button[class*='Next'], button[class*='continue']")
                    before_url = self.driver.current_url
                    before_source = self.driver.page_source[:1000]
                    before_hash = hash(before_source)
            
            print(f"  ✗ 다음 버튼을 찾을 수 없거나 페이지가 변경되지 않음 (최대 재시도 횟수 도달)")
            return False
        except Exception as e:
            print(f"  다음 버튼 클릭 오류: {e}")
            return False
    
    def fill_survey(self, paginated: bool = True):
        """설문조사 자동 작성 (반복 실행)"""
        if not self._init_driver():
            return False
        
        try:
            self.driver.get(self.url)
            time.sleep(5)
            
            page_num = 1
            max_pages = 50
            same_page_count = 0  # 같은 페이지에 머무는 횟수
            
            while page_num <= max_pages:
                print(f"\n=== 페이지 {page_num} ===")
                
                # 현재 페이지 상태 저장
                current_url = self.driver.current_url
                current_source = self.driver.page_source[:1000]
                current_hash = hash(current_source)
                
                # 페이지 파싱
                questions = self._parse_page()
                print(f"  파싱된 질문 수: {len(questions)}")
                
                if not questions:
                    # 질문이 없으면 텍스트 입력란과 버튼 찾기
                    text_inputs = self.driver.find_elements(By.CSS_SELECTOR, 
                        "input[type='text'], input[type='email'], input[type='number'], input[type='tel'], textarea")
                    filled_count = 0
                    for inp in text_inputs:
                        try:
                            if inp.is_displayed() and inp.is_enabled():
                                label = self._get_label_for_input(inp)
                                input_type = inp.get_attribute('type') or 'text'
                                if inp.tag_name == 'textarea':
                                    input_type = 'textarea'
                                answer = self.transformer.generate_text_answer(label, input_type)
                                if self._fill_input(inp, answer, input_type):
                                    filled_count += 1
                                    print(f"  텍스트 입력 ({input_type}): {answer[:30]}")
                        except:
                            continue
                    
                    if filled_count > 0:
                        time.sleep(1)
                    
                    # 버튼 클릭 시도
                    if paginated:
                        if self._click_next_button():
                            # 페이지 변경 확인
                            time.sleep(2)
                            if self._detect_page_change(current_url, current_hash):
                                page_num += 1
                                same_page_count = 0
                                time.sleep(2)
                                continue
                            else:
                                same_page_count += 1
                                if same_page_count >= 3:
                                    print(f"  ⚠️ 같은 페이지에 3번 머무름, 종료")
                                    break
                        else:
                            same_page_count += 1
                            if same_page_count >= 3:
                                print(f"  ⚠️ 다음 버튼을 찾을 수 없어 종료")
                                break
                    else:
                        break
                
                # 각 질문 처리
                answered_count = 0
                for q in questions:
                    try:
                        question_text = q.get('question', '')
                        q_type = q.get('type', '')
                        
                        print(f"  질문: {question_text[:50]}...")
                        
                        # DeepSeek으로 질문 타입 확인 (옵션)
                        options = q.get('options', [])
                        inputs = q.get('inputs', [])
                        if self.transformer.api_client and self.transformer.api_provider == 'deepseek':
                            deepseek_type = self.transformer._determine_question_type_with_deepseek(
                                question_text, options, inputs
                            )
                            if deepseek_type:
                                print(f"    DeepSeek 타입 판단: {deepseek_type}")
                        
                        print(f"    감지된 타입: {q_type}")
                        
                        if q_type == 'radio' or q_type == 'checkbox':
                            # DeepSeek으로 질문 타입 확인 및 답변 선택
                            if options:
                                # 질문 타입 출력
                                type_label = '라디오 버튼' if q_type == 'radio' else '체크박스'
                                print(f"    타입: {type_label} (객관식, {len(options)}개 옵션)")
                                
                                # DeepSeek으로 질문과 옵션 분석
                                if self.transformer.api_client and self.transformer.api_provider == 'deepseek':
                                    print(f"    DeepSeek으로 질문 분석 중...")
                                
                                answer_idx = self.transformer.understand_question(question_text, options)
                                if answer_idx < len(options):
                                    element = options[answer_idx].get('element')
                                    if element:
                                        if self._click_element(element):
                                            answered_count += 1
                                            option_label = options[answer_idx].get('label', '옵션')
                                            print(f"    ✓ 선택: {option_label} (인덱스: {answer_idx + 1}/{len(options)})")
                        
                        elif q_type == 'text':
                            # 텍스트 입력 (DeepSeek으로 질문 타입 확인)
                            print(f"    타입: 텍스트 입력란 (주관식, {len(inputs)}개 입력란)")
                            
                            for inp_info in inputs:
                                try:
                                    element = inp_info.get('element')
                                    input_type = inp_info.get('type', 'text')
                                    label = inp_info.get('label', question_text)
                                    
                                    # DeepSeek으로 질문 타입 확인 및 답변 생성
                                    if self.transformer.api_client and self.transformer.api_provider == 'deepseek':
                                        print(f"    DeepSeek으로 입력 타입 분석 중... ({input_type})")
                                    
                                    # 질문 텍스트와 라벨을 조합하여 더 정확한 답변 생성
                                    combined_text = f"{question_text} {label}".strip()
                                    answer = self.transformer.generate_text_answer(combined_text, input_type)
                                    if self._fill_input(element, answer, input_type):
                                        answered_count += 1
                                        print(f"    ✓ 입력 ({input_type}): {answer[:30]}")
                                except:
                                    continue
                        
                        elif q_type == 'button':
                            # 버튼 클릭
                            buttons = q.get('options', [])
                            if buttons:
                                element = buttons[0].get('element')
                                if element:
                                    if self._click_element(element):
                                        answered_count += 1
                                        print(f"    ✓ 클릭: {buttons[0].get('label', '버튼')}")
                        
                        time.sleep(0.5)
                    except Exception as e:
                        print(f"    질문 처리 오류: {e}")
                        continue
                
                print(f"  처리 완료: {answered_count}개 답변")
                
                # 다음 페이지로 이동
                if paginated:
                    if answered_count > 0 or len(questions) == 0:
                        # 답변을 했거나 질문이 없으면 다음 페이지로 이동 시도
                        print(f"  다음 페이지로 이동 시도...")
                        if self._click_next_button():
                            # 페이지 변경 확인
                            time.sleep(2)
                            if self._detect_page_change(current_url, current_hash):
                                print(f"  ✓ 페이지 {page_num} → {page_num + 1}로 이동 성공")
                                page_num += 1
                                same_page_count = 0
                                time.sleep(2)  # 페이지 로딩 대기
                            else:
                                same_page_count += 1
                                print(f"  ⚠️ 페이지 변경 없음 ({same_page_count}/3)")
                                if same_page_count >= 3:
                                    print(f"  ⚠️ 같은 페이지에 3번 머무름, 종료")
                                    break
                        else:
                            same_page_count += 1
                            if same_page_count >= 3:
                                print(f"  ⚠️ 다음 버튼을 찾을 수 없어 종료")
                                break
                    else:
                        print(f"  답변한 질문이 없어 종료")
                        break
                else:
                    break
            
            print(f"\n=== 설문조사 작성 완료 (총 {page_num} 페이지) ===")
            return True
        except Exception as e:
            print(f"설문조사 작성 오류: {e}")
            traceback.print_exc()
            return False
        finally:
            if self.driver:
                self.driver.quit()


# 호환성을 위한 래퍼 클래스
class SurveyAnalyzer:
    """호환성을 위한 래퍼"""
    def __init__(self, url: str):
        self.auto_fill = SurveyAutoFill(url)
    
    def analyze(self):
        """설문조사 분석"""
        if not self.auto_fill._init_driver():
            return []
        try:
            self.auto_fill.driver.get(self.auto_fill.url)
            time.sleep(5)
            questions = self.auto_fill._parse_page()
            result = []
            for q in questions:
                item = {'question': q.get('question', ''), 'type': q.get('type', '')}
                if q.get('options'):
                    item['options'] = [opt.get('label', '옵션') for opt in q['options']]
                if q.get('inputs'):
                    item['inputs'] = [inp.get('label', '입력') for inp in q['inputs']]
                result.append(item)
            return result
        finally:
            if self.auto_fill.driver:
                self.auto_fill.driver.quit()
