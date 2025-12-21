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
    
    def understand_question(self, question_text: str, options: list) -> int:
        """질문을 이해하고 답변 인덱스 반환"""
        if not question_text or not options:
            return random.randint(0, len(options) - 1) if options else 0
        
        # O, X 기호 기반 판단 (먼저 처리)
        o_x_result = self._interpret_o_x_symbols(options)
        if o_x_result is not None:
            return o_x_result
        
        # 모르는 문자/그림 감지
        if self._has_unreadable_content(question_text, options):
            return random.randint(0, len(options) - 1)
        
        # 키워드 기반 판단
        question_lower = question_text.lower()
        positive_keywords = ['만족', '좋', '긍정', '동의', '예', 'yes', '좋다', '좋아', 'satisfied', 'good', 'positive', '추천']
        negative_keywords = ['불만', '나쁜', '부정', '비동의', '아니요', 'no', '나쁘다', 'disappointed', 'bad', 'negative']
        
        if any(kw in question_lower for kw in positive_keywords):
            return len(options) - 1
        elif any(kw in question_lower for kw in negative_keywords):
            return 0
        else:
            return random.randint(0, len(options) - 1)
    
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
    
    def generate_text_answer(self, question_text: str) -> str:
        """텍스트 입력용 답변 생성"""
        if not question_text:
            return "답변"
        
        # API 클라이언트가 있으면 먼저 시도
        if self.api_client:
            api_answer = self._generate_text_with_api(question_text)
            if api_answer:
                return api_answer
        
        question_lower = question_text.lower()
        if '이름' in question_text or 'name' in question_lower:
            return "홍길동"
        elif '이메일' in question_text or 'email' in question_lower:
            return "test@example.com"
        elif '전화' in question_text or 'phone' in question_lower:
            return "010-1234-5678"
        elif '주소' in question_text or 'address' in question_lower:
            return "서울특별시 강남구 테헤란로 123"
        elif '생년월일' in question_text or 'birth' in question_lower:
            return "2000-01-01"
        elif '의견' in question_text or 'opinion' in question_lower:
            return "전반적으로 만족스럽고 좋은 서비스라고 생각합니다."
        else:
            return "답변" if len(question_text) < 20 else "설문조사에 참여하게 되어 기쁩니다."
    
    def _generate_text_with_api(self, question_text: str) -> str:
        """API를 사용하여 텍스트 답변 생성"""
        if not self.api_client:
            return None
        
        try:
            prompt = f"""다음 설문조사 질문에 적절한 한 문장 답변을 작성하세요. 간결하고 자연스러운 답변을 작성하세요.

질문: {question_text}

답변:"""

            response = self.api_client.chat.completions.create(
                model="gpt-3.5-turbo" if self.api_provider == 'openai' else "deepseek-chat",
                messages=[
                    {"role": "system", "content": "당신은 설문조사에 답변하는 사용자입니다. 질문에 맞는 간결하고 자연스러운 답변을 작성하세요."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API를 사용한 텍스트 생성 중 오류 발생: {e}")
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
    
    def _parse_page(self):
        """AI 기반 페이지 파싱 - 질문, 문항, 버튼, 입력란 찾기"""
        try:
            WebDriverWait(self.driver, 15).until(
                lambda d: d.execute_script('return document.readyState') == 'complete'
            )
            time.sleep(2)
            
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
            
            # 2. 라디오/체크박스 찾기
            radio_inputs = self.driver.find_elements(By.CSS_SELECTOR, "input[type='radio']")
            checkbox_inputs = self.driver.find_elements(By.CSS_SELECTOR, "input[type='checkbox']")
            
            radio_groups = {}
            for radio in radio_inputs:
                try:
                    name = radio.get_attribute('name') or ''
                    if name:
                        if name not in radio_groups:
                            radio_groups[name] = []
                        label = self._get_label_for_input(radio)
                        radio_groups[name].append({'element': radio, 'label': label})
                except:
                    continue
            
            checkbox_groups = {}
            for cb in checkbox_inputs:
                try:
                    name = cb.get_attribute('name') or ''
                    if name:
                        if name not in checkbox_groups:
                            checkbox_groups[name] = []
                        label = self._get_label_for_input(cb)
                        checkbox_groups[name].append({'element': cb, 'label': label})
                except:
                    continue
            
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
            
            # 5. 질문과 요소 매칭
            for question_elem, question_text in question_elements:
                try:
                    cleaned = self._clean_question_text(question_text)
                    if not cleaned or len(cleaned) < 3:
                        continue
                    
                    question_key = cleaned[:50]
                    if question_key in seen_questions:
                        continue
                    seen_questions.add(question_key)
                    
                    try:
                        container = question_elem.find_element(By.XPATH, "./..")
                    except:
                        container = None
                    
                    # 라디오 버튼 찾기
                    options = []
                    for name, radios in radio_groups.items():
                        try:
                            if not container or container.find_elements(By.CSS_SELECTOR, f"input[name='{name}']"):
                                options.extend(radios)
                        except:
                            pass
                    
                    # 체크박스 찾기
                    checkbox_options = []
                    for name, checkboxes in checkbox_groups.items():
                        try:
                            if not container or container.find_elements(By.CSS_SELECTOR, f"input[name='{name}']"):
                                checkbox_options.extend(checkboxes)
                        except:
                            pass
                    
                    # 텍스트 입력란 찾기
                    inputs = []
                    for text_inp in text_inputs:
                        try:
                            if not container or container.find_elements(By.CSS_SELECTOR, "input, textarea"):
                                inputs.append(text_inp)
                        except:
                            pass
                    
                    # 질문 타입 결정
                    if options:
                        questions.append({'type': 'radio', 'question': cleaned, 'options': options})
                    elif checkbox_options:
                        questions.append({'type': 'checkbox', 'question': cleaned, 'options': checkbox_options})
                    elif inputs:
                        questions.append({'type': 'text', 'question': cleaned, 'inputs': inputs})
                    elif buttons:
                        questions.append({'type': 'button', 'question': cleaned, 'options': buttons})
                except:
                    continue
            
            # 질문이 없지만 요소가 있는 경우
            if not questions:
                if text_inputs:
                    questions.append({'type': 'text', 'question': '텍스트 입력', 'inputs': text_inputs})
                elif buttons:
                    questions.append({'type': 'button', 'question': '버튼 클릭', 'options': buttons})
            
            return questions
        except Exception as e:
            print(f"파싱 오류: {e}")
            return []
    
    def _is_question_text(self, text: str) -> bool:
        """질문인지 판단"""
        if not text or len(text) < 3:
            return False
        text_lower = text.lower()
        markers = ['?', '질문', '입력', '선택', '답변', '어디', '무엇', '언제', '누구', '어떻게', '얼마']
        return any(marker in text_lower for marker in markers) or bool(re.search(r'[A-Z]?Q\d+\.', text, re.IGNORECASE))
    
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
        """입력 필드의 라벨 찾기"""
        try:
            inp_id = input_elem.get_attribute('id')
            if inp_id:
                try:
                    label = self.driver.find_element(By.CSS_SELECTOR, f"label[for='{inp_id}']")
                    text = label.text.strip()
                    if text and text != '옵션':
                        return text
                except:
                    pass
            
            try:
                parent = input_elem.find_element(By.XPATH, "./..")
                labels = parent.find_elements(By.CSS_SELECTOR, "label, span, div")
                for label in labels:
                    text = label.text.strip()
                    if text and text != '옵션' and len(text) < 50:
                        return text
            except:
                pass
            
            aria_label = input_elem.get_attribute('aria-label')
            if aria_label and aria_label != '옵션':
                return aria_label
        except:
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
        """입력 필드에 값 입력"""
        try:
            self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
            time.sleep(0.3)
            
            if input_type == 'select':
                Select(element).select_by_index(0)
            else:
                element.clear()
                element.send_keys(value)
                time.sleep(0.3)
            
            return True
        except:
            try:
                self.driver.execute_script(f"arguments[0].value = '{value}';", element)
                return True
            except:
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
                    "input[type='text'], input[type='email'], textarea")
                for inp in text_inputs:
                    try:
                        if inp.is_displayed() and inp.is_enabled():
                            label = self._get_label_for_input(inp)
                            answer = self.transformer.generate_text_answer(label)
                            if self._fill_input(inp, answer):
                                print(f"    텍스트 입력: {answer[:30]}")
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
                        # AI로 답변 선택
                        options = q.get('options', [])
                        if options:
                            answer_idx = self.transformer.understand_question(question_text, options)
                            if answer_idx < len(options):
                                element = options[answer_idx].get('element')
                                if element:
                                    if self._click_element(element):
                                        answered_count += 1
                                        print(f"      ✓ 선택: {options[answer_idx].get('label', '옵션')}")
                    
                    elif q_type == 'text':
                        # 텍스트 입력
                        inputs = q.get('inputs', [])
                        for inp_info in inputs:
                            try:
                                element = inp_info.get('element')
                                input_type = inp_info.get('type', 'text')
                                answer = self.transformer.generate_text_answer(question_text)
                                if self._fill_input(element, answer, input_type):
                                    answered_count += 1
                                    print(f"      ✓ 입력: {answer[:30]}")
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
                        "input[type='text'], input[type='email'], textarea")
                    filled_count = 0
                    for inp in text_inputs:
                        try:
                            if inp.is_displayed() and inp.is_enabled():
                                label = self._get_label_for_input(inp)
                                answer = self.transformer.generate_text_answer(label)
                                if self._fill_input(inp, answer):
                                    filled_count += 1
                                    print(f"  텍스트 입력: {answer[:30]}")
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
                        
                        if q_type == 'radio' or q_type == 'checkbox':
                            # AI로 답변 선택
                            options = q.get('options', [])
                            if options:
                                answer_idx = self.transformer.understand_question(question_text, options)
                                if answer_idx < len(options):
                                    element = options[answer_idx].get('element')
                                    if element:
                                        if self._click_element(element):
                                            answered_count += 1
                                            print(f"    ✓ 선택: {options[answer_idx].get('label', '옵션')}")
                        
                        elif q_type == 'text':
                            # 텍스트 입력
                            inputs = q.get('inputs', [])
                            for inp_info in inputs:
                                try:
                                    element = inp_info.get('element')
                                    input_type = inp_info.get('type', 'text')
                                    answer = self.transformer.generate_text_answer(question_text)
                                    if self._fill_input(element, answer, input_type):
                                        answered_count += 1
                                        print(f"    ✓ 입력: {answer[:30]}")
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
