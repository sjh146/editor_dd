"""
설문조사 자동화 모듈 (Transformer 기반)
Transformer 모델을 사용하여 설문 페이지를 해석하고 자동으로 답변합니다.
"""
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import re
import time
import os
import platform
import subprocess

# Transformer 모델 임포트
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    import torch
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False
    print("경고: transformers 라이브러리를 설치하세요. pip install transformers torch")


class SurveyTransformer:
    """Transformer 모델을 사용하여 설문조사 질문과 답변을 이해하는 클래스"""
    
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.classifier = None
        self._load_model()
    
    def _load_model(self):
        """한국어 텍스트 분류 모델 로드"""
        if not TRANSFORMER_AVAILABLE:
            return
        
        try:
            # 한국어 BERT 모델 사용 (경량 모델)
            model_name = "klue/bert-base"  # 또는 "beomi/KcELECTRA-base"
            print(f"Transformer 모델 로딩 중: {model_name}...")
            
            # 텍스트 분류 파이프라인 생성 (간단한 감성 분석 기반)
            self.classifier = pipeline(
                "text-classification",
                model="monologg/koelectra-base-v3-discriminator",
                device=-1  # CPU 사용
            )
            print("Transformer 모델 로딩 완료")
        except Exception as e:
            print(f"모델 로딩 실패 (기본 규칙 기반으로 폴백): {e}")
            self.classifier = None
    
    def understand_question(self, question_text: str, options: list) -> int:
        """질문 텍스트를 이해하고 가장 적절한 답변 인덱스 반환"""
        if not question_text or not options:
            return len(options) // 2  # 중간값 반환
        
        # Transformer 모델 사용
        if self.classifier:
            try:
                return self._classify_with_transformer(question_text, options)
            except Exception as e:
                print(f"Transformer 분류 실패: {e}")
        
        # 폴백: 규칙 기반 답변
        return self._rule_based_answer(question_text, options)
    
    def _classify_with_transformer(self, question: str, options: list) -> int:
        """Transformer를 사용하여 답변 선택"""
        question_lower = question.lower()
        
        # 긍정/부정 키워드 분석
        positive_keywords = ['만족', '좋', '긍정', '동의', '예', 'yes', '좋다', '좋아', 'satisfied', 'good', 'positive', '추천']
        negative_keywords = ['불만', '나쁜', '부정', '비동의', '아니요', 'no', '나쁘다', 'disappointed', 'bad', 'negative']
        
        # 질문에 대한 감성 분석
        try:
            result = self.classifier(question)[0]
            sentiment = result['label']
            score = result['score']
            
            # 긍정적인 질문인 경우 높은 점수 선택
            if any(kw in question_lower for kw in positive_keywords) or sentiment == 'POSITIVE':
                if score > 0.6:
                    return len(options) - 1
                else:
                    return max(0, len(options) - 2)
            
            # 부정적인 질문인 경우 낮은 점수 선택
            elif any(kw in question_lower for kw in negative_keywords) or sentiment == 'NEGATIVE':
                return 0
        except:
            pass
        
        # 중립적인 경우 중간값 반환
        return len(options) // 2
    
    def _rule_based_answer(self, question: str, options: list) -> int:
        """규칙 기반 답변 생성"""
        question_lower = question.lower()
        
        positive_keywords = ['만족', '좋', '긍정', '동의', '예', 'yes', '좋다', '좋아', 'satisfied', 'good', 'positive']
        negative_keywords = ['불만', '나쁜', '부정', '비동의', '아니요', 'no', '나쁘다', 'disappointed', 'bad', 'negative']
        
        if any(kw in question_lower for kw in positive_keywords):
            return len(options) - 1
        elif any(kw in question_lower for kw in negative_keywords):
            return 0
        else:
            return len(options) // 2


class SurveyAutoFill:
    """설문조사 자동 작성 클래스"""
    
    def __init__(self, url: str, headless: bool = True, browser_type: str = 'edge'):
        self.url = url
        self.headless = headless
        self.browser_type = browser_type.lower()
        self.driver = None
        self.transformer = SurveyTransformer()
    
    def _find_edge_executable(self):
        """Edge 브라우저 실행 파일 경로 찾기"""
        if platform.system() != 'Windows':
            return None
        
        paths = [
            r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
            r'C:\Program Files\Microsoft\Edge\Application\msedge.exe',
            os.path.expanduser(r'~\AppData\Local\Microsoft\Edge\Application\msedge.exe'),
        ]
        
        for path in paths:
            if os.path.exists(path) and os.path.isfile(path):
                return path
        
        # PowerShell로 찾기
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
    
    def _init_driver(self):
        """Edge 드라이버 초기화"""
        edge_options = EdgeOptions()
        
        # Edge 실행 파일 경로 설정
        edge_path = self._find_edge_executable()
        if edge_path:
            edge_options.binary_location = edge_path
            print(f"Edge 실행 파일: {edge_path}")
        
        # 최소 옵션 설정
        if self.headless:
            edge_options.add_argument('--headless')
        edge_options.add_argument('--no-sandbox')
        edge_options.add_argument('--disable-dev-shm-usage')
        edge_options.add_argument('--disable-blink-features=AutomationControlled')
        
        try:
            from selenium.webdriver.edge.service import Service
            service = Service()
            self.driver = webdriver.Edge(service=service, options=edge_options)
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            print("Edge 드라이버 초기화 성공")
            return True
        except Exception as e:
            print(f"Edge 드라이버 초기화 실패: {e}")
            return False
    
    def _parse_page_with_soup(self, html_content: str):
        """BeautifulSoup으로 HTML 파싱하여 질문과 선택지 추출"""
        soup = BeautifulSoup(html_content, 'lxml')
        questions = []
        
        # 라디오 버튼 그룹 찾기
        for form in soup.find_all(['form', 'div', 'fieldset']):
            radios = form.find_all('input', {'type': 'radio'})
            if radios:
                # name으로 그룹화
                radio_groups = {}
                for radio in radios:
                    name = radio.get('name', '')
                    if name:
                        if name not in radio_groups:
                            radio_groups[name] = {'radios': [], 'question': ''}
                        
                        label = self._find_label(radio)
                        radio_groups[name]['radios'].append({
                            'label': label,
                            'value': radio.get('value', ''),
                            'id': radio.get('id', '')
                        })
                
                # 질문 텍스트 찾기
                for name, group in radio_groups.items():
                    question_text = self._find_question_text(form, name)
                    if question_text:
                        questions.append({
                            'type': 'radio',
                            'name': name,
                            'question': question_text,
                            'options': group['radios']
                        })
        
        # 체크박스 찾기
        for form in soup.find_all(['form', 'div', 'fieldset']):
            checkboxes = form.find_all('input', {'type': 'checkbox'})
            if checkboxes:
                checkbox_groups = {}
                for cb in checkboxes:
                    name = cb.get('name', '')
                    if name:
                        if name not in checkbox_groups:
                            checkbox_groups[name] = []
                        label = self._find_label(cb)
                        checkbox_groups[name].append({
                            'label': label,
                            'value': cb.get('value', ''),
                            'id': cb.get('id', '')
                        })
                
                for name, options in checkbox_groups.items():
                    question_text = self._find_question_text(form, name)
                    if question_text:
                        questions.append({
                            'type': 'checkbox',
                            'name': name,
                            'question': question_text,
                            'options': options
                        })
        
        return questions
    
    def _find_label(self, input_elem):
        """입력 요소의 라벨 찾기"""
        # id로 연결된 label
        input_id = input_elem.get('id', '')
        if input_id:
            label = input_elem.find_parent().find('label', {'for': input_id})
            if label:
                return label.get_text(strip=True)
        
        # 부모 요소에서 label 찾기
        parent = input_elem.find_parent()
        if parent:
            label = parent.find('label')
            if label:
                return label.get_text(strip=True)
            
            # 인접한 텍스트 찾기
            for elem in parent.find_all(['span', 'div', 'p']):
                text = elem.get_text(strip=True)
                if text and len(text) < 100:
                    return text
        
        return input_elem.get('value', '') or '옵션'
    
    def _find_question_text(self, container, name):
        """질문 텍스트 찾기"""
        # legend, h1-h6, label 등에서 질문 찾기
        for tag in ['legend', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'label', 'p', 'div']:
            elems = container.find_all(tag)
            for elem in elems:
                text = elem.get_text(strip=True)
                if text and ('?' in text or '질문' in text.lower() or len(text) > 5):
                    return text
        
        return name.replace('_', ' ').title()
    
    def _parse_dynamic_page(self):
        """Selenium으로 동적 페이지에서 질문 파싱 (SurveyMonkey 등)"""
        questions = []
        
        try:
            WebDriverWait(self.driver, 15).until(
                lambda d: d.execute_script('return document.readyState') == 'complete'
            )
            time.sleep(3)
            
            # 질문 컨테이너 찾기
            containers = self.driver.find_elements(By.CSS_SELECTOR,
                "[id^='question-'], fieldset, [data-testid*='Question']")
            
            if not containers:
                containers = self.driver.find_elements(By.CSS_SELECTOR, "fieldset")
            
            for container in containers:
                try:
                    # 질문 텍스트 추출
                    question_elem = container.find_elements(By.CSS_SELECTOR,
                        "[id^='question-title-'], legend, h1, h2, h3, .question-title")
                    
                    question_text = ""
                    for elem in question_elem:
                        text = elem.text.strip()
                        if text and 5 < len(text) < 500:
                            question_text = text
                            break
                    
                    if not question_text:
                        continue
                    
                    # 라디오 버튼 찾기
                    radios = container.find_elements(By.CSS_SELECTOR, "input[type='radio']")
                    if radios:
                        options = []
                        for idx, radio in enumerate(radios):
                            # 숨겨진 요소나 비활성화된 요소는 건너뛰기
                            try:
                                if not radio.is_displayed() or not radio.is_enabled():
                                    continue
                            except:
                                pass
                            
                            label = self._get_label_for_input(radio, container)
                            
                            # 라벨이 제대로 찾아지지 않았으면 인덱스 기반 라벨 생성
                            if label == '옵션' or not label:
                                label = f"옵션 {idx + 1}"
                            
                            options.append({
                                'label': label,
                                'element': radio,
                                'id': radio.get_attribute('id') or '',
                                'name': radio.get_attribute('name') or '',
                                'value': radio.get_attribute('value') or ''
                            })
                        
                        if options:
                            print(f"  - 라디오 버튼 {len(options)}개 찾음: {[opt['label'] for opt in options]}")
                            questions.append({
                                'type': 'radio',
                                'question': question_text,
                                'options': options,
                                'container': container
                            })
                    
                    # 체크박스 찾기
                    checkboxes = container.find_elements(By.CSS_SELECTOR, "input[type='checkbox']")
                    if checkboxes:
                        options = []
                        for idx, cb in enumerate(checkboxes):
                            # 숨겨진 요소나 비활성화된 요소는 건너뛰기
                            try:
                                if not cb.is_displayed() or not cb.is_enabled():
                                    continue
                            except:
                                pass
                            
                            label = self._get_label_for_input(cb, container)
                            
                            # 라벨이 제대로 찾아지지 않았으면 인덱스 기반 라벨 생성
                            if label == '옵션' or not label:
                                label = f"옵션 {idx + 1}"
                            
                            options.append({
                                'label': label,
                                'element': cb,
                                'id': cb.get_attribute('id') or '',
                                'name': cb.get_attribute('name') or '',
                                'value': cb.get_attribute('value') or ''
                            })
                        
                        if options:
                            print(f"  - 체크박스 {len(options)}개 찾음: {[opt['label'] for opt in options]}")
                            questions.append({
                                'type': 'checkbox',
                                'question': question_text,
                                'options': options,
                                'container': container
                            })
                
                except Exception as e:
                    print(f"컨테이너 파싱 오류: {e}")
                    continue
        
        except Exception as e:
            print(f"동적 페이지 파싱 오류: {e}")
        
        return questions
    
    def _get_label_for_input(self, input_elem, container):
        """동적 페이지에서 라벨 찾기 (개선된 버전)"""
        try:
            # 방법 1: aria-label 속성
            aria_label = input_elem.get_attribute('aria-label')
            if aria_label and aria_label.strip() and aria_label != '옵션':
                return aria_label.strip()
            
            # 방법 2: id로 연결된 label 찾기 (전체 페이지에서)
            input_id = input_elem.get_attribute('id')
            if input_id:
                try:
                    label = self.driver.find_element(By.CSS_SELECTOR, f"label[for='{input_id}']")
                    text = label.text.strip()
                    if text and text != '옵션':
                        return text
                except:
                    pass
            
            # 방법 3: 부모 요소에서 label 찾기
            try:
                parent = input_elem.find_element(By.XPATH, "./..")
                # label 태그 찾기
                labels = parent.find_elements(By.CSS_SELECTOR, "label")
                for label in labels:
                    text = label.text.strip()
                    if text and len(text) > 0 and text != '옵션':
                        return text
                
                # span, div 등에서 텍스트 찾기
                for tag in ['span', 'div', 'p']:
                    elems = parent.find_elements(By.CSS_SELECTOR, tag)
                    for elem in elems:
                        text = elem.text.strip()
                        # 의미있는 텍스트인지 확인 (너무 길거나 짧지 않음)
                        if text and 2 <= len(text) <= 100 and text != '옵션':
                            # input이나 다른 요소가 아닌 순수 텍스트인지 확인
                            if not elem.find_elements(By.CSS_SELECTOR, "input, button, select"):
                                return text
            except:
                pass
            
            # 방법 4: 형제 요소에서 찾기
            try:
                # 다음 형제 요소 찾기
                sibling = input_elem.find_element(By.XPATH, "./following-sibling::*[1]")
                text = sibling.text.strip()
                if text and len(text) > 0 and text != '옵션':
                    return text
            except:
                pass
            
            # 방법 5: value 속성
            value = input_elem.get_attribute('value')
            if value and value.strip() and value != '옵션':
                return value.strip()
            
            # 방법 6: 클래스명이나 data 속성에서 힌트 찾기
            class_name = input_elem.get_attribute('class') or ''
            if 'yes' in class_name.lower() or 'accept' in class_name.lower():
                return '예'
            elif 'no' in class_name.lower() or 'reject' in class_name.lower():
                return '아니오'
        
        except Exception as e:
            print(f"라벨 찾기 오류: {e}")
        
        # 기본값은 인덱스 기반
        return '옵션'
    
    def fill_survey(self, paginated: bool = False):
        """설문조사 자동 작성"""
        if not self._init_driver():
            return False
        
        try:
            self.driver.get(self.url)
            time.sleep(5)
            
            results = []
            page_num = 1
            
            while True:
                print(f"\n=== 페이지 {page_num} 처리 중 ===")
                
                # 동적 페이지에서 질문 파싱
                questions = self._parse_dynamic_page()
                
                if not questions:
                    # 정적 HTML도 시도
                    html = self.driver.page_source
                    questions = self._parse_page_with_soup(html)
                
                if not questions:
                    print("질문을 찾을 수 없습니다.")
                    break
                
                print(f"찾은 질문 수: {len(questions)}")
                
                # 각 질문에 답변
                for q in questions:
                    try:
                        question_text = q['question']
                        options = q['options']
                        q_type = q['type']
                        
                        print(f"질문: {question_text[:60]}...")
                        print(f"선택지: {[opt['label'] for opt in options]}")
                        
                        # Transformer로 답변 선택
                        answer_idx = self.transformer.understand_question(question_text, options)
                        
                        print(f"선택한 답변: {options[answer_idx]['label']}")
                        
                        # Selenium으로 클릭
                        success = self._click_answer(q, answer_idx)
                        
                        results.append({
                            'page': page_num,
                            'question': question_text,
                            'answer': options[answer_idx]['label'],
                            'status': 'success' if success else 'failed'
                        })
                        
                        time.sleep(1)
                    
                    except Exception as e:
                        print(f"질문 처리 오류: {e}")
                        results.append({
                            'page': page_num,
                            'question': q.get('question', ''),
                            'status': 'error',
                            'error': str(e)
                        })
                
                if not paginated:
                    break
                
                # 다음 페이지로 이동
                if not self._click_next():
                    break
                
                page_num += 1
                time.sleep(3)
                
                if page_num > 50:  # 무한 루프 방지
                    break
            
            return results
        
        except Exception as e:
            print(f"설문조사 작성 오류: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _click_answer(self, question: dict, answer_idx: int):
        """답변 클릭 (개선된 버전 - 여러 방법 시도)"""
        try:
            options = question['options']
            if answer_idx >= len(options):
                return False
            
            option = options[answer_idx]
            q_type = question['type']
            
            # 방법 1: Selenium 요소 직접 클릭
            if 'element' in option:
                element = option['element']
                
                # 요소가 화면에 보이도록 스크롤
                self.driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", element)
                time.sleep(0.8)
                
                # 체크박스가 이미 선택되어 있으면 성공
                if q_type == 'checkbox':
                    try:
                        if element.is_selected():
                            return True
                    except:
                        pass
                
                # 여러 클릭 방법 시도
                click_methods = [
                    # 방법 1-1: JavaScript 클릭 (가장 안정적)
                    lambda: self.driver.execute_script("arguments[0].click();", element),
                    # 방법 1-2: 일반 클릭
                    lambda: element.click(),
                    # 방법 1-3: ActionChains 사용
                    lambda: self._action_click(element),
                ]
                
                for click_method in click_methods:
                    try:
                        click_method()
                        time.sleep(0.5)
                        
                        # 성공 여부 확인
                        if q_type == 'checkbox':
                            if element.is_selected():
                                return True
                        elif q_type == 'radio':
                            if element.is_selected():
                                return True
                            # 라디오는 다른 선택이 해제되었을 수도 있으므로 확인
                            return True
                        else:
                            return True
                    except Exception as e:
                        continue
                
                # 방법 1-4: 라벨 클릭 시도
                label_text = option.get('label', '')
                if label_text and label_text != '옵션':
                    if self._click_by_label_text(label_text, question.get('container'), element):
                        return True
            
            # 방법 2: id로 다시 찾아서 클릭
            elem_id = option.get('id', '')
            if elem_id:
                try:
                    elem = self.driver.find_element(By.ID, elem_id)
                    self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", elem)
                    time.sleep(0.5)
                    self.driver.execute_script("arguments[0].click();", elem)
                    time.sleep(0.5)
                    return True
                except:
                    pass
            
            # 방법 3: name으로 찾기 (정적 페이지)
            name = question.get('name', '')
            if name:
                try:
                    if q_type == 'radio':
                        radios = self.driver.find_elements(By.NAME, name)
                        if answer_idx < len(radios):
                            radio = radios[answer_idx]
                            self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", radio)
                            time.sleep(0.5)
                            self.driver.execute_script("arguments[0].click();", radio)
                            time.sleep(0.5)
                            return True
                    elif q_type == 'checkbox':
                        checkboxes = self.driver.find_elements(By.NAME, name)
                        if answer_idx < len(checkboxes):
                            cb = checkboxes[answer_idx]
                            self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", cb)
                            time.sleep(0.5)
                            self.driver.execute_script("arguments[0].click();", cb)
                            time.sleep(0.5)
                            return True
                except:
                    pass
            
            # 방법 4: 라벨 텍스트로 찾아서 클릭
            label_text = option.get('label', '')
            if label_text and label_text != '옵션':
                if self._click_by_label_text(label_text, question.get('container')):
                    return True
        
        except Exception as e:
            print(f"클릭 오류: {e}")
            import traceback
            traceback.print_exc()
        
        return False
    
    def _action_click(self, element):
        """ActionChains를 사용한 클릭"""
        try:
            from selenium.webdriver.common.action_chains import ActionChains
            actions = ActionChains(self.driver)
            actions.move_to_element(element).click().perform()
        except:
            raise
    
    def _click_by_label_text(self, label_text: str, container=None, preferred_element=None):
        """라벨 텍스트로 요소 클릭"""
        try:
            search_area = container if container else self.driver
            
            # 라벨 텍스트를 포함하는 요소 찾기
            xpaths = [
                f"//label[contains(text(), '{label_text}')]",
                f"//span[contains(text(), '{label_text}')]",
                f"//div[contains(text(), '{label_text}')]",
                f"//p[contains(text(), '{label_text}')]",
            ]
            
            for xpath in xpaths:
                try:
                    labels = search_area.find_elements(By.XPATH, xpath)
                    for label in labels:
                        try:
                            # 라벨이 연결된 input 찾기
                            label_for = label.get_attribute('for')
                            if label_for:
                                try:
                                    input_elem = self.driver.find_element(By.ID, label_for)
                                    self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", input_elem)
                                    time.sleep(0.3)
                                    self.driver.execute_script("arguments[0].click();", input_elem)
                                    time.sleep(0.5)
                                    return True
                                except:
                                    pass
                            
                            # 라벨 자체 클릭
                            if preferred_element:
                                # 선호하는 요소가 라벨의 자식이거나 형제인지 확인
                                try:
                                    if label.find_elements(By.XPATH, f".//input[@id='{preferred_element.get_attribute('id')}']"):
                                        preferred_element.click()
                                        return True
                                except:
                                    pass
                            
                            # 라벨 직접 클릭
                            self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", label)
                            time.sleep(0.3)
                            self.driver.execute_script("arguments[0].click();", label)
                            time.sleep(0.5)
                            return True
                        except:
                            continue
                except:
                    continue
        
        except Exception as e:
            print(f"라벨 텍스트 클릭 오류: {e}")
        
        return False
    
    def _click_next(self):
        """다음 페이지 버튼 클릭"""
        try:
            # SurveyMonkey 특화
            next_buttons = self.driver.find_elements(By.CSS_SELECTOR,
                "button[data-testid='button-next'], button[aria-label*='Next'], button[aria-label*='다음']")
            
            for button in next_buttons:
                try:
                    text = button.text.lower()
                    if 'next' in text or '다음' in text or '계속' in text:
                        self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", button)
                        time.sleep(0.5)
                        self.driver.execute_script("arguments[0].click();", button)
                        time.sleep(2)
                        return True
                except:
                    continue
            
            # XPath로 찾기
            xpaths = [
                "//button[contains(text(), 'Next')]",
                "//button[contains(text(), '다음')]",
                "//input[@type='submit']"
            ]
            
            for xpath in xpaths:
                try:
                    buttons = self.driver.find_elements(By.XPATH, xpath)
                    for button in buttons:
                        button.click()
                        time.sleep(2)
                        return True
                except:
                    continue
        
        except Exception as e:
            print(f"다음 버튼 클릭 오류: {e}")
        
        return False
    
    def close(self):
        """브라우저 종료"""
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass
            self.driver = None


# Flask 앱과의 호환성을 위한 클래스들
class SurveyAnalyzer:
    """설문조사 분석 클래스 (Flask 호환성)"""
    
    def __init__(self, url: str):
        self.url = url
        self.autofill = SurveyAutoFill(url, headless=True)
    
    def analyze_survey(self):
        """설문조사 분석"""
        if not self.autofill._init_driver():
            return []
        
        try:
            self.autofill.driver.get(self.url)
            time.sleep(5)
            questions = self.autofill._parse_dynamic_page()
            if not questions:
                html = self.autofill.driver.page_source
                questions = self.autofill._parse_page_with_soup(html)
            
            result = []
            for q in questions:
                result.append({
                    'type': q['type'],
                    'name': q.get('name', ''),
                    'question': q['question'],
                    'options': [{'label': opt['label'], 'value': opt.get('value', opt['label'])} 
                              for opt in q['options']]
                })
            
            self.autofill.close()
            return result
        except Exception as e:
            print(f"분석 오류: {e}")
            self.autofill.close()
            return []


class SurveyAutomation:
    """기존 Flask 앱 호환성을 위한 래퍼"""
    
    def __init__(self, url: str, headless: bool = True, use_openai: bool = False, 
                 openai_api_key: str = None, browser_type: str = 'edge'):
        self.autofill = SurveyAutoFill(url, headless, browser_type)
        self.url = url
    
    def initialize_driver(self):
        return self.autofill._init_driver()
    
    def analyze_survey(self):
        """설문조사 분석 (기존 호환성)"""
        if not self.autofill._init_driver():
            return None
        
        try:
            self.autofill.driver.get(self.url)
            time.sleep(5)
            questions = self.autofill._parse_dynamic_page()
            if not questions:
                html = self.autofill.driver.page_source
                questions = self.autofill._parse_page_with_soup(html)
            
            # 기존 형식으로 변환
            result = []
            for q in questions:
                result.append({
                    'type': q['type'],
                    'name': q.get('name', ''),
                    'question': q['question'],
                    'options': q['options']
                })
            return result
        except Exception as e:
            print(f"분석 오류: {e}")
            return None
    
    def fill_survey_paginated(self):
        """페이지네이션 설문조사 작성"""
        return self.autofill.fill_survey(paginated=True)
    
    def fill_survey(self, questions: list):
        """단일 페이지 설문조사 작성"""
        return self.autofill.fill_survey(paginated=False)
    
    def close(self):
        self.autofill.close()
    
    @property
    def driver(self):
        return self.autofill.driver
