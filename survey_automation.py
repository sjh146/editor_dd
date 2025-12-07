"""
설문조사 자동화 모듈
웹페이지를 분석하고 AI로 답변을 생성한 후 자동으로 체크합니다.
"""
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import json
import re
import time
import os


class SurveyAnalyzer:
    """설문조사 웹페이지를 분석하는 클래스"""
    
    def __init__(self, url: str):
        self.url = url
        self.html_content = None
        self.soup = None
        
    def fetch_page(self):
        """웹페이지를 가져옵니다"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(self.url, headers=headers, timeout=10)
            response.raise_for_status()
            self.html_content = response.text
            self.soup = BeautifulSoup(self.html_content, 'lxml')
            return True
        except Exception as e:
            print(f"페이지 가져오기 실패: {e}")
            return False
    
    def extract_survey_questions(self):
        """설문조사 질문과 선택지를 추출합니다"""
        questions = []
        
        if not self.soup:
            return questions
        
        # 다양한 형태의 설문조사 요소 찾기
        # 라디오 버튼 그룹
        radio_groups = self.soup.find_all(['input', 'div'], {'type': 'radio', 'class': re.compile(r'radio|choice|option', re.I)})
        
        # 체크박스 그룹
        checkbox_groups = self.soup.find_all(['input', 'div'], {'type': 'checkbox'})
        
        # select 요소
        selects = self.soup.find_all('select')
        
        # 일반적인 질문 형식 찾기
        question_labels = self.soup.find_all(['label', 'span', 'div', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'], 
                                             string=re.compile(r'.*\?|질문|Question', re.I))
        
        # 라디오 버튼과 체크박스를 기준으로 질문 추출
        form_elements = self.soup.find_all(['form', 'div', 'fieldset'])
        
        for form_elem in form_elements:
            # 라디오 버튼 그룹 찾기
            radios = form_elem.find_all('input', {'type': 'radio'})
            if radios:
                # 같은 name 속성을 가진 라디오 버튼들을 그룹화
                radio_groups_dict = {}
                for radio in radios:
                    name = radio.get('name', '')
                    if name:
                        if name not in radio_groups_dict:
                            radio_groups_dict[name] = []
                        
                        label = self._find_label_for_input(radio)
                        value = radio.get('value', '')
                        radio_groups_dict[name].append({
                            'label': label,
                            'value': value,
                            'element': radio
                        })
                
                # 각 그룹을 질문으로 변환
                for name, options in radio_groups_dict.items():
                    question_text = self._find_question_text_for_name(name, form_elem)
                    questions.append({
                        'type': 'radio',
                        'name': name,
                        'question': question_text,
                        'options': options
                    })
            
            # 체크박스 찾기
            checkboxes = form_elem.find_all('input', {'type': 'checkbox'})
            if checkboxes:
                checkbox_groups_dict = {}
                for checkbox in checkboxes:
                    name = checkbox.get('name', '')
                    if name:
                        if name not in checkbox_groups_dict:
                            checkbox_groups_dict[name] = []
                        
                        label = self._find_label_for_input(checkbox)
                        value = checkbox.get('value', '')
                        checkbox_groups_dict[name].append({
                            'label': label,
                            'value': value,
                            'element': checkbox
                        })
                
                for name, options in checkbox_groups_dict.items():
                    question_text = self._find_question_text_for_name(name, form_elem)
                    questions.append({
                        'type': 'checkbox',
                        'name': name,
                        'question': question_text,
                        'options': options
                    })
            
            # Select 드롭다운 찾기
            selects = form_elem.find_all('select')
            for select in selects:
                name = select.get('name', '')
                question_text = self._find_label_for_select(select)
                options = []
                for option in select.find_all('option'):
                    if option.get('value'):
                        options.append({
                            'label': option.text.strip(),
                            'value': option.get('value')
                        })
                
                if options:
                    questions.append({
                        'type': 'select',
                        'name': name,
                        'question': question_text,
                        'options': options
                    })
        
        return questions
    
    def _find_label_for_input(self, input_elem):
        """입력 요소에 대한 라벨을 찾습니다"""
        # label 태그의 for 속성으로 찾기
        input_id = input_elem.get('id', '')
        if input_id:
            label = self.soup.find('label', {'for': input_id})
            if label:
                return label.get_text(strip=True)
        
        # 부모 요소에서 라벨 찾기
        parent = input_elem.find_parent(['label', 'div', 'span'])
        if parent:
            text = parent.get_text(strip=True)
            if text and len(text) < 200:  # 너무 긴 텍스트는 제외
                return text
        
        # 다음 형제 요소에서 텍스트 찾기
        next_sibling = input_elem.find_next_sibling(['span', 'label', 'div'])
        if next_sibling:
            text = next_sibling.get_text(strip=True)
            if text and len(text) < 200:
                return text
        
        return input_elem.get('value', '옵션')
    
    def _find_label_for_select(self, select_elem):
        """select 요소에 대한 라벨을 찾습니다"""
        select_id = select_elem.get('id', '')
        if select_id:
            label = self.soup.find('label', {'for': select_id})
            if label:
                return label.get_text(strip=True)
        
        # 부모나 이전 형제에서 질문 찾기
        parent = select_elem.find_parent(['div', 'fieldset'])
        if parent:
            labels = parent.find_all(['label', 'span', 'p', 'h1', 'h2', 'h3', 'h4'])
            for label in labels:
                text = label.get_text(strip=True)
                if text and '?' in text:
                    return text
        
        return '선택 질문'
    
    def _find_question_text_for_name(self, name, container):
        """name 속성을 기반으로 질문 텍스트를 찾습니다"""
        # name 속성과 관련된 라벨이나 텍스트 찾기
        name_pattern = re.compile(name, re.I)
        
        # 컨테이너 내에서 질문 텍스트 찾기
        text_elements = container.find_all(['label', 'span', 'p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        for elem in text_elements:
            text = elem.get_text(strip=True)
            if text and ('?' in text or '질문' in text or len(text) > 10):
                # name과 관련이 있는지 확인
                if name_pattern.search(text) or name_pattern.search(str(elem)):
                    return text
        
        # 가장 가까운 라벨이나 텍스트 반환
        for elem in text_elements:
            text = elem.get_text(strip=True)
            if text and len(text) > 5 and len(text) < 200:
                return text
        
        return f"질문 ({name})"
    
    def analyze_survey(self):
        """설문조사를 분석합니다 (fetch_page + extract_survey_questions)"""
        if not self.fetch_page():
            return None
        questions = self.extract_survey_questions()
        return questions


class AISurveyAnswerer:
    """AI를 사용하여 설문조사 답변을 생성하는 클래스"""
    
    def __init__(self, use_openai=False, openai_api_key=None):
        self.use_openai = use_openai
        self.openai_api_key = openai_api_key
        if use_openai and openai_api_key:
            try:
                import openai
                self.openai_client = openai.OpenAI(api_key=openai_api_key)
            except ImportError:
                print("OpenAI 라이브러리가 설치되지 않았습니다. pip install openai 실행하세요.")
                self.use_openai = False
    
    def generate_answer(self, question: dict, context: str = ""):
        """질문에 대한 답변을 생성합니다"""
        question_text = question.get('question', '')
        options = question.get('options', [])
        question_type = question.get('type', 'radio')
        
        if not options:
            return None
        
        # 선택지 텍스트 추출
        option_texts = []
        for idx, option in enumerate(options):
            label = option.get('label', option.get('value', f'옵션 {idx+1}'))
            option_texts.append(f"{idx+1}. {label}")
        
        options_text = "\n".join(option_texts)
        
        # 프롬프트 생성
        prompt = f"""다음 설문조사 질문에 적절한 답변을 선택하세요.

질문: {question_text}

선택지:
{options_text}

위 질문에 대해 가장 적절한 답변의 번호만 숫자로 답변하세요 (예: 1, 2, 3 등).
다중 선택이 가능한 경우(체크박스)에는 적절한 모든 번호를 쉼표로 구분하여 답변하세요."""

        if context:
            prompt = f"맥락: {context}\n\n{prompt}"
        
        # AI로 답변 생성
        if self.use_openai:
            return self._generate_with_openai(prompt, question_type, len(options))
        else:
            return self._generate_simple_answer(question_text, options, question_type)
    
    def _generate_with_openai(self, prompt: str, question_type: str, num_options: int):
        """OpenAI API를 사용하여 답변 생성"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "당신은 설문조사 답변을 생성하는 AI입니다. 주어진 질문에 대해 가장 적절한 선택지를 선택하세요."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=50
            )
            
            answer_text = response.choices[0].message.content.strip()
            # 숫자 추출
            numbers = re.findall(r'\d+', answer_text)
            if numbers:
                if question_type == 'checkbox':
                    # 다중 선택 가능
                    selected = [int(n) - 1 for n in numbers if 1 <= int(n) <= num_options]
                    return selected if selected else [0]
                else:
                    # 단일 선택
                    num = int(numbers[0])
                    if 1 <= num <= num_options:
                        return num - 1
            return 0
        except Exception as e:
            print(f"OpenAI API 오류: {e}")
            return self._generate_simple_answer(prompt, [], question_type)
    
    def _generate_simple_answer(self, question_text: str, options: list, question_type: str):
        """간단한 규칙 기반 답변 생성 (OpenAI 없이)"""
        # 기본적으로 첫 번째 옵션 선택 (실제로는 더 똑똑한 로직 필요)
        # 긍정적인 답변을 선호하거나, 중립적인 답변 선택
        
        question_lower = question_text.lower()
        
        # 특정 키워드에 따라 답변 변경
        if any(word in question_lower for word in ['만족', '좋', '긍정', '동의']):
            # 긍정적인 질문이면 중간 이상 선택
            return min(len(options) - 1, len(options) // 2 + 1) if question_type == 'radio' else [min(len(options) - 1, len(options) // 2 + 1)]
        elif any(word in question_lower for word in ['불만', '나쁜', '부정', '반대']):
            # 부정적인 질문이면 낮은 값 선택
            return 0 if question_type == 'radio' else [0]
        else:
            # 중립적인 답변 (중간 선택)
            mid = len(options) // 2
            return mid if question_type == 'radio' else [mid]


class SurveyAutomation:
    """설문조사를 자동으로 작성하는 클래스"""
    
    def __init__(self, url: str, headless: bool = True, use_openai: bool = False, openai_api_key: str = None):
        self.url = url
        self.headless = headless
        self.driver = None
        self.analyzer = SurveyAnalyzer(url)
        self.answerer = AISurveyAnswerer(use_openai=use_openai, openai_api_key=openai_api_key)
        
    def initialize_driver(self):
        """Selenium 드라이버를 초기화합니다"""
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        
        try:
            # webdriver-manager 사용 시도 (설치되어 있다면)
            try:
                from webdriver_manager.chrome import ChromeDriverManager
                from selenium.webdriver.chrome.service import Service
                service = Service(ChromeDriverManager().install())
                self.driver = webdriver.Chrome(service=service, options=chrome_options)
            except ImportError:
                # webdriver-manager가 없으면 직접 ChromeDriver 사용
                self.driver = webdriver.Chrome(options=chrome_options)
            
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            return True
        except Exception as e:
            print(f"드라이버 초기화 실패: {e}")
            print("ChromeDriver가 설치되어 있는지 확인하세요.")
            print("또는 'pip install webdriver-manager'를 실행하여 자동 설치를 사용하세요.")
            return False
    
    def analyze_survey(self):
        """설문조사를 분석합니다"""
        if not self.analyzer.fetch_page():
            return None
        questions = self.analyzer.extract_survey_questions()
        return questions
    
    def fill_survey(self, questions: list):
        """설문조사를 자동으로 작성합니다"""
        if not self.driver:
            if not self.initialize_driver():
                return False
        
        try:
            self.driver.get(self.url)
            time.sleep(2)  # 페이지 로딩 대기
            
            results = []
            
            for question in questions:
                question_type = question.get('type')
                name = question.get('name')
                options = question.get('options', [])
                
                if not options:
                    continue
                
                # AI로 답변 생성
                answer_index = self.answerer.generate_answer(question)
                
                try:
                    if question_type == 'radio':
                        # 라디오 버튼 선택
                        if isinstance(answer_index, int) and 0 <= answer_index < len(options):
                            selected_option = options[answer_index]
                            value = selected_option.get('value', '')
                            
                            # 라디오 버튼 찾기 및 클릭
                            xpath = f"//input[@type='radio' and @name='{name}' and @value='{value}']"
                            radio = WebDriverWait(self.driver, 5).until(
                                EC.presence_of_element_located((By.XPATH, xpath))
                            )
                            self.driver.execute_script("arguments[0].click();", radio)
                            
                            results.append({
                                'question': question.get('question', ''),
                                'answer': selected_option.get('label', value),
                                'status': 'success'
                            })
                    
                    elif question_type == 'checkbox':
                        # 체크박스 선택 (다중 선택 가능)
                        if isinstance(answer_index, list):
                            for idx in answer_index:
                                if 0 <= idx < len(options):
                                    selected_option = options[idx]
                                    value = selected_option.get('value', '')
                                    
                                    xpath = f"//input[@type='checkbox' and @name='{name}' and @value='{value}']"
                                    checkbox = WebDriverWait(self.driver, 5).until(
                                        EC.presence_of_element_located((By.XPATH, xpath))
                                    )
                                    if not checkbox.is_selected():
                                        self.driver.execute_script("arguments[0].click();", checkbox)
                                    
                            results.append({
                                'question': question.get('question', ''),
                                'answer': [options[i].get('label', '') for i in answer_index],
                                'status': 'success'
                            })
                    
                    elif question_type == 'select':
                        # 드롭다운 선택
                        if isinstance(answer_index, int) and 0 <= answer_index < len(options):
                            selected_option = options[answer_index]
                            value = selected_option.get('value', '')
                            
                            xpath = f"//select[@name='{name}']"
                            select_elem = WebDriverWait(self.driver, 5).until(
                                EC.presence_of_element_located((By.XPATH, xpath))
                            )
                            from selenium.webdriver.support.ui import Select
                            select = Select(select_elem)
                            select.select_by_value(value)
                            
                            results.append({
                                'question': question.get('question', ''),
                                'answer': selected_option.get('label', value),
                                'status': 'success'
                            })
                    
                    time.sleep(0.5)  # 각 질문 사이 대기
                
                except (TimeoutException, NoSuchElementException) as e:
                    results.append({
                        'question': question.get('question', ''),
                        'answer': '요소를 찾을 수 없음',
                        'status': 'error',
                        'error': str(e)
                    })
            
            return results
        
        except Exception as e:
            print(f"설문조사 작성 중 오류: {e}")
            return False
    
    def submit_survey(self):
        """설문조사를 제출합니다 (선택사항)"""
        if not self.driver:
            return False
        
        try:
            # 제출 버튼 찾기
            submit_selectors = [
                "//button[@type='submit']",
                "//input[@type='submit']",
                "//button[contains(text(), '제출')]",
                "//button[contains(text(), 'Submit')]",
                "//button[contains(text(), '완료')]"
            ]
            
            for selector in submit_selectors:
                try:
                    submit_btn = self.driver.find_element(By.XPATH, selector)
                    self.driver.execute_script("arguments[0].click();", submit_btn)
                    time.sleep(2)
                    return True
                except:
                    continue
            
            return False
        except Exception as e:
            print(f"제출 중 오류: {e}")
            return False
    
    def close(self):
        """드라이버를 종료합니다"""
        if self.driver:
            self.driver.quit()

