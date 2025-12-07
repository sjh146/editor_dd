# AI 설문조사 자동화 기능 사용 가이드

## 개요

이 기능은 설문조사 웹페이지를 AI로 분석하고 자동으로 답변을 체크하는 기능입니다. 
**본인이 만든 설문조사를 테스트하는 용도로 사용하세요.**

## 필요한 라이브러리 설치

```bash
pip install selenium beautifulsoup4 requests lxml openai
```

## ChromeDriver 설치

Selenium을 사용하기 위해 ChromeDriver가 필요합니다.

### Windows:
1. Chrome 브라우저 버전 확인 (chrome://version/)
2. https://chromedriver.chromium.org/downloads 에서 해당 버전 다운로드
3. 다운로드한 chromedriver.exe를 PATH에 추가하거나 Python 스크립트와 같은 폴더에 저장

### 자동 설치 (권장):
```bash
# webdriver-manager 설치
pip install webdriver-manager
```

그리고 `survey_automation.py`를 수정하여 webdriver-manager를 사용하도록 변경할 수 있습니다.

## 사용 방법

### 1. 웹 인터페이스 사용

1. Flask 서버 실행: `python app.py`
2. 브라우저에서 `http://localhost:5001` 접속
3. "🤖 AI 설문조사 자동화" 섹션으로 이동
4. 설문조사 URL 입력
5. "📋 설문조사 분석" 버튼 클릭하여 질문 확인
6. "✏️ AI로 자동 작성" 버튼 클릭하여 자동으로 답변 체크

### 2. 옵션 설정

- **OpenAI API 사용**: 더 정확한 답변을 위해 OpenAI API 사용 가능 (선택사항)
  - OpenAI API 키를 입력하거나 환경변수 `OPENAI_API_KEY` 설정
  - 비어있으면 기본 규칙 기반 답변 생성
  
- **백그라운드 모드**: 브라우저 창을 숨기고 실행 (기본값: 켜짐)

- **자동 제출**: 답변 작성 후 자동으로 제출 버튼 클릭 (선택사항)

## 기능 설명

### 1. 설문조사 분석
- 웹페이지를 읽어서 다음 요소들을 찾습니다:
  - 라디오 버튼 (단일 선택)
  - 체크박스 (다중 선택)
  - 드롭다운 (select)
- 각 질문과 선택지를 추출합니다.

### 2. AI 답변 생성
- **OpenAI 사용 시**: GPT-3.5-turbo 모델로 질문을 분석하여 가장 적절한 답변 선택
- **기본 모드**: 키워드 기반 규칙으로 답변 생성
  - "만족", "좋", "긍정" 등의 키워드가 있으면 중간 이상 선택
  - "불만", "나쁜", "부정" 등의 키워드가 있으면 낮은 값 선택
  - 그 외에는 중간 값 선택

### 3. 자동 작성
- Selenium을 사용하여 브라우저를 제어
- 각 질문에 대해 AI가 선택한 답변을 자동으로 체크
- 결과를 화면에 표시

## 지원하는 설문조사 형식

- 일반 HTML 폼 (radio, checkbox, select)
- 표준 웹 폼 요소
- 대부분의 설문조사 플랫폼 (Google Forms, Typeform 등은 구조가 복잡할 수 있음)

## 문제 해결

### ChromeDriver 오류
```
드라이버 초기화 실패
```
**해결**: ChromeDriver를 설치하고 PATH에 추가하거나, webdriver-manager를 사용하세요.

### 설문조사를 찾을 수 없음
```
설문조사를 찾을 수 없습니다
```
**해결**: 
- URL이 올바른지 확인
- 페이지가 로그인이 필요한지 확인
- JavaScript로 동적 생성되는 요소는 제한적으로 지원

### 요소를 찾을 수 없음
일부 질문에서 "요소를 찾을 수 없음" 오류가 발생할 수 있습니다.
**원인**: 
- 페이지 구조가 표준적이지 않음
- 동적으로 생성되는 요소
- 복잡한 JavaScript 프레임워크 사용

**해결**:
- 브라우저 개발자 도구로 실제 HTML 구조 확인
- `survey_automation.py`의 선택자 로직 개선 필요할 수 있음

## 코드 사용 예제

### Python 코드로 직접 사용

```python
from survey_automation import SurveyAutomation

# 설문조사 자동화 객체 생성
automation = SurveyAutomation(
    url='https://example.com/survey',
    headless=True,  # 브라우저 숨김
    use_openai=True,  # OpenAI 사용
    openai_api_key='your-api-key'  # 또는 환경변수 사용
)

# 설문조사 분석
questions = automation.analyze_survey()
print(f"찾은 질문 수: {len(questions)}")

# 자동으로 답변 작성
results = automation.fill_survey(questions)

# 자동 제출 (선택사항)
automation.submit_survey()

# 브라우저 종료
automation.close()
```

## 주의사항

1. **윤리적 사용**: 본인이 만든 설문조사를 테스트하는 용도로만 사용하세요.
2. **서버 부하**: 대량의 요청을 보내면 서버에 부하를 줄 수 있습니다.
3. **JavaScript**: 복잡한 JavaScript로 만들어진 설문조사는 제한적으로 지원될 수 있습니다.
4. **개인정보**: OpenAI API를 사용하는 경우 API 키를 안전하게 관리하세요.

## 향후 개선 사항

- 더 많은 설문조사 플랫폼 지원
- 더 정교한 HTML 구조 분석
- 답변 패턴 학습 기능
- 스크린샷 저장 기능
- 여러 번 실행하여 다양한 답변 생성

## 라이선스 및 책임

이 도구는 교육 및 테스트 목적으로 제작되었습니다. 사용자는 이 도구의 사용에 대한 모든 책임을 집니다.

