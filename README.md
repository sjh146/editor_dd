# 편집 조수 챗봇 학습 가이드

이 문서는 편집 조수 챗봇을 학습시키는 다양한 방법을 설명합니다.

## 목차

1. [학습 방법 개요](#학습-방법-개요)
2. [Q&A 쌍 학습](#qa-쌍-학습)
3. [예제 대화 학습](#예제-대화-학습)
4. [피드백 학습](#피드백-학습)
5. [자동 학습](#자동-학습)
6. [학습 데이터 관리](#학습-데이터-관리)
7. [API 사용 예시](#api-사용-예시)
8. [학습 팁](#학습-팁)

---

## 학습 방법 개요

챗봇은 다음 4가지 방법으로 학습할 수 있습니다:

1. **Q&A 쌍 학습**: 질문과 답변을 직접 가르치기
2. **예제 대화 학습**: 대화 예제를 추가하여 Few-shot Learning
3. **피드백 학습**: 응답에 대한 피드백을 주어 개선
4. **자동 학습**: 모든 대화가 자동으로 저장되어 학습 데이터로 활용

---

## Q&A 쌍 학습

가장 효과적인 학습 방법입니다. 질문과 답변을 직접 가르치면, 비슷한 질문이 들어왔을 때 학습된 답변을 참고합니다.

### 📝 Q&A 쌍 학습이란?

Q&A 쌍 학습은 **질문(Question)**과 **답변(Answer)**을 쌍으로 저장하여, 나중에 비슷한 질문이 들어왔을 때 저장된 답변을 참고하여 응답하는 방식입니다.

**예시:**
- 질문: "Flask 앱을 만드는 방법은?"
- 답변: "Flask 앱을 만들려면..."

이렇게 학습하면, 사용자가 "Flask 앱 만들기 도와줘"라고 물어봤을 때 학습된 답변을 참고하여 더 정확한 응답을 할 수 있습니다.

### 🎯 단계별 가이드

#### 방법 1: 웹 UI에서 학습하기 (가장 쉬움) ⭐ 추천

**단계별 설명:**

1. **Flask 앱 실행**
   ```bash
   python app.py
   ```

2. **브라우저에서 접속**
   - 주소: `http://localhost:5001`
   - "편집 조수" 탭 클릭

3. **학습 패널 열기**
   - "📚 챗봇 학습하기" 버튼 클릭
   - 학습 패널이 펼쳐집니다

4. **Q&A 학습하기**
   - "Q&A 학습" 탭이 기본으로 선택되어 있습니다
   - **질문 입력**: 예) "Flask 라우트 만드는 방법?"
   - **답변 입력**: 예) "Flask에서 라우트는 @app.route 데코레이터로 만듭니다..."
   - "Q&A 학습 저장" 버튼 클릭
   - ✅ 성공 메시지가 표시됩니다!

5. **학습 확인**
   - "📊 학습 통계" 버튼을 클릭하여 학습된 Q&A 개수 확인

**실제 예시:**

```
질문 입력란: "Python으로 파일 읽는 방법?"
답변 입력란: 
파일을 읽는 방법은 다음과 같습니다:

```python
with open("file.txt", "r", encoding="utf-8") as f:
    content = f.read()
    print(content)
```

이 코드는 file.txt 파일을 읽어서 내용을 출력합니다.
```

→ "Q&A 학습 저장" 클릭 → ✅ "학습 완료! 총 1개의 Q&A가 저장되었습니다."

#### 방법 2: Python 코드로 학습하기

```python
import requests

# 1. Flask 앱이 실행 중인지 확인 (http://localhost:5001)

# 2. Q&A 쌍 학습
response = requests.post(
    'http://localhost:5001/chatbot/learn/qa',
    json={
        'question': 'Flask 라우트 만드는 방법?',
        'answer': '''Flask에서 라우트는 @app.route 데코레이터로 만듭니다.

예시:
```python
from flask import Flask
app = Flask(__name__)

@app.route("/")
def index():
    return "Hello World"

if __name__ == '__main__':
    app.run()
```

이렇게 하면 "/" 경로로 접속했을 때 "Hello World"가 표시됩니다.'''
    }
)

# 3. 결과 확인
result = response.json()
print(result)
# 출력: {'success': True, 'message': 'Q&A 쌍이 학습되었습니다.', 'total_qa': 1}
```

#### 방법 3: cURL로 학습하기

```bash
curl -X POST http://localhost:5001/chatbot/learn/qa \
  -H "Content-Type: application/json" \
  -d "{
    \"question\": \"Python으로 파일 읽는 방법?\",
    \"answer\": \"파일을 읽는 방법:\\n\\n```python\\nwith open(\\\"file.txt\\\", \\\"r\\\") as f:\\n    content = f.read()\\n    print(content)\\n```\"
  }"
```

### 📋 실제 사용 예시

#### 예시 1: 기본적인 Q&A 학습

```python
import requests

# 질문과 답변 정의
qa_data = {
    'question': 'Python 리스트 정렬하는 방법?',
    'answer': '''Python에서 리스트를 정렬하는 방법은 두 가지가 있습니다:

1. sorted() 함수 사용 (새 리스트 반환):
```python
numbers = [3, 1, 4, 1, 5]
sorted_numbers = sorted(numbers)
print(sorted_numbers)  # [1, 1, 3, 4, 5]
```

2. list.sort() 메서드 사용 (원본 수정):
```python
numbers = [3, 1, 4, 1, 5]
numbers.sort()
print(numbers)  # [1, 1, 3, 4, 5]
```'''
}

# 학습 요청
response = requests.post('http://localhost:5001/chatbot/learn/qa', json=qa_data)
print(response.json())
```

#### 예시 2: 여러 Q&A 한 번에 학습하기

```python
import requests

qa_list = [
    {
        'question': 'Flask에서 POST 요청 받는 방법?',
        'answer': 'Flask에서 POST 요청을 받으려면:\n\n```python\nfrom flask import request\n\n@app.route("/submit", methods=["POST"])\ndef submit():\n    data = request.form.get("key")\n    return f"받은 데이터: {data}"\n```'
    },
    {
        'question': 'JSON 파일 읽는 방법?',
        'answer': 'JSON 파일을 읽는 방법:\n\n```python\nimport json\n\nwith open("data.json", "r") as f:\n    data = json.load(f)\n```'
    },
    {
        'question': 'HTML 폼 만드는 방법?',
        'answer': 'HTML 폼 예시:\n\n```html\n<form action="/submit" method="post">\n  <input type="text" name="username">\n  <input type="password" name="password">\n  <button type="submit">제출</button>\n</form>\n```'
    }
]

# 각 Q&A를 순차적으로 학습
for qa in qa_list:
    response = requests.post('http://localhost:5001/chatbot/learn/qa', json=qa)
    print(f"학습 완료: {qa['question']}")

# 학습 통계 확인
stats = requests.get('http://localhost:5001/chatbot/learn/stats').json()
print(f"총 {stats['knowledge_base']['qa_pairs']}개의 Q&A가 학습되었습니다.")
```

### ✅ 학습 확인 방법

학습이 제대로 되었는지 확인하려면:

```python
import requests

# 1. 학습 통계 확인
stats = requests.get('http://localhost:5001/chatbot/learn/stats').json()
print(f"Q&A 쌍 개수: {stats['knowledge_base']['qa_pairs']}개")

# 2. 학습 데이터 내보내기
export_data = requests.get('http://localhost:5001/chatbot/learn/export').json()
print("학습된 Q&A 목록:")
for i, qa in enumerate(export_data['knowledge_base']['qa_pairs'], 1):
    print(f"{i}. Q: {qa['question']}")
    print(f"   A: {qa['answer'][:50]}...")
```

### 🔍 학습 효과 확인

학습한 Q&A가 실제로 사용되는지 확인:

1. 챗봇에 학습한 질문과 비슷한 질문을 해보세요
2. 챗봇이 학습한 답변을 참고하여 응답하는지 확인하세요

예를 들어, "Flask 라우트 만드는 방법?"을 학습했다면:
- "Flask 라우트 어떻게 만들어?" → 학습된 답변 참고
- "Flask에서 경로 설정하는 방법?" → 학습된 답변 참고

### 💡 학습 팁

1. **구체적인 질문**: "코드 작성해줘"보다 "Python으로 CSV 파일 읽는 코드 작성해줘"가 좋습니다
2. **상세한 답변**: 코드 예시, 설명, 주석을 포함하세요
3. **다양한 표현**: 같은 내용을 다양한 질문으로 학습하세요
   - "Flask 라우트 만드는 방법?"
   - "Flask에서 경로 설정하는 방법?"
   - "Flask 라우트 정의하는 방법?"

### API 엔드포인트 상세

```
POST /chatbot/learn/qa
Content-Type: application/json

{
    "question": "질문 내용",
    "answer": "답변 내용"
}
```

**응답:**
```json
{
    "success": true,
    "message": "Q&A 쌍이 학습되었습니다.",
    "total_qa": 1
}
```

---

## 예제 대화 학습

Few-shot Learning 방식으로, 대화 예제를 추가하면 비슷한 상황에서 참고하여 응답을 생성합니다.

### API 엔드포인트

```
POST /chatbot/learn/example
```

### 요청 형식

```json
{
    "user": "Python으로 파일 읽는 코드 작성해줘",
    "assistant": "파일을 읽는 Python 코드입니다:\n\n```python\nwith open('file.txt', 'r', encoding='utf-8') as f:\n    content = f.read()\n    print(content)\n```\n\n이 코드는 file.txt 파일을 읽어서 내용을 출력합니다."
}
```

### Python 예시

```python
import requests

response = requests.post('http://localhost:5001/chatbot/learn/example', json={
    'user': 'HTML 폼 만들기 도와줘',
    'assistant': 'HTML 폼 예시:\n\n```html\n<form action="/submit" method="post">\n  <input type="text" name="username" placeholder="사용자명">\n  <input type="password" name="password" placeholder="비밀번호">\n  <button type="submit">제출</button>\n</form>\n```'
})

print(response.json())
```

---

## 피드백 학습

챗봇의 응답에 대한 피드백을 주면, 자동으로 학습 데이터에 반영됩니다. 특히 "bad" 피드백과 개선 제안을 함께 주면 지식 베이스에 자동으로 추가됩니다.

### API 엔드포인트

```
POST /chatbot/learn/feedback
```

### 요청 형식

```json
{
    "user_message": "코드 작성해줘",
    "assistant_response": "기존 응답 내용",
    "feedback": "bad",
    "improvement": "더 자세한 설명과 예제를 포함해주세요. 주석도 추가해주세요."
}
```

### 피드백 타입

- `"good"`: 좋은 응답
- `"bad"`: 나쁜 응답 (개선 필요)
- `"improve"`: 개선 제안

### Python 예시

```python
import requests

# 좋은 응답 피드백
requests.post('http://localhost:5001/chatbot/learn/feedback', json={
    'user_message': '파일 읽기 코드 작성해줘',
    'assistant_response': 'with open("file.txt", "r") as f:\n    print(f.read())',
    'feedback': 'good'
})

# 나쁜 응답 피드백 (개선 제안 포함)
requests.post('http://localhost:5001/chatbot/learn/feedback', json={
    'user_message': 'Flask 앱 만들기',
    'assistant_response': '기존 응답',
    'feedback': 'bad',
    'improvement': 'Flask 앱을 만들 때는 다음과 같이 해야 합니다:\n1. Flask 임포트\n2. app 객체 생성\n3. 라우트 정의\n4. 실행 코드 추가'
})
```

---

## 자동 학습

모든 대화는 자동으로 `chatbot_learning_data.json` 파일에 저장됩니다. 최근 1000개의 대화가 저장되며, 이를 통해 챗봇의 응답 패턴을 분석할 수 있습니다.

### 저장 위치

- 파일: `chatbot_learning_data.json`
- 형식: JSON
- 최대 저장 개수: 1000개 대화

---

## 학습 데이터 관리

### 학습 통계 조회

현재 학습 상태를 확인할 수 있습니다.

#### API 엔드포인트

```
GET /chatbot/learn/stats
```

#### 응답 예시

```json
{
    "success": true,
    "knowledge_base": {
        "qa_pairs": 15,
        "examples": 8
    },
    "learning_data": {
        "conversations": 234,
        "feedback": 12
    }
}
```

### 학습 데이터 내보내기

학습 데이터를 백업하거나 다른 곳으로 옮길 수 있습니다.

#### API 엔드포인트

```
GET /chatbot/learn/export
```

#### 응답 예시

```json
{
    "success": true,
    "knowledge_base": {
        "qa_pairs": [...],
        "examples": [...]
    },
    "learning_data": {
        "conversations": [...],
        "feedback": [...]
    }
}
```

### 학습 데이터 가져오기

이전에 내보낸 학습 데이터를 다시 불러올 수 있습니다.

#### API 엔드포인트

```
POST /chatbot/learn/import
```

#### 요청 형식

```json
{
    "knowledge_base": {
        "qa_pairs": [...],
        "examples": [...]
    },
    "learning_data": {
        "conversations": [...],
        "feedback": [...]
    }
}
```

---

## API 사용 예시

### 전체 학습 워크플로우 예시

```python
import requests

BASE_URL = 'http://localhost:5001'

# 1. Q&A 쌍 학습
qa_data = {
    'question': 'Python 리스트 정렬 방법?',
    'answer': '리스트 정렬 방법:\n1. sorted() 함수 사용 (새 리스트 반환)\n2. list.sort() 메서드 사용 (원본 수정)'
}
requests.post(f'{BASE_URL}/chatbot/learn/qa', json=qa_data)

# 2. 예제 대화 학습
example_data = {
    'user': 'JSON 파일 읽기 코드 작성해줘',
    'assistant': '```python\nimport json\n\nwith open("data.json", "r") as f:\n    data = json.load(f)\n```'
}
requests.post(f'{BASE_URL}/chatbot/learn/example', json=example_data)

# 3. 학습 통계 확인
stats = requests.get(f'{BASE_URL}/chatbot/learn/stats').json()
print(f"Q&A 쌍: {stats['knowledge_base']['qa_pairs']}개")
print(f"예제: {stats['knowledge_base']['examples']}개")

# 4. 학습 데이터 백업
backup = requests.get(f'{BASE_URL}/chatbot/learn/export').json()
with open('backup.json', 'w', encoding='utf-8') as f:
    import json
    json.dump(backup, f, ensure_ascii=False, indent=2)
```

---

## 학습 팁

### 1. 효과적인 Q&A 작성

- **구체적인 질문**: 모호한 질문보다 구체적인 질문이 좋습니다
  - ❌ "코드 작성해줘"
  - ✅ "Python으로 CSV 파일을 읽고 데이터를 처리하는 코드 작성해줘"

- **상세한 답변**: 답변에 예제 코드와 설명을 포함하세요
  - ✅ 코드 예시
  - ✅ 단계별 설명
  - ✅ 주석 포함

### 2. 예제 대화 작성

- **다양한 상황**: 다양한 상황의 예제를 추가하세요
- **자연스러운 대화**: 실제 대화처럼 자연스럽게 작성하세요
- **구체적인 요청**: 사용자의 요청이 구체적일수록 좋습니다

### 3. 피드백 활용

- **즉시 피드백**: 응답이 마음에 들지 않으면 즉시 피드백을 주세요
- **구체적인 개선 제안**: "더 좋게"보다 "예제 코드 추가해주세요"처럼 구체적으로
- **긍정적 피드백**: 좋은 응답에도 피드백을 주면 더 좋은 응답을 유도합니다

### 4. 학습 데이터 관리

- **정기적 백업**: 학습 데이터를 정기적으로 백업하세요
- **데이터 정리**: 오래되거나 잘못된 데이터는 제거하세요
- **점진적 학습**: 한 번에 많은 데이터를 넣기보다 조금씩 추가하세요

---

## 학습 데이터 파일 구조

### chatbot_knowledge.json

```json
{
    "qa_pairs": [
        {
            "question": "질문 내용",
            "answer": "답변 내용",
            "timestamp": "2026-01-05T23:00:00"
        }
    ],
    "examples": [
        {
            "user": "사용자 메시지",
            "assistant": "조수 응답",
            "timestamp": "2026-01-05T23:00:00"
        }
    ]
}
```

### chatbot_learning_data.json

```json
{
    "conversations": [
        {
            "user": "사용자 메시지",
            "assistant": "조수 응답",
            "timestamp": "2026-01-05T23:00:00"
        }
    ],
    "feedback": [
        {
            "user_message": "사용자 메시지",
            "assistant_response": "조수 응답",
            "feedback": "good",
            "improvement": "개선 제안",
            "timestamp": "2026-01-05T23:00:00"
        }
    ]
}
```

---

## 문제 해결

### 학습 데이터가 저장되지 않을 때

1. 파일 권한 확인: `chatbot_knowledge.json`과 `chatbot_learning_data.json` 파일에 쓰기 권한이 있는지 확인
2. 디스크 공간 확인: 디스크 공간이 충분한지 확인
3. 로그 확인: 서버 로그에서 오류 메시지 확인

### 학습 효과가 없을 때

1. 충분한 데이터: 최소 10개 이상의 Q&A 쌍이나 예제 추가
2. 질문 유사성: 학습한 질문과 실제 질문이 유사한지 확인
3. 프롬프트 확인: 시스템 프롬프트가 적절한지 확인

---

## 추가 정보

- 챗봇은 학습 데이터를 기반으로 응답을 생성합니다
- 학습 데이터가 많을수록 더 정확한 응답을 생성합니다
- 피드백을 통해 지속적으로 개선할 수 있습니다

문의사항이나 문제가 있으면 이슈를 등록해주세요.

