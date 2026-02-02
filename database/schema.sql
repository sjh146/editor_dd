-- pgvector 확장 활성화
CREATE EXTENSION IF NOT EXISTS vector;

-- 미디어 파일 임베딩 저장 테이블
CREATE TABLE IF NOT EXISTS media_embeddings (
    id SERIAL PRIMARY KEY,
    file_path TEXT NOT NULL,
    file_type TEXT NOT NULL, -- 'video', 'image', 'audio'
    file_name TEXT NOT NULL,
    folder_path TEXT NOT NULL, -- 'video', 'tts', 'frame', 'analysis'
    
    -- 임베딩 벡터들
    text_embedding vector(384), -- 텍스트 임베딩 (음성 인식 결과)
    image_embedding vector(512), -- 이미지 임베딩 (CLIP 또는 BLIP)
    
    -- 메타데이터
    transcription TEXT, -- 음성 인식 텍스트
    summary TEXT, -- 요약
    duration FLOAT, -- 동영상/오디오 길이 (초)
    metadata JSONB, -- 추가 메타데이터
    
    -- 타임스탬프
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- 인덱스
    UNIQUE(file_path)
);

-- 벡터 검색을 위한 인덱스 생성
CREATE INDEX IF NOT EXISTS idx_text_embedding ON media_embeddings 
USING ivfflat (text_embedding vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_image_embedding ON media_embeddings 
USING ivfflat (image_embedding vector_cosine_ops)
WITH (lists = 100);

-- 파일 경로 인덱스
CREATE INDEX IF NOT EXISTS idx_file_path ON media_embeddings(file_path);
CREATE INDEX IF NOT EXISTS idx_file_type ON media_embeddings(file_type);

