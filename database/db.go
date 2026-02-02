package database

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"log"

	_ "github.com/lib/pq"
	"github.com/pgvector/pgvector-go"
)

type DB struct {
	conn *sql.DB
}

type MediaEmbedding struct {
	ID             int
	FilePath       string
	FileType       string
	FileName       string
	FolderPath     string
	TextEmbedding  pgvector.Vector
	ImageEmbedding pgvector.Vector
	Transcription  string
	Summary        string
	Duration       float64
	Metadata       string
}

func NewDB(connStr string) (*DB, error) {
	conn, err := sql.Open("postgres", connStr)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to database: %w", err)
	}

	if err := conn.Ping(); err != nil {
		return nil, fmt.Errorf("failed to ping database: %w", err)
	}

	// pgvector 확장 활성화
	_, err = conn.Exec("CREATE EXTENSION IF NOT EXISTS vector")
	if err != nil {
		log.Printf("Warning: failed to create vector extension: %v", err)
	}

	// 테이블 생성
	if err := createTables(conn); err != nil {
		return nil, fmt.Errorf("failed to create tables: %w", err)
	}

	return &DB{conn: conn}, nil
}

func createTables(conn *sql.DB) error {
	schema := `
	CREATE EXTENSION IF NOT EXISTS vector;

	CREATE TABLE IF NOT EXISTS media_embeddings (
		id SERIAL PRIMARY KEY,
		file_path TEXT NOT NULL,
		file_type TEXT NOT NULL,
		file_name TEXT NOT NULL,
		folder_path TEXT NOT NULL,
		text_embedding vector(384),
		image_embedding vector(512),
		transcription TEXT,
		summary TEXT,
		duration FLOAT,
		metadata JSONB,
		created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
		updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
		UNIQUE(file_path)
	);

	CREATE INDEX IF NOT EXISTS idx_text_embedding ON media_embeddings 
	USING ivfflat (text_embedding vector_cosine_ops)
	WITH (lists = 100);

	CREATE INDEX IF NOT EXISTS idx_image_embedding ON media_embeddings 
	USING ivfflat (image_embedding vector_cosine_ops)
	WITH (lists = 100);

	CREATE INDEX IF NOT EXISTS idx_file_path ON media_embeddings(file_path);
	CREATE INDEX IF NOT EXISTS idx_file_type ON media_embeddings(file_type);
	`

	_, err := conn.Exec(schema)
	return err
}

func (db *DB) SaveEmbedding(embedding *MediaEmbedding) error {
	query := `
	INSERT INTO media_embeddings 
	(file_path, file_type, file_name, folder_path, text_embedding, image_embedding, transcription, summary, duration, metadata)
	VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
	ON CONFLICT (file_path) 
	DO UPDATE SET
		text_embedding = EXCLUDED.text_embedding,
		image_embedding = EXCLUDED.image_embedding,
		transcription = EXCLUDED.transcription,
		summary = EXCLUDED.summary,
		duration = EXCLUDED.duration,
		metadata = EXCLUDED.metadata,
		updated_at = CURRENT_TIMESTAMP
	`

	var textVec, imageVec interface{}
	// pgvector.Vector는 구조체이므로 빈 벡터인지 확인 필요
	// 빈 벡터는 nil로 전달하여 SQL NULL로 처리
	// 벡터의 길이를 확인하여 빈 벡터인지 판단
	if vec := embedding.TextEmbedding.Slice(); len(vec) > 0 {
		textVec = embedding.TextEmbedding
	}
	if vec := embedding.ImageEmbedding.Slice(); len(vec) > 0 {
		imageVec = embedding.ImageEmbedding
	}

	// Metadata를 JSONB로 변환
	var metadataJSON interface{}
	if embedding.Metadata != "" {
		// 이미 JSON 문자열인지 확인
		var jsonData interface{}
		if err := json.Unmarshal([]byte(embedding.Metadata), &jsonData); err != nil {
			// JSON이 아니면 빈 객체로 설정
			metadataJSON = nil
		} else {
			metadataJSON = embedding.Metadata
		}
	}

	_, err := db.conn.Exec(query,
		embedding.FilePath,
		embedding.FileType,
		embedding.FileName,
		embedding.FolderPath,
		textVec,
		imageVec,
		embedding.Transcription,
		embedding.Summary,
		embedding.Duration,
		metadataJSON,
	)

	return err
}

func (db *DB) SearchSimilar(queryEmbedding pgvector.Vector, limit int, fileType string) ([]*MediaEmbedding, error) {
	if db == nil || db.conn == nil {
		return nil, fmt.Errorf("database connection is not initialized")
	}

	var query string
	if fileType != "" {
		query = `
		SELECT id, file_path, file_type, file_name, folder_path, transcription, summary, duration, metadata
		FROM media_embeddings
		WHERE file_type = $1 AND text_embedding IS NOT NULL
		ORDER BY text_embedding <=> $2
		LIMIT $3
		`
	} else {
		// 전체 검색 시 text_embedding이 있는 모든 파일 검색
		// 이미지 파일도 BLIP 설명이 있으면 text_embedding이 생성됨
		query = `
		SELECT id, file_path, file_type, file_name, folder_path, transcription, summary, duration, metadata
		FROM media_embeddings
		WHERE text_embedding IS NOT NULL
		ORDER BY text_embedding <=> $1
		LIMIT $2
		`
	}

	var rows *sql.Rows
	var err error
	if fileType != "" {
		rows, err = db.conn.Query(query, fileType, queryEmbedding, limit)
	} else {
		rows, err = db.conn.Query(query, queryEmbedding, limit)
	}

	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []*MediaEmbedding
	for rows.Next() {
		var e MediaEmbedding
		var transcription, summary, metadata sql.NullString
		var duration sql.NullFloat64
		err := rows.Scan(
			&e.ID,
			&e.FilePath,
			&e.FileType,
			&e.FileName,
			&e.FolderPath,
			&transcription,
			&summary,
			&duration,
			&metadata,
		)
		if err != nil {
			return nil, err
		}
		// NULL 처리
		if transcription.Valid {
			e.Transcription = transcription.String
		} else {
			e.Transcription = ""
		}
		if summary.Valid {
			e.Summary = summary.String
		} else {
			e.Summary = ""
		}
		if duration.Valid {
			e.Duration = duration.Float64
		} else {
			e.Duration = 0
		}
		if metadata.Valid {
			e.Metadata = metadata.String
		} else {
			e.Metadata = ""
		}
		results = append(results, &e)
	}

	return results, nil
}

func (db *DB) Close() error {
	return db.conn.Close()
}

