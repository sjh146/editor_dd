package main

import (
	"editor_dd/database"
	"editor_dd/handlers"
	"editor_dd/services"
	"log"
	"os"

	"github.com/gin-gonic/gin"
)

func main() {
	// 다운로드 폴더 생성
	downloadFolder := "downloads"
	if err := os.MkdirAll(downloadFolder, 0755); err != nil {
		log.Fatalf("Failed to create downloads folder: %v", err)
	}

	// PostgreSQL 연결 (WSL 환경)
	// 환경변수에서 가져오거나 기본값 사용
	dbConnStr := os.Getenv("DATABASE_URL")
	if dbConnStr == "" {
		// 기본 연결 문자열 (WSL의 PostgreSQL)
		dbConnStr = "postgres://postgres:postgres@localhost:5432/editor_dd?sslmode=disable"
	}

	var db *database.DB
	var err error
	if dbConnStr != "" {
		db, err = database.NewDB(dbConnStr)
		if err != nil {
			log.Printf("Warning: Failed to connect to database: %v", err)
			log.Printf("Connection string: %s", dbConnStr)
			log.Printf("Embedding features will be disabled.")
			db = nil
		} else {
			log.Println("Database connected successfully")
		}
	}

	// ML 서비스 클라이언트 초기화
	mlService := services.NewMLService("http://localhost:5002")

	// 핸들러 초기화
	h := handlers.NewHandler(downloadFolder, mlService, db)

	// Gin 라우터 설정
	r := gin.Default()

	// 템플릿 함수 설정 (LoadHTMLGlob 전에 설정해야 함)
	r.SetFuncMap(map[string]interface{}{
		"hasSuffix": func(s, suffix string) bool {
			return len(s) >= len(suffix) && s[len(s)-len(suffix):] == suffix
		},
	})

	// 정적 파일 및 템플릿 설정
	r.Static("/static", "./static")
	r.LoadHTMLGlob("templates/*.html")

	// 라우트 설정
	r.GET("/", h.Index)
	r.POST("/download", h.Download)
	r.POST("/edit", h.Edit)
	r.POST("/extract_frame", h.ExtractFrame)
	r.POST("/add_tts_to_video", h.AddTTSToVideo)
	r.POST("/concat_videos", h.ConcatVideos)
	r.POST("/tts", h.TextToSpeech)
	r.POST("/analyze_video", h.AnalyzeVideo)
	r.GET("/downloads/*filename", h.DownloadFile)
	r.GET("/video_analysis/*filename", h.GetVideoAnalysis)
	
	// 임베딩 관련 라우트
	r.POST("/save_embedding", h.SaveEmbedding)
	r.GET("/search", h.SearchSimilar)

	// 서버 시작
	port := ":5001"
	log.Printf("Server starting on port %s", port)
	if err := r.Run(port); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}

