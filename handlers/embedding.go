package handlers

import (
	"editor_dd/database"
	"editor_dd/services"
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/pgvector/pgvector-go"
)

func (h *Handler) SaveEmbedding(c *gin.Context) {
	fileType := c.PostForm("file_type") // 'video', 'image', 'audio'
	filename := c.PostForm("filename")
	folderPath := c.PostForm("folder_path") // 'video', 'tts', 'frame', 'analysis'

	if fileType == "" || filename == "" {
		c.JSON(http.StatusBadRequest, gin.H{
			"success": false,
			"error":   "file_type and filename are required",
		})
		return
	}

	if folderPath == "" {
		// 기본값 설정
		switch fileType {
		case "video":
			folderPath = "video"
		case "audio":
			folderPath = "tts"
		case "image":
			folderPath = "frame"
		default:
			folderPath = ""
		}
	}

	// 이미지 파일은 frame 또는 analysis 폴더에 있을 수 있음
	var filePath string
	if fileType == "image" {
		// frame과 analysis 폴더에서 찾기
		possiblePaths := []string{
			filepath.Join(h.downloadFolder, "frame", filename),
			filepath.Join(h.downloadFolder, "analysis", filename),
		}
		for _, path := range possiblePaths {
			if _, err := os.Stat(path); err == nil {
				filePath = path
				folderPath = filepath.Base(filepath.Dir(path))
				break
			}
		}
		if filePath == "" {
			c.JSON(http.StatusNotFound, gin.H{
				"success": false,
				"error":   "Image file not found",
			})
			return
		}
	} else {
		filePath = filepath.Join(h.downloadFolder, folderPath, filename)
		if _, err := os.Stat(filePath); os.IsNotExist(err) {
			c.JSON(http.StatusNotFound, gin.H{
				"success": false,
				"error":   "File not found",
			})
			return
		}
	}

	// Python ML 서비스를 통해 임베딩 생성
	embeddingService := services.NewEmbeddingService("http://localhost:5002")

	// 동영상/오디오의 경우 음성 인식 텍스트 가져오기
	var text string
	var summary string
	if fileType == "video" || fileType == "audio" {
		// 분석 결과에서 transcription 가져오기
		analysisPath := filepath.Join(h.downloadFolder, "analysis")
		entries, _ := filepath.Glob(filepath.Join(analysisPath, "*.json"))
		
		// 파일명 기반으로 분석 결과 찾기
		baseName := strings.TrimSuffix(filename, filepath.Ext(filename))
		for _, entry := range entries {
			// JSON 파일 읽기
			data, err := os.ReadFile(entry)
			if err != nil {
				continue
			}
			
			var analysisData map[string]interface{}
			if err := json.Unmarshal(data, &analysisData); err != nil {
				continue
			}
			
			// 동영상 파일명이 일치하는지 확인
			if videoFilename, ok := analysisData["video_filename"].(string); ok {
				if videoFilename == filename || strings.Contains(videoFilename, baseName) {
					if trans, ok := analysisData["transcription"].(string); ok {
						text = trans
					}
					if summ, ok := analysisData["transcription_summary"].(string); ok {
						summary = summ
					}
					break
				}
			}
		}
		
		// 분석 결과가 없으면 직접 음성 인식 수행
		if text == "" && (fileType == "video" || fileType == "audio") {
			// Python ML 서비스를 통해 동영상/오디오 분석
			// 오디오 파일도 analyze_video 엔드포인트로 전송 (Python 서비스에서 처리)
			result, err := h.mlService.AnalyzeVideo(filePath)
			if err == nil {
				if trans, ok := result["transcription"].(string); ok {
					text = trans
				}
				if summ, ok := result["transcription_summary"].(string); ok {
					summary = summ
				}
			} else {
				// 분석 실패 시 오류 메시지
				c.JSON(http.StatusBadRequest, gin.H{
					"success": false,
					"error":   "Failed to analyze audio/video. Please analyze the file first or ensure it contains audio.",
				})
				return
			}
		}
		
		// 텍스트가 여전히 없으면 오류
		if text == "" && (fileType == "video" || fileType == "audio") {
			c.JSON(http.StatusBadRequest, gin.H{
				"success": false,
				"error":   "No transcription found. Please analyze the file first using 'AI Video Analysis'.",
			})
			return
		}
	}

	req := services.EmbeddingRequest{
		FileType: fileType,
		FilePath: filePath,
		Text:     text,
	}

	// 이미지인 경우 이미지 경로 설정 및 BLIP으로 설명 생성
	if fileType == "image" {
		req.ImagePath = filePath
		// 이미지 설명을 위해 BLIP 사용하여 텍스트 생성
		// Python ML 서비스에서 이미지 설명을 생성하도록 요청
		if text == "" {
			// BLIP으로 이미지 설명 생성 (Python 서비스에서 처리)
			// 이미지 설명을 텍스트로 사용하여 텍스트 임베딩도 생성
		}
	}

	embeddingResp, err := embeddingService.GenerateEmbedding(req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"success": false,
			"error":   "Failed to generate embedding: " + err.Error(),
		})
		return
	}

	// PostgreSQL에 저장
	if h.db == nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"success": false,
			"error":   "Database not initialized",
		})
		return
	}

	// 임베딩이 하나라도 있어야 저장 가능
	if len(embeddingResp.Result.TextEmbedding) == 0 && len(embeddingResp.Result.ImageEmbedding) == 0 {
		c.JSON(http.StatusBadRequest, gin.H{
			"success": false,
			"error":   "No embedding generated. Please ensure the file has content (text for video/audio, image for image files).",
		})
		return
	}

	var textVec, imageVec pgvector.Vector
	if len(embeddingResp.Result.TextEmbedding) > 0 {
		textVec = pgvector.NewVector(embeddingResp.Result.TextEmbedding)
	}
	if len(embeddingResp.Result.ImageEmbedding) > 0 {
		imageVec = pgvector.NewVector(embeddingResp.Result.ImageEmbedding)
	}

	embedding := &database.MediaEmbedding{
		FilePath:       filePath,
		FileType:       fileType,
		FileName:       filename,
		FolderPath:     folderPath,
		TextEmbedding:  textVec,
		ImageEmbedding: imageVec,
		Transcription:  text,
		Summary:        summary,
	}

	if err := h.db.SaveEmbedding(embedding); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"success": false,
			"error":   "Failed to save embedding: " + err.Error(),
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"success": true,
		"message": "Embedding saved successfully",
	})
}

func (h *Handler) SearchSimilar(c *gin.Context) {
	query := c.Query("q")
	fileType := c.Query("type")
	limit := c.DefaultQuery("limit", "10")

	if query == "" {
		c.JSON(http.StatusBadRequest, gin.H{
			"success": false,
			"error":   "query parameter is required",
		})
		return
	}

	// 데이터베이스 연결 확인
	if h.db == nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"success": false,
			"error":   "Database not initialized. Please check PostgreSQL connection.",
		})
		return
	}

	// Python ML 서비스를 통해 쿼리 텍스트 임베딩 생성
	embeddingService := services.NewEmbeddingService("http://localhost:5002")
	req := services.EmbeddingRequest{
		FileType: "text",
		FilePath: "",
		Text:     query,
	}

	embeddingResp, err := embeddingService.GenerateEmbedding(req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"success": false,
			"error":   "Failed to generate query embedding: " + err.Error(),
		})
		return
	}

	if len(embeddingResp.Result.TextEmbedding) == 0 {
		c.JSON(http.StatusInternalServerError, gin.H{
			"success": false,
			"error":   "Failed to generate embedding",
		})
		return
	}

	queryVec := pgvector.NewVector(embeddingResp.Result.TextEmbedding)

	limitInt, err := strconv.Atoi(limit)
	if err != nil || limitInt <= 0 {
		limitInt = 10
	}

	results, err := h.db.SearchSimilar(queryVec, limitInt, fileType)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"success": false,
			"error":   "Search failed: " + err.Error(),
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"success": true,
		"results": results,
	})
}

