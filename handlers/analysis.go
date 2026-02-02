package handlers

import (
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/gin-gonic/gin"
)

func (h *Handler) AnalyzeVideo(c *gin.Context) {
	filename := c.PostForm("filename")
	if filename == "" {
		c.JSON(http.StatusBadRequest, gin.H{
			"success": false,
			"error":   "Video file is required",
		})
		return
	}

	// 동영상 파일 경로 확인 (video 폴더 또는 downloads 폴더)
	videoPath := filepath.Join(h.downloadFolder, "video", filename)
	if _, err := os.Stat(videoPath); os.IsNotExist(err) {
		// video 폴더에 없으면 downloads 폴더에서 찾기
		videoPath = filepath.Join(h.downloadFolder, filename)
		if _, err := os.Stat(videoPath); os.IsNotExist(err) {
			c.JSON(http.StatusNotFound, gin.H{
				"success": false,
				"error":   "Video file not found",
			})
			return
		}
	}

	// Python ML 서비스를 통해 동영상 분석
	result, err := h.mlService.AnalyzeVideo(videoPath)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"success": false,
			"error":   "Analysis failed: " + err.Error(),
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"success": true,
		"result":  result,
	})
}

func (h *Handler) GetVideoAnalysis(c *gin.Context) {
	filename := c.Param("filename")
	filename = strings.TrimPrefix(filename, "/")
	resultPath := filepath.Join(h.downloadFolder, "analysis", filename)

	if _, err := os.Stat(resultPath); os.IsNotExist(err) {
		c.JSON(http.StatusNotFound, gin.H{"error": "Analysis result not found"})
		return
	}

	// JSON 파일 읽기 및 반환
	c.File(resultPath)
}

