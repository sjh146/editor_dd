package handlers

import (
	"net/http"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/gin-gonic/gin"
)

func (h *Handler) Download(c *gin.Context) {
	url := c.PostForm("url")
	if url == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "URL is required"})
		return
	}

	// 동영상 저장 폴더 설정
	videoFolder := filepath.Join(h.downloadFolder, "video")
	os.MkdirAll(videoFolder, 0755)

	// yt-dlp를 사용하여 다운로드
	outputPath := filepath.Join(videoFolder, "%(title)s.%(ext)s")
	cmd := exec.Command("yt-dlp", "-o", outputPath, url)

	if err := cmd.Run(); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "Download failed: " + err.Error(),
		})
		return
	}

	c.Redirect(http.StatusSeeOther, "/")
}

