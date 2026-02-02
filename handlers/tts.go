package handlers

import (
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
)

func (h *Handler) TextToSpeech(c *gin.Context) {
	text := strings.TrimSpace(c.PostForm("text"))
	language := c.PostForm("language")
	if language == "" {
		language = "auto"
	}

	if text == "" {
		c.Redirect(http.StatusSeeOther, "/")
		return
	}

	// 언어 자동 감지
	if language == "auto" {
		language = detectLanguage(text)
	}

	// Python ML 서비스를 통해 TTS 생성
	_, err := h.mlService.GenerateTTS(text, language)
	if err != nil {
		c.Redirect(http.StatusSeeOther, "/")
		return
	}

	c.Redirect(http.StatusSeeOther, "/")
}

func detectLanguage(text string) string {
	koreanChars := 0
	totalChars := 0

	for _, char := range text {
		if (char >= 0xAC00 && char <= 0xD7A3) {
			koreanChars++
		}
		if (char >= 'a' && char <= 'z') || (char >= 'A' && char <= 'Z') || (char >= 0xAC00 && char <= 0xD7A3) {
			totalChars++
		}
	}

	if totalChars > 0 && float64(koreanChars)/float64(totalChars) > 0.3 {
		return "ko"
	}
	return "en"
}

