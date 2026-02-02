package handlers

import (
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/gin-gonic/gin"
)

func (h *Handler) Edit(c *gin.Context) {
	filename := c.PostForm("filename")
	startTime := c.PostForm("start_time")
	endTime := c.PostForm("end_time")

	if filename == "" || startTime == "" || endTime == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "All fields are required"})
		return
	}

	// 동영상 파일은 downloads/video 폴더에 있음
	inputPath := filepath.Join(h.downloadFolder, "video", filename)
	if _, err := os.Stat(inputPath); os.IsNotExist(err) {
		// video 폴더에 없으면 downloads 폴더에서 찾기 (하위 호환성)
		inputPath = filepath.Join(h.downloadFolder, filename)
	}
	
	ext := filepath.Ext(filename)
	name := strings.TrimSuffix(filename, ext)
	outputFilename := name + "_trimmed" + ext
	outputPath := filepath.Join(h.downloadFolder, "video", outputFilename)

	cmd := exec.Command("ffmpeg",
		"-y",
		"-i", inputPath,
		"-ss", startTime,
		"-to", endTime,
		"-c", "copy",
		outputPath,
	)

	if err := cmd.Run(); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "Failed to trim video: " + err.Error(),
		})
		return
	}

	c.Redirect(http.StatusSeeOther, "/")
}

func (h *Handler) ExtractFrame(c *gin.Context) {
	filename := c.PostForm("filename")
	frameTime := c.PostForm("frame_time")

	if filename == "" || frameTime == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Filename and time are required"})
		return
	}

	// 동영상 파일은 downloads/video 폴더에 있음
	inputPath := filepath.Join(h.downloadFolder, "video", filename)
	if _, err := os.Stat(inputPath); os.IsNotExist(err) {
		// video 폴더에 없으면 downloads 폴더에서 찾기 (하위 호환성)
		inputPath = filepath.Join(h.downloadFolder, filename)
	}
	
	ext := filepath.Ext(filename)
	name := strings.TrimSuffix(filename, ext)
	timeStr := strings.ReplaceAll(frameTime, ":", "-")
	outputFilename := name + "_frame_" + timeStr + ".jpg"
	outputPath := filepath.Join(h.downloadFolder, "frame", outputFilename)
	os.MkdirAll(filepath.Dir(outputPath), 0755)

	cmd := exec.Command("ffmpeg",
		"-y",
		"-ss", frameTime,
		"-i", inputPath,
		"-vframes", "1",
		"-q:v", "2",
		outputPath,
	)

	if err := cmd.Run(); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "Failed to extract frame: " + err.Error(),
		})
		return
	}

	c.Redirect(http.StatusSeeOther, "/")
}

func (h *Handler) AddTTSToVideo(c *gin.Context) {
	videoFilename := c.PostForm("video_filename")
	ttsFilename := c.PostForm("tts_filename")
	audioMode := c.PostForm("audio_mode")
	if audioMode == "" {
		audioMode = "replace"
	}
	outputFilename := strings.TrimSpace(c.PostForm("output_filename"))

	// 동영상 파일은 downloads/video 폴더에 있음
	videoPath := filepath.Join(h.downloadFolder, "video", videoFilename)
	if _, err := os.Stat(videoPath); os.IsNotExist(err) {
		// video 폴더에 없으면 downloads 폴더에서 찾기 (하위 호환성)
		videoPath = filepath.Join(h.downloadFolder, videoFilename)
	}
	
	// TTS 파일은 downloads/tts 폴더에 있음
	ttsPath := filepath.Join(h.downloadFolder, "tts", ttsFilename)
	if _, err := os.Stat(ttsPath); os.IsNotExist(err) {
		// tts 폴더에 없으면 downloads 폴더에서 찾기 (하위 호환성)
		ttsPath = filepath.Join(h.downloadFolder, ttsFilename)
	}

	if _, err := os.Stat(videoPath); os.IsNotExist(err) {
		c.JSON(http.StatusNotFound, gin.H{"error": "Video file not found"})
		return
	}

	if _, err := os.Stat(ttsPath); os.IsNotExist(err) {
		c.JSON(http.StatusNotFound, gin.H{"error": "TTS file not found"})
		return
	}

	// 출력 파일명 생성
	if outputFilename == "" {
		ext := filepath.Ext(videoFilename)
		name := strings.TrimSuffix(videoFilename, ext)
		outputFilename = name + "_with_tts" + ext
	} else {
		if !strings.Contains(outputFilename, ".") {
			ext := filepath.Ext(videoFilename)
			outputFilename = outputFilename + ext
		}
	}

	outputPath := filepath.Join(h.downloadFolder, "video", outputFilename)

	var cmd *exec.Cmd
	if audioMode == "replace" {
		cmd = exec.Command("ffmpeg",
			"-y",
			"-i", videoPath,
			"-i", ttsPath,
			"-c:v", "copy",
			"-c:a", "aac",
			"-map", "0:v:0",
			"-map", "1:a:0",
			"-shortest",
			outputPath,
		)
	} else {
		// 믹스 모드
		cmd = exec.Command("ffmpeg",
			"-y",
			"-i", videoPath,
			"-i", ttsPath,
			"-filter_complex", "[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=2[a]",
			"-map", "0:v:0",
			"-map", "[a]",
			"-c:v", "copy",
			"-c:a", "aac",
			outputPath,
		)
	}

	if err := cmd.Run(); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "Failed to add TTS: " + err.Error(),
		})
		return
	}

	c.Redirect(http.StatusSeeOther, "/")
}

func (h *Handler) DownloadFile(c *gin.Context) {
	filename := c.Param("filename")
	filename = strings.TrimPrefix(filename, "/")
	
	// 파일 경로 찾기 (video, tts, frame, analysis 폴더 확인)
	possiblePaths := []string{
		filepath.Join(h.downloadFolder, "video", filename),
		filepath.Join(h.downloadFolder, "tts", filename),
		filepath.Join(h.downloadFolder, "frame", filename),
		filepath.Join(h.downloadFolder, "analysis", filename),
		filepath.Join(h.downloadFolder, filename), // 하위 호환성
	}
	
	var filePath string
	for _, path := range possiblePaths {
		if _, err := os.Stat(path); err == nil {
			filePath = path
			break
		}
	}
	
	if filePath == "" {
		c.JSON(http.StatusNotFound, gin.H{"error": "File not found"})
		return
	}
	
	c.File(filePath)
}

