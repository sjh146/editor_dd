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

func (h *Handler) ConcatVideos(c *gin.Context) {
	video1Filename := c.PostForm("video1_filename")
	video2Filename := c.PostForm("video2_filename")
	outputFilename := strings.TrimSpace(c.PostForm("output_filename"))

	if video1Filename == "" || video2Filename == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Both video files are required"})
		return
	}

	// 동영상 파일 경로 확인
	video1Path := filepath.Join(h.downloadFolder, "video", video1Filename)
	if _, err := os.Stat(video1Path); os.IsNotExist(err) {
		video1Path = filepath.Join(h.downloadFolder, video1Filename)
	}

	video2Path := filepath.Join(h.downloadFolder, "video", video2Filename)
	if _, err := os.Stat(video2Path); os.IsNotExist(err) {
		video2Path = filepath.Join(h.downloadFolder, video2Filename)
	}

	if _, err := os.Stat(video1Path); os.IsNotExist(err) {
		c.JSON(http.StatusNotFound, gin.H{"error": "First video file not found"})
		return
	}

	if _, err := os.Stat(video2Path); os.IsNotExist(err) {
		c.JSON(http.StatusNotFound, gin.H{"error": "Second video file not found"})
		return
	}

	// 출력 파일명 생성
	if outputFilename == "" {
		ext := filepath.Ext(video1Filename)
		name1 := strings.TrimSuffix(video1Filename, ext)
		name2 := strings.TrimSuffix(video2Filename, ext)
		outputFilename = name1 + "_concat_" + name2 + ext
	} else {
		if !strings.Contains(outputFilename, ".") {
			ext := filepath.Ext(video1Filename)
			outputFilename = outputFilename + ext
		}
	}

	outputPath := filepath.Join(h.downloadFolder, "video", outputFilename)

	// 임시 파일 리스트 생성 (FFmpeg concat demuxer용)
	tempDir := filepath.Join(h.downloadFolder, "temp")
	os.MkdirAll(tempDir, 0755)
	fileListPath := filepath.Join(tempDir, "concat_list.txt")

	// 파일 리스트 작성 (절대 경로 사용)
	absVideo1Path, err1 := filepath.Abs(video1Path)
	absVideo2Path, err2 := filepath.Abs(video2Path)
	if err1 != nil || err2 != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "Failed to get absolute paths",
		})
		return
	}
	
	// Windows 경로를 FFmpeg가 이해할 수 있는 형식으로 변환
	// Windows 드라이브 문자를 처리하고 백슬래시를 슬래시로 변환
	normalizePath := func(path string) string {
		// Windows 경로를 Unix 스타일로 변환
		path = strings.ReplaceAll(path, "\\", "/")
		// 드라이브 문자 처리 (예: C:/path -> /C:/path 또는 그대로 유지)
		return path
	}
	
	fileListContent := "file '" + normalizePath(absVideo1Path) + "'\n"
	fileListContent += "file '" + normalizePath(absVideo2Path) + "'\n"

	if err := os.WriteFile(fileListPath, []byte(fileListContent), 0644); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "Failed to create file list: " + err.Error(),
		})
		return
	}
	defer os.Remove(fileListPath) // 처리 후 임시 파일 삭제

	// FFmpeg concat 명령 실행
	// concat demuxer 사용 (같은 코덱일 때 빠름)
	cmd := exec.Command("ffmpeg",
		"-y",
		"-f", "concat",
		"-safe", "0",
		"-i", fileListPath,
		"-c", "copy",
		outputPath,
	)

	if err := cmd.Run(); err != nil {
		// concat demuxer가 실패하면 concat filter 사용 (다른 코덱/해상도일 때)
		cmd = exec.Command("ffmpeg",
			"-y",
			"-i", video1Path,
			"-i", video2Path,
			"-filter_complex", "[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[outv][outa]",
			"-map", "[outv]",
			"-map", "[outa]",
			"-c:v", "libx264",
			"-c:a", "aac",
			outputPath,
		)

		if err := cmd.Run(); err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{
				"error": "Failed to concatenate videos: " + err.Error(),
			})
			return
		}
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

