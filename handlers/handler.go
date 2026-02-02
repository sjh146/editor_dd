package handlers

import (
	"editor_dd/database"
	"editor_dd/services"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/gin-gonic/gin"
)

type Handler struct {
	downloadFolder string
	mlService      *services.MLService
	db             *database.DB
}

func NewHandler(downloadFolder string, mlService *services.MLService, db *database.DB) *Handler {
	return &Handler{
		downloadFolder: downloadFolder,
		mlService:      mlService,
		db:             db,
	}
}

func (h *Handler) Index(c *gin.Context) {
	files, err := h.getDownloadedFiles()
	if err != nil {
		c.HTML(http.StatusInternalServerError, "index.html", gin.H{
			"error": "Failed to list files",
		})
		return
	}

	ttsFiles := h.getTTSFiles(files)
	videoFiles := h.getVideoFiles(files)
	imageFiles := h.getImageFiles()

	c.HTML(http.StatusOK, "index.html", gin.H{
		"files":       files,
		"tts_files":   ttsFiles,
		"video_files": videoFiles,
		"image_files": imageFiles,
	})
}

func (h *Handler) getDownloadedFiles() ([]string, error) {
	entries, err := filepath.Glob(filepath.Join(h.downloadFolder, "*"))
	if err != nil {
		return nil, err
	}

	var files []string
	for _, entry := range entries {
		info, err := os.Stat(entry)
		if err != nil {
			continue
		}
		if !info.IsDir() {
			files = append(files, filepath.Base(entry))
		}
	}

	// 정렬 (최신순)
	for i := 0; i < len(files)-1; i++ {
		for j := i + 1; j < len(files); j++ {
			infoI, _ := os.Stat(filepath.Join(h.downloadFolder, files[i]))
			infoJ, _ := os.Stat(filepath.Join(h.downloadFolder, files[j]))
			if infoI.ModTime().Before(infoJ.ModTime()) {
				files[i], files[j] = files[j], files[i]
			}
		}
	}

	return files, nil
}

func (h *Handler) getTTSFiles(files []string) []string {
	// downloads/tts 폴더에서만 TTS 파일 가져오기
	ttsFolder := filepath.Join(h.downloadFolder, "tts")
	entries, err := filepath.Glob(filepath.Join(ttsFolder, "*"))
	if err != nil {
		return []string{}
	}

	var ttsFiles []string
	for _, entry := range entries {
		info, err := os.Stat(entry)
		if err != nil {
			continue
		}
		if info.IsDir() {
			continue
		}

		filename := filepath.Base(entry)
		if strings.HasPrefix(filename, "tts_") && (strings.HasSuffix(filename, ".wav") || strings.HasSuffix(filename, ".mp3")) {
			ttsFiles = append(ttsFiles, filename)
		}
	}

	// 정렬 (최신순)
	for i := 0; i < len(ttsFiles)-1; i++ {
		for j := i + 1; j < len(ttsFiles); j++ {
			infoI, _ := os.Stat(filepath.Join(ttsFolder, ttsFiles[i]))
			infoJ, _ := os.Stat(filepath.Join(ttsFolder, ttsFiles[j]))
			if infoI.ModTime().Before(infoJ.ModTime()) {
				ttsFiles[i], ttsFiles[j] = ttsFiles[j], ttsFiles[i]
			}
		}
	}

	return ttsFiles
}

func (h *Handler) getVideoFiles(files []string) []string {
	// downloads/video 폴더에서만 동영상 파일 가져오기
	videoFolder := filepath.Join(h.downloadFolder, "video")
	entries, err := filepath.Glob(filepath.Join(videoFolder, "*"))
	if err != nil {
		return []string{}
	}

	videoExts := []string{".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".mpg", ".mpeg"}
	var videoFiles []string
	for _, entry := range entries {
		info, err := os.Stat(entry)
		if err != nil {
			continue
		}
		if info.IsDir() {
			continue
		}

		filename := filepath.Base(entry)
		for _, ext := range videoExts {
			if strings.HasSuffix(strings.ToLower(filename), ext) {
				videoFiles = append(videoFiles, filename)
				break
			}
		}
	}

	// 정렬 (최신순)
	for i := 0; i < len(videoFiles)-1; i++ {
		for j := i + 1; j < len(videoFiles); j++ {
			infoI, _ := os.Stat(filepath.Join(videoFolder, videoFiles[i]))
			infoJ, _ := os.Stat(filepath.Join(videoFolder, videoFiles[j]))
			if infoI.ModTime().Before(infoJ.ModTime()) {
				videoFiles[i], videoFiles[j] = videoFiles[j], videoFiles[i]
			}
		}
	}

	return videoFiles
}

func (h *Handler) getImageFiles() []string {
	// downloads/frame과 downloads/analysis 폴더에서 이미지 파일 가져오기
	var imageFiles []string
	
	folders := []string{"frame", "analysis"}
	imageExts := []string{".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
	
	for _, folder := range folders {
		folderPath := filepath.Join(h.downloadFolder, folder)
		entries, err := filepath.Glob(filepath.Join(folderPath, "*"))
		if err != nil {
			continue
		}
		
		for _, entry := range entries {
			info, err := os.Stat(entry)
			if err != nil {
				continue
			}
			if info.IsDir() {
				continue
			}
			
			filename := filepath.Base(entry)
			for _, ext := range imageExts {
				if strings.HasSuffix(strings.ToLower(filename), ext) {
					imageFiles = append(imageFiles, filename)
					break
				}
			}
		}
	}
	
	// 정렬 (최신순)
	for i := 0; i < len(imageFiles)-1; i++ {
		for j := i + 1; j < len(imageFiles); j++ {
			// 파일 경로 찾기
			var pathI, pathJ string
			for _, folder := range folders {
				p := filepath.Join(h.downloadFolder, folder, imageFiles[i])
				if _, err := os.Stat(p); err == nil {
					pathI = p
					break
				}
			}
			for _, folder := range folders {
				p := filepath.Join(h.downloadFolder, folder, imageFiles[j])
				if _, err := os.Stat(p); err == nil {
					pathJ = p
					break
				}
			}
			
			if pathI != "" && pathJ != "" {
				infoI, _ := os.Stat(pathI)
				infoJ, _ := os.Stat(pathJ)
				if infoI.ModTime().Before(infoJ.ModTime()) {
					imageFiles[i], imageFiles[j] = imageFiles[j], imageFiles[i]
				}
			}
		}
	}
	
	return imageFiles
}

