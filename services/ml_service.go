package services

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
)

type MLService struct {
	baseURL string
	client  *http.Client
}

func NewMLService(baseURL string) *MLService {
	return &MLService{
		baseURL: baseURL,
		client:  &http.Client{},
	}
}

type TTSRequest struct {
	Text     string `json:"text"`
	Language string `json:"language"`
}

type TTSResponse struct {
	Success  bool   `json:"success"`
	Filename string `json:"filename"`
	Message  string `json:"message"`
}

type VideoAnalysisRequest struct {
	VideoPath string `json:"video_path"`
}

type VideoAnalysisResponse struct {
	Success bool                   `json:"success"`
	Result  map[string]interface{} `json:"result"`
	Error   string                 `json:"error,omitempty"`
}

func (s *MLService) GenerateTTS(text, language string) (*TTSResponse, error) {
	reqBody := TTSRequest{
		Text:     text,
		Language: language,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}

	resp, err := s.client.Post(
		s.baseURL+"/tts",
		"application/json",
		bytes.NewBuffer(jsonData),
	)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result TTSResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	if !result.Success {
		return nil, fmt.Errorf("TTS generation failed: %s", result.Message)
	}

	return &result, nil
}

func (s *MLService) AnalyzeVideo(videoPath string) (map[string]interface{}, error) {
	// 파일 열기
	file, err := os.Open(videoPath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	// 멀티파트 폼 생성
	var requestBody bytes.Buffer
	writer := multipart.NewWriter(&requestBody)

	// 파일 추가
	part, err := writer.CreateFormFile("video", filepath.Base(videoPath))
	if err != nil {
		return nil, err
	}

	_, err = io.Copy(part, file)
	if err != nil {
		return nil, err
	}

	writer.Close()

	// 요청 전송
	req, err := http.NewRequest("POST", s.baseURL+"/analyze_video", &requestBody)
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", writer.FormDataContentType())

	resp, err := s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result VideoAnalysisResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	if !result.Success {
		return nil, fmt.Errorf("analysis failed: %s", result.Error)
	}

	return result.Result, nil
}

