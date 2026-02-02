package services

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
)

type EmbeddingService struct {
	baseURL string
	client  *http.Client
}

func NewEmbeddingService(baseURL string) *EmbeddingService {
	return &EmbeddingService{
		baseURL: baseURL,
		client:  &http.Client{},
	}
}

type EmbeddingRequest struct {
	FileType  string `json:"file_type"`
	FilePath  string `json:"file_path"`
	Text      string `json:"text,omitempty"`
	ImagePath string `json:"image_path,omitempty"`
}

type EmbeddingResponse struct {
	Success bool `json:"success"`
	Result  struct {
		FilePath      string    `json:"file_path"`
		FileType      string    `json:"file_type"`
		TextEmbedding []float32 `json:"text_embedding"`
		ImageEmbedding []float32 `json:"image_embedding"`
	} `json:"result"`
	Error string `json:"error,omitempty"`
}

func (s *EmbeddingService) GenerateEmbedding(req EmbeddingRequest) (*EmbeddingResponse, error) {
	jsonData, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	resp, err := s.client.Post(
		s.baseURL+"/generate_embedding",
		"application/json",
		bytes.NewBuffer(jsonData),
	)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result EmbeddingResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	if !result.Success {
		return nil, fmt.Errorf("embedding generation failed: %s", result.Error)
	}

	return &result, nil
}

