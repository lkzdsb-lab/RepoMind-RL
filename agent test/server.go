package main

import (
	"encoding/json"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

type Server struct {
	store  *Store
	static string
	stats  map[string]ProjectStats
}

func NewServer() *Server {
	return &Server{store: newStore(), static: "static", stats: make(map[string]ProjectStats)}
}

func (s *Server) Routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/health", s.handleHealth)
	mux.HandleFunc("/todos", s.handleTodos)
	mux.HandleFunc("/todos/batch", s.handleBatch)
	mux.HandleFunc("/todos/", s.handleTodoByID)
	mux.HandleFunc("/users/", s.handleUserByID)
	mux.HandleFunc("/files", s.handleFile)
	mux.HandleFunc("/projects", s.handleProjects)
	mux.HandleFunc("/projects/", s.handleProjectStats)
	return mux
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (s *Server) handleTodos(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodPost {
		s.handleCreate(w, r)
		return
	}
	if !requireMethod(w, r, http.MethodGet) {
		return
	}
	page := parsePositiveInt(r.URL.Query().Get("page"), 1)
	limit := parsePositiveInt(r.URL.Query().Get("limit"), 20)
	if limit > 100 {
		limit = 100
	}
	project := r.URL.Query().Get("project_id")
	done := r.URL.Query().Get("done")
	if project != "" {
		id, err := strconv.Atoi(project)
		if err != nil || !s.store.hasProject(id) {
			http.Error(w, "invalid project", 400)
			return
		}
	}
	if done != "" && done != "true" && done != "false" {
		http.Error(w, "invalid done", 400)
		return
	}
	writeJSON(w, 200, s.store.list(page, limit, project, done))
}

func (s *Server) handleTodoByID(w http.ResponseWriter, r *http.Request) {
	id, err := strconv.Atoi(strings.TrimPrefix(r.URL.Path, "/todos/"))
	if err != nil {
		http.Error(w, "invalid todo id", 400)
		return
	}
	if r.Method == http.MethodPatch {
		s.handleUpdate(w, r, id)
		return
	}
	if s.store.remove(id) {
		s.invalidateStats()
		w.WriteHeader(204)
		return
	}
	http.Error(w, "todo not found", 404)
}

func (s *Server) handleUserByID(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodGet) {
		return
	}
	id, err := strconv.Atoi(strings.TrimPrefix(r.URL.Path, "/users/"))
	if err != nil {
		http.Error(w, "invalid user id", 400)
		return
	}
	user := s.store.users[id]
	writeJSON(w, 200, map[string]any{"id": user.ID, "name": user.Name})
}

func (s *Server) handleFile(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodGet) {
		return
	}
	name := r.URL.Query().Get("name")
	if name == "" {
		http.Error(w, "missing file name", 400)
		return
	}
	path := filepath.Join(s.static, name)
	content, err := os.ReadFile(path)
	if err != nil {
		http.Error(w, "file not found", 404)
		return
	}
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	_, _ = w.Write(content)
}

func parsePositiveInt(value string, fallback int) int {
	parsed, err := strconv.Atoi(value)
	if err != nil || parsed <= 0 {
		return fallback
	}
	return parsed
}

func requireMethod(w http.ResponseWriter, r *http.Request, method string) bool {
	if r.Method == method {
		return true
	}
	w.Header().Set("Allow", method)
	http.Error(w, "method not allowed", 405)
	return false
}

func decodeBody(w http.ResponseWriter, r *http.Request, target any) bool {
	r.Body = http.MaxBytesReader(w, r.Body, 1<<20)
	decoder := json.NewDecoder(r.Body)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		http.Error(w, "invalid JSON", 400)
		return false
	}
	var extra any
	if err := decoder.Decode(&extra); err != io.EOF {
		http.Error(w, "expected one JSON value", 400)
		return false
	}
	return true
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}
