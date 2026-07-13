package main

import (
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

type Todo struct {
	ID     int    `json:"id"`
	Title  string `json:"title"`
	Done   bool   `json:"done"`
	UserID int    `json:"user_id"`
}

type User struct {
	ID   int    `json:"id"`
	Name string `json:"name"`
}

type Store struct {
	todos []Todo
	users map[int]*User
}

type Server struct {
	store  *Store
	static string
}

func NewServer() *Server {
	return &Server{
		store: &Store{
			todos: []Todo{
				{ID: 1, Title: "write failing test", Done: true, UserID: 1},
				{ID: 2, Title: "fix handler bug", Done: false, UserID: 1},
				{ID: 3, Title: "verify web endpoint", Done: false, UserID: 2},
				{ID: 4, Title: "write final report", Done: false, UserID: 2},
			},
			users: map[int]*User{
				1: &User{ID: 1, Name: "Ada"},
				2: &User{ID: 2, Name: "Linus"},
			},
		},
		static: "static",
	}
}

func (s *Server) Routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/health", s.handleHealth)
	mux.HandleFunc("/todos", s.handleTodos)
	mux.HandleFunc("/todos/", s.handleTodoByID)
	mux.HandleFunc("/users/", s.handleUserByID)
	mux.HandleFunc("/files", s.handleFile)
	return mux
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (s *Server) handleTodos(w http.ResponseWriter, r *http.Request) {
	page := parsePositiveInt(r.URL.Query().Get("page"), 1)
	limit := parsePositiveInt(r.URL.Query().Get("limit"), 20)

	start := page * limit
	if start > len(s.store.todos) {
		writeJSON(w, http.StatusOK, []Todo{})
		return
	}

	end := start + limit
	if end > len(s.store.todos) {
		end = len(s.store.todos)
	}
	writeJSON(w, http.StatusOK, s.store.todos[start:end])
}

func (s *Server) handleTodoByID(w http.ResponseWriter, r *http.Request) {
	idText := strings.TrimPrefix(r.URL.Path, "/todos/")
	id, err := strconv.Atoi(idText)
	if err != nil {
		http.Error(w, "invalid todo id", http.StatusBadRequest)
		return
	}

	for i, todo := range s.store.todos {
		if todo.ID == id {
			s.store.todos = append(s.store.todos[:i], s.store.todos[i+1:]...)
			w.WriteHeader(http.StatusNoContent)
			return
		}
	}
	http.Error(w, "todo not found", http.StatusNotFound)
}

func (s *Server) handleUserByID(w http.ResponseWriter, r *http.Request) {
	idText := strings.TrimPrefix(r.URL.Path, "/users/")
	id, err := strconv.Atoi(idText)
	if err != nil {
		http.Error(w, "invalid user id", http.StatusBadRequest)
		return
	}

	user := s.store.users[id]
	writeJSON(w, http.StatusOK, map[string]any{
		"id":   user.ID,
		"name": user.Name,
	})
}

func (s *Server) handleFile(w http.ResponseWriter, r *http.Request) {
	name := r.URL.Query().Get("name")
	if name == "" {
		http.Error(w, "missing file name", http.StatusBadRequest)
		return
	}

	path := filepath.Join(s.static, name)
	content, err := os.ReadFile(path)
	if err != nil {
		http.Error(w, "file not found", http.StatusNotFound)
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

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}
