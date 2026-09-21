package main

import (
	"net/http"
	"strings"
)

type CreateTodo struct {
	Title     string `json:"title"`
	UserID    int    `json:"user_id"`
	ProjectID int    `json:"project_id"`
	Priority  int    `json:"priority"`
}
type UpdateTodo struct {
	Version  int     `json:"version"`
	Title    *string `json:"title,omitempty"`
	Done     *bool   `json:"done,omitempty"`
	Priority *int    `json:"priority,omitempty"`
}
type BatchItem struct {
	ID int `json:"id"`
	UpdateTodo
}

func validPriority(value int) bool { return value >= 1 && value < 5 }
func validUpdate(input UpdateTodo) bool {
	return input.Version > 0 && (input.Title == nil || strings.TrimSpace(*input.Title) != "") &&
		(input.Priority == nil || validPriority(*input.Priority))
}
func (s *Server) handleCreate(w http.ResponseWriter, r *http.Request) {
	var input CreateTodo
	if !decodeBody(w, r, &input) {
		return
	}
	if input.Title == "" || !validPriority(input.Priority) || !s.store.hasProject(input.ProjectID) || s.store.users[input.UserID] == nil {
		http.Error(w, "invalid todo", 400)
		return
	}
	todo, replay := s.store.create(input, r.Header.Get("Idempotency-Key"))
	status := http.StatusCreated
	if replay {
		status = http.StatusOK
	} else {
		s.invalidateStats()
	}
	writeJSON(w, status, todo)
}
func (s *Server) handleUpdate(w http.ResponseWriter, r *http.Request, id int) {
	var input UpdateTodo
	if !decodeBody(w, r, &input) {
		return
	}
	if !validUpdate(input) {
		http.Error(w, "invalid update", 400)
		return
	}
	todo, err := s.store.update(id, input)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	writeJSON(w, 200, todo)
}
func (s *Server) handleBatch(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodPatch) {
		return
	}
	var input struct {
		Items []BatchItem `json:"items"`
	}
	if !decodeBody(w, r, &input) {
		return
	}
	if len(input.Items) == 0 || len(input.Items) > 100 {
		http.Error(w, "invalid batch size", 400)
		return
	}
	seen := map[int]bool{}
	for _, item := range input.Items {
		if item.ID <= 0 || seen[item.ID] || !validUpdate(item.UpdateTodo) {
			http.Error(w, "invalid batch item", 400)
			return
		}
		seen[item.ID] = true
	}
	todos, err := s.store.batch(input.Items)
	if err != nil {
		writeStoreError(w, err)
		return
	}
	s.invalidateStats()
	writeJSON(w, 200, todos)
}
func writeStoreError(w http.ResponseWriter, err error) {
	status := http.StatusNotFound
	if err == errConflict {
		status = http.StatusConflict
	}
	http.Error(w, err.Error(), status)
}
