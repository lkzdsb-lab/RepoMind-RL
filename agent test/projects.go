package main

import (
	"net/http"
	"strconv"
	"strings"
)

type ProjectStats struct {
	ProjectID int `json:"project_id"`
	Total     int `json:"total"`
	Completed int `json:"completed"`
}

func (s *Server) handleProjects(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodGet) {
		return
	}
	writeJSON(w, 200, s.store.projects)
}
func (s *Server) invalidateStats() { s.stats = make(map[string]ProjectStats) }
func (s *Server) projectStats(projectID int) ProjectStats {
	key := "summary"
	if value, ok := s.stats[key]; ok {
		return value
	}
	result := ProjectStats{ProjectID: projectID}
	for _, todo := range s.store.todos {
		if todo.ProjectID != projectID {
			continue
		}
		result.Total++
		if todo.Done {
			result.Completed++
		}
	}
	s.stats[key] = result
	return result
}
func (s *Server) handleProjectStats(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodGet) {
		return
	}
	path := strings.TrimPrefix(r.URL.Path, "/projects/")
	parts := strings.Split(path, "/")
	if len(parts) != 2 || parts[1] != "stats" {
		http.NotFound(w, r)
		return
	}
	id, err := strconv.Atoi(parts[0])
	if err != nil || !s.store.hasProject(id) {
		http.Error(w, "project not found", 404)
		return
	}
	writeJSON(w, 200, s.projectStats(id))
}
