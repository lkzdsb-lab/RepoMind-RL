package main

import (
	"errors"
	"strconv"
)

type Todo struct {
	ID        int    `json:"id"`
	Title     string `json:"title"`
	Done      bool   `json:"done"`
	UserID    int    `json:"user_id"`
	ProjectID int    `json:"project_id"`
	Priority  int    `json:"priority"`
	Version   int    `json:"version"`
}
type User struct {
	ID   int    `json:"id"`
	Name string `json:"name"`
}
type Project struct {
	ID   int    `json:"id"`
	Name string `json:"name"`
}
type Store struct {
	todos       []Todo
	users       map[int]*User
	projects    []Project
	nextID      int
	idempotency map[string]Todo
}

var errMissing = errors.New("todo not found")
var errConflict = errors.New("version conflict")

func newStore() *Store {
	return &Store{
		todos: []Todo{
			{ID: 1, Title: "write failing test", Done: true, UserID: 1, ProjectID: 1, Priority: 1, Version: 1},
			{ID: 2, Title: "fix handler bug", UserID: 1, ProjectID: 1, Priority: 2, Version: 1},
			{ID: 3, Title: "verify web endpoint", UserID: 2, ProjectID: 2, Priority: 3, Version: 1},
			{ID: 4, Title: "write final report", UserID: 2, ProjectID: 2, Priority: 1, Version: 1},
		},
		users:    map[int]*User{1: {ID: 1, Name: "Ada"}, 2: {ID: 2, Name: "Linus"}},
		projects: []Project{{ID: 1, Name: "Engine"}, {ID: 2, Name: "Console"}},
		nextID:   5, idempotency: make(map[string]Todo),
	}
}
func (s *Store) hasProject(id int) bool {
	for _, p := range s.projects {
		if p.ID == id {
			return true
		}
	}
	return false
}
func window(todos []Todo, start, limit int) []Todo {
	if start < 0 || start >= len(todos) {
		return []Todo{}
	}
	end := start + limit
	if end > len(todos) {
		end = len(todos)
	}
	return append([]Todo{}, todos[start:end]...)
}
func filterTodos(todos []Todo, project, done string) []Todo {
	result := []Todo{}
	for _, todo := range todos {
		if project != "" && strconv.Itoa(todo.ProjectID) != project {
			continue
		}
		if done != "" && strconv.FormatBool(todo.Done) != done {
			continue
		}
		result = append(result, todo)
	}
	return result
}
func (s *Store) list(page, limit int, project, done string) []Todo {
	if page > len(s.todos)+1 {
		return []Todo{}
	}
	if project == "" && done == "" {
		return window(s.todos, page*limit, limit)
	}
	return filterTodos(window(s.todos, (page-1)*limit, limit), project, done)
}
func (s *Store) remove(id int) bool {
	for i, todo := range s.todos {
		if todo.ID == id {
			s.todos = append(s.todos[:i], s.todos[i+1:]...)
			return true
		}
	}
	return false
}
func (s *Store) create(input CreateTodo, key string) (Todo, bool) {
	if key != "" {
		if old, ok := s.idempotency[key]; ok {
			return old, true
		}
	}
	todo := Todo{ID: s.nextID, Title: input.Title, UserID: input.UserID, ProjectID: input.ProjectID, Priority: input.Priority, Version: 1}
	s.nextID++
	s.todos = append(s.todos, todo)
	if key != "" {
		s.idempotency[key] = todo
	}
	return todo, false
}
func (s *Store) update(id int, input UpdateTodo) (Todo, error) {
	for i := range s.todos {
		if s.todos[i].ID != id {
			continue
		}
		todo := &s.todos[i]
		oldVersion := todo.Version
		if input.Title != nil {
			todo.Title = *input.Title
		}
		if input.Done != nil {
			todo.Done = *input.Done
		}
		if input.Priority != nil {
			todo.Priority = *input.Priority
		}
		todo.Version++
		if input.Version != oldVersion {
			return Todo{}, errConflict
		}
		return *todo, nil
	}
	return Todo{}, errMissing
}
func (s *Store) batch(items []BatchItem) ([]Todo, error) {
	result := []Todo{}
	for _, item := range items {
		todo, err := s.update(item.ID, item.UpdateTodo)
		if err != nil {
			return nil, err
		}
		result = append(result, todo)
	}
	return result, nil
}
