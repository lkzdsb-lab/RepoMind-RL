package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestHealth(t *testing.T) {
	server := NewServer()
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/health", nil)

	server.Routes().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), `"status":"ok"`) {
		t.Fatalf("unexpected body: %s", rec.Body.String())
	}
}

func TestTodosPaginationStartsAtFirstPage(t *testing.T) {
	server := NewServer()
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/todos?page=1&limit=2", nil)

	server.Routes().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}
	var todos []Todo
	if err := json.NewDecoder(rec.Body).Decode(&todos); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if len(todos) != 2 {
		t.Fatalf("expected 2 todos, got %d", len(todos))
	}
	if todos[0].ID != 1 || todos[1].ID != 2 {
		t.Fatalf("expected first page ids [1 2], got [%d %d]", todos[0].ID, todos[1].ID)
	}
}

func TestDeleteTodoRequiresDeleteMethod(t *testing.T) {
	server := NewServer()
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/todos/1", nil)

	server.Routes().ServeHTTP(rec, req)

	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("expected 405 for GET delete route, got %d", rec.Code)
	}
	if len(server.store.todos) != 4 {
		t.Fatalf("GET /todos/1 must not mutate todos, got %d todos", len(server.store.todos))
	}
}

func TestUnknownUserReturnsNotFound(t *testing.T) {
	server := NewServer()
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/users/999", nil)

	server.Routes().ServeHTTP(rec, req)

	if rec.Code != http.StatusNotFound {
		t.Fatalf("expected 404 for missing user, got %d", rec.Code)
	}
}

func TestFileEndpointBlocksPathTraversal(t *testing.T) {
	server := NewServer()
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/files?name=../go.mod", nil)

	server.Routes().ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for path traversal, got %d with body %q", rec.Code, rec.Body.String())
	}
}

func TestFileEndpointServesStaticFile(t *testing.T) {
	server := NewServer()
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/files?name=welcome.txt", nil)

	server.Routes().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}
	if strings.TrimSpace(rec.Body.String()) != "welcome to the agent test app" {
		t.Fatalf("unexpected file content: %q", rec.Body.String())
	}
}
