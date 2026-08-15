package main

import (
	"log"
	"net/http"
)

func main() {
	server := NewServer()
	log.Println("listening on http://localhost:8080")
	if err := http.ListenAndServe(":8080", server.Routes()); err != nil {
		log.Fatal(err)
	}
}
