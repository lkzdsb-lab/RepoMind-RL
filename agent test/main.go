package main

import (
	"log"
	"net"
	"net/http"
	"os"
)

func main() {
	server := NewServer()
	address := os.Getenv("AGENT_TEST_ADDR")
	if address == "" {
		address = "127.0.0.1:8080"
	}
	listener, err := net.Listen("tcp", address)
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("listening on http://%s", listener.Addr())
	if err := http.Serve(listener, server.Routes()); err != nil {
		log.Fatal(err)
	}
}
