#!/usr/bin/env python3
"""Serveur local qui INTERDIT le cache — un .wasm périmé fait conclure à tort
qu'un correctif n'a rien changé."""
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

class NoCache(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
root = sys.argv[2] if len(sys.argv) > 2 else 'dist'
NoCache.directory = root
ThreadingHTTPServer(('127.0.0.1', port), lambda *a: NoCache(*a, directory=root)).serve_forever()
