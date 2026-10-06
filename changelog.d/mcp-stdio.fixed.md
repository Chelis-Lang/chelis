`chelis tide mcp` uses newline-delimited JSON instead of LSP-style
`Content-Length` headers, so standard MCP clients can initialize, discover tools,
and call them over stdio. Notifications no longer receive spurious error
responses, and the server answers `ping` requests.
