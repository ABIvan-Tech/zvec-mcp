# Zvec MCP Bridge

This repository contains a local MCP bridge that indexes project source files into a Zvec vector database and exposes semantic search through MCP tools.

## What the bridge does

The current implementation is a stdio-based MCP server that:

- creates or opens a local Zvec collection at `.zvec/knowledge.db` inside the project root;
- embeds text chunks with the Hugging Face Transformers feature-extraction pipeline using `Xenova/all-MiniLM-L6-v2`;
- walks the project tree once at startup to build the initial index;
- watches for file add/change/delete events and updates the index automatically;
- exposes four MCP tools for search, re-indexing, single-file indexing, and status checks.

## Requirements

- Node.js 18 or newer
- npm dependencies from the project package file
- a local model download on first use (the bridge uses the Hugging Face Transformers pipeline)

## Installation

From the repository root:

```bash
npm install
```

### Node.js 22+ / onnxruntime-node install failures

`onnxruntime-node` (a transitive dependency of `@huggingface/transformers`) ships
its core CPU binaries inside the npm tarball but runs a postinstall script that
attempts to download **optional CUDA provider libraries** from a NuGet feed. On
machines without CUDA — or behind a firewall — this download times out and,
under Node.js 22+, the unhandled promise rejection causes the entire `npm install`
to fail.

The repository includes an `.npmrc` that sets `onnxruntime-node-install=skip`,
telling the postinstall script to exit immediately. The bundled CPU binaries are
sufficient for embedding inference; CUDA is not required.

## Running the bridge

Run the bridge directly:

```bash
PROJECT_ROOT=/absolute/path/to/your/project node zvec-mcp-bridge.js
```

If `PROJECT_ROOT` is not provided, the bridge uses the current working directory.

## MCP configuration example

This bridge is not tied to Google Antigravity specifically; it works with any MCP-compatible client that can launch a stdio process.

```json
{
  "mcpServers": {
    "zvec-project-knowledge": {
      "command": "node",
      "args": ["/absolute/path/to/zvec-mcp-bridge.js"],
      "env": {
        "PROJECT_ROOT": "/absolute/path/to/your/project"
      }
    }
  }
}
```

## Supported files

The bridge indexes these file extensions:

- `.js`, `.jsx`, `.ts`, `.tsx`
- `.kt`, `.erl`, `.hrl`
- `.py`, `.go`, `.java`, `.cs`, `.rb`, `.php`
- `.cpp`, `.c`, `.h`, `.hpp`, `.rs`, `.swift`, `.scala`

It skips common config and lock files such as `package.json`, `package-lock.json`, `tsconfig.json`, `vite.config.*`, and similar files. It also ignores generated or dependency-heavy directories like `node_modules`, `.git`, `dist`, `build`, `.cache`, `.next`, and `.vscode`.

## MCP tools

### `search_project_knowledge`

Searches the local Zvec knowledge base for relevant code snippets.

Input:

- `query` (required): a natural-language search request
- `exclude_paths` (optional): path substrings to exclude from results
- `include_paths` (optional): path substrings that must be present in results

Behavior:

- the bridge creates embeddings for the query;
- it runs a vector search with `topk: 15`;
- results are filtered and re-ranked in JavaScript using keyword and path heuristics;
- each result includes a short explanation and the matched code chunk.

### `initialize_project_knowledge`

Initializes the knowledge base and indexes the project contents.

Input:

- `force_rebuild` (optional): if `true`, clears the existing index before re-indexing

### `index_file`

Indexes or refreshes a single file immediately.

Input:

- `file_path` (required): relative or absolute path to the file

### `get_knowledge_status`

Returns current database status information such as:

- database path
- whether the database exists
- document count
- initialization state
- project root

## Indexing details

- text is chunked into roughly 1000-character pieces with a 200-character overlap;
- the bridge stores each chunk as a document with fields for `text_content`, `file_path`, and `language`;
- the embedding vector field is named `code_embedding`.

## Example prompt

A useful example prompt for the LLM can be taken from [AGENTS.md](AGENTS.md).
