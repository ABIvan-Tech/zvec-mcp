# Agent Instructions

For repository and implementation questions, first consult the local project knowledge base through the MCP tool `search_project_knowledge` before doing any broad repository search. If `search_project_knowledge` returns a tool error or is unavailable, do not silently continue as if the knowledge base had been checked successfully; explicitly mention that the MCP knowledge base was unavailable and then fall back to a targeted grep search. If `search_project_knowledge` returns no useful results, explain that the knowledge base did not provide a useful hit and then proceed to a focused grep search.

> [!NOTE]
> The database indexes source files only. Files with extensions like `.md` (documentation), configuration files (`package.json`, `tsconfig.json`, `vite.config.ts`, `eslint.config.js`), and declaration files (`.d.ts`) are **ignored** by design. For these files or queries about general architecture/configs, skip vector search and use `grep_search` or `view_file` directly.
> If the MCP tool throws locking/concurrency errors (e.g. `Can't lock collection`), it may mean there are stale bridge processes holding the file lock. Proactively list and terminate any stale `node zvec-mcp-bridge.js` processes.
> If the index seems outdated or returns zero results on valid codebase queries, call `initialize_project_knowledge` with `force_rebuild: true` to regenerate the database.

## Available MCP Tools

| Tool | Description |
|------|-------------|
| `search_project_knowledge` | Semantic search through the indexed project codebase. Use as first step before broad repository search. |
| `initialize_project_knowledge` | Create or rebuild the knowledge base index. Call when index is missing, stale, or corrupted. |
| `index_file` | Force indexing of a specific file immediately. Useful for newly created or modified files. |
| `get_knowledge_status` | Get health, statistics, and configuration (document count, init state, detected extensions, project type). |

## Available MCP Resources

| Resource URI | Description |
|--------------|-------------|
| `zvec://project/info` | Metadata about the indexed project (root path, detected type, extensions, git remote). |
| `zvec://knowledge/status` | Current health and statistics of the knowledge base. |
| `zvec://file/{path}` | Read the original content of an indexed file (resource template). |

## Available MCP Prompts

| Prompt | Description |
|--------|-------------|
| `explore_codebase` | Get an overview of the project structure and key files. |
| `find_similar_code` | Find code similar to a given snippet or pattern. |
| `explain_file` | Get detailed information about a specific file and its role. |
| `debug_help` | Search for error handling patterns, logging, and debugging approaches. |

## Required workflow

1. Start with `search_project_knowledge`.
2. Treat a match as useful only if it directly names the queried symbol, file, or feature and the entry was indexed within the last 30 days; when a match is useful, use it as the primary source of context. If matches are only tangentially related, treat the result as insufficient.
3. If `search_project_knowledge` returns a tool error, unavailable state, or zero/weak results, explicitly state that the knowledge base was unavailable or did not provide a useful match, then proceed to a focused grep search.
4. If the user explicitly asks for a fresh code inspection, search directly instead of starting with the knowledge base.
5. If both `search_project_knowledge` and grep return no relevant results, tell the user explicitly that no information was found in the knowledge base or codebase, and ask for clarification such as the specific file path, symbol name, or module.
6. Avoid starting with a broad grep when the knowledge base already contains relevant context.

## Checking knowledge base status

When the user asks about the knowledge base health, indexing progress, or database status, call `get_knowledge_status` with no arguments. Use the response to answer:

- `initializationState` is `ready` — the index is fully built and searchable.
- `initializationState` is `initializing` — the index is being built; searches may return partial results.
- `initializationState` is `failed` — the index failed to build; suggest calling `initialize_project_knowledge` with `force_rebuild: true`.
- `docCount` is 0 while the state is `ready` — the indexed project has no supported source files; explain which extensions are indexed.
- `exists` is false — the database has not been created yet; suggest calling `initialize_project_knowledge`.

If `initializationState` is `initializing` and the user needs results immediately, search anyway — partial results are still useful. If the user can wait, suggest retrying after a short delay.

## Configuration (Environment Variables)

| Variable | Description | Default |
|----------|-------------|---------|
| `PROJECT_ROOT` | Root directory of the project to index | Current working directory |
| `ZVEC_EMBEDDING_MODEL` | HuggingFace model for embeddings | `Xenova/all-MiniLM-L6-v2` |
| `ZVEC_EXTENSIONS` | Comma-separated list of file extensions to index (JSON array or CSV) | Auto-detected from project type |
| `ZVEC_IGNORE_DIRS` | Comma-separated list of directories to ignore (JSON array or CSV) | Standard defaults (node_modules, .git, dist, build, etc.) |
| `ZVEC_EXCLUDE_FILES` | Comma-separated list of filenames to exclude (JSON array or CSV) | Standard config/lock files |
