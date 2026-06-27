import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { CallToolRequestSchema, ListToolsRequestSchema } from "@modelcontextprotocol/sdk/types.js";
import crypto from "crypto";
import { 
    ZVecCreateAndOpen, 
    ZVecOpen,
    ZVecCollectionSchema, 
    ZVecDataType, 
    ZVecIndexType, 
    ZVecMetricType 
} from "@zvec/zvec";

import chokidar from "chokidar";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { pipeline, env } from '@huggingface/transformers';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Fully disable logs and progress bars in stdout/stderr for MCP safety
if (env.backends && env.backends.onnx) {
    env.backends.onnx.logLevel = 'error';
}

process.on('uncaughtException', (err) => {
    console.error("[Zvec Critical Error]", err);
});

// --- PATH CONFIGURATION ---
const PROJECT_ROOT = process.env.PROJECT_ROOT || process.cwd();

// 1. The database will now be created directly at the root of the scanned project!
const DB_FILE = path.join(PROJECT_ROOT, ".zvec/knowledge.db");

// 2. Hugging Face model cache will now live globally in your home directory (~/.cache/huggingface)
// --- HUGGING FACE ISOLATION SETUP ---
const HOME_DIR = process.env.HOME || process.env.USERPROFILE;
const MODELS_CACHE = path.join(HOME_DIR, ".cache", "huggingface", "transformers");

env.localModelPath = MODELS_CACHE;
env.cacheDir = MODELS_CACHE;
env.allowRemoteModels = true;

const ALLOWED_EXTENSIONS = [".js", ".jsx", ".ts", ".tsx", ".kt", ".erl", ".hrl", ".py", ".go", ".java", ".cs", ".rb", ".php", ".cpp", ".c", ".h", ".hpp", ".rs", ".swift", ".scala"];
const IGNORED_DIRS = ["node_modules", ".git", "dist", "build", ".gradle", ".cache", ".next", ".turbo", ".vscode", ".idea"];
const EXCLUDED_FILE_NAMES = new Set([
    "package.json",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "tsconfig.json",
    "tsconfig.app.json",
    "tsconfig.node.json",
    "eslint.config.js",
    "eslint.config.cjs",
    "eslint.config.mjs",
    "vite.config.js",
    "vite.config.ts",
    "vitest.config.js",
    "vitest.config.ts",
    "babel.config.js",
    "next.config.js",
    "tailwind.config.js",
    "postcss.config.js",
    "jest.config.js",
    "init-options.json"
]);

console.error(`[Zvec Bridge] Start. DB: ${DB_FILE}`);

const schema = new ZVecCollectionSchema({
    name: "project_code",
    fields: [
        { name: "text_content", dataType: ZVecDataType.STRING },
        { name: "file_path", dataType: ZVecDataType.STRING },
        { name: "language", dataType: ZVecDataType.STRING }
    ],
    vectors: [
        {
            name: "code_embedding",
            dataType: ZVecDataType.VECTOR_FP32,
            dimension: 384,
            indexParams: {
                indexType: ZVecIndexType.HNSW,
                metricType: ZVecMetricType.COSINE
            }
        }
    ]
});

let collection = null;
let initializationPromise = null;
let initializationStarted = false;
let initializationState = "idle";
let initializationMessage = "Knowledge base is not initialized yet.";

async function runInitializationOnce(initializer) {
    if (!initializationStarted) {
        initializationStarted = true;
        initializationState = "initializing";
        initializationMessage = "Knowledge base is still initializing. Please retry shortly.";
        initializationPromise = Promise.resolve()
            .then(() => initializer())
            .then((result) => {
                initializationState = "ready";
                initializationMessage = "Knowledge base ready.";
                return result;
            })
            .catch((err) => {
                initializationState = "failed";
                initializationMessage = `Knowledge base initialization failed: ${err.message}`;
                initializationStarted = false;
                initializationPromise = null;
                throw err;
            });
    }

    return initializationPromise;
}

function removeBrokenCollectionStorage() {
    if (!fs.existsSync(DB_FILE)) return;

    const stats = fs.statSync(DB_FILE);
    if (stats.isDirectory()) {
        const lockPath = path.join(DB_FILE, "LOCK");
        if (fs.existsSync(lockPath)) {
            fs.rmSync(lockPath, { force: true });
            console.error("[Zvec Bridge] Removed outdated database lock:", lockPath);
        }
        fs.rmSync(DB_FILE, { recursive: true, force: true });
        console.error("[Zvec Bridge] Database storage removed for recreation:", DB_FILE);
    } else if (stats.isFile()) {
        fs.rmSync(DB_FILE, { force: true });
        console.error("[Zvec Bridge] Damaged database file removed for recreation:", DB_FILE);
    }
}

function ensureCollection() {
    if (collection) return collection;

    try {
        if (fs.existsSync(DB_FILE)) {
            collection = ZVecOpen(DB_FILE);
            console.error("[Zvec Bridge] Database successfully opened.");
        } else {
            collection = ZVecCreateAndOpen(DB_FILE, schema);
            console.error("[Zvec Bridge] New database successfully created.");
        }
    } catch (err) {
        const lockPath = path.join(DB_FILE, "LOCK");
        const lockFileExists = fs.existsSync(lockPath);
        if (lockFileExists) {
            try {
                fs.rmSync(lockPath, { force: true });
                console.error("[Zvec Bridge] Removed outdated database lock:", lockPath);
                try {
                    collection = ZVecOpen(DB_FILE);
                    console.error("[Zvec Bridge] Database successfully opened after clearing the lock.");
                    return collection;
                } catch (retryErr) {
                    console.error("[Zvec Bridge] Retry to open after clearing the lock failed:", retryErr.message);
                }
            } catch (cleanupErr) {
                console.error("[Zvec Bridge] Could not clear the database lock:", cleanupErr.message);
            }
        }

        try {
            removeBrokenCollectionStorage();
            collection = ZVecCreateAndOpen(DB_FILE, schema);
            console.error("[Zvec Bridge] Database successfully recreated after recovery.");
            return collection;
        } catch (rebuildErr) {
            console.error("[Zvec Bridge] Error initializing the Zvec collection:", rebuildErr);
            throw rebuildErr;
        }
    }

    return collection;
}

ensureCollection();

let extractor = null;
async function getEmbedding(text, timeoutMs = 15000) {
    const loadModel = async () => {
        if (!extractor) {
            extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
                progress_callback: () => {}
            });
        }

        const output = await extractor(text, { pooling: 'mean', normalize: true });
        return Array.from(output.data);
    };

    try {
        if (timeoutMs > 0) {
            return await Promise.race([
                loadModel(),
                new Promise((_, reject) => {
                    setTimeout(() => reject(new Error(`Embedding timed out after ${timeoutMs}ms`)), timeoutMs);
                })
            ]);
        }

        return await loadModel();
    } catch (err) {
        console.error("[Zvec Bridge] Error calculating vector:", err);
        return new Array(384).fill(0.0);
    }
}

function chunkText(text, size = 1000, overlap = 200) {
    if (text.length <= size) return [text];
    const chunks = [];
    let offset = 0;
    while (offset < text.length) {
        chunks.push(text.substring(offset, offset + size));
        offset += (size - overlap);
        if (size <= overlap) break; // Prevent infinite loop if overlap is too large
    }
    return chunks;
}

function isSupportedFile(filePath) {
    if (!filePath || shouldIgnorePath(filePath)) return false;

    const normalizedPath = filePath.toLowerCase();
    const baseName = path.basename(filePath).toLowerCase();
    if (EXCLUDED_FILE_NAMES.has(baseName)) return false;
    if (/\.d\.ts$/i.test(normalizedPath)) return false;
    if (/(^|\/)(package\.json|package-lock\.json|pnpm-lock\.yaml|yarn\.lock|tsconfig\.json|tsconfig\.app\.json|tsconfig\.node\.json|eslint\.config\.(js|cjs|mjs)|vite\.config\.(js|ts)|vitest\.config\.(js|ts)|babel\.config\.js|next\.config\.(js|mjs|ts)|tailwind\.config\.(js|cjs|mjs|ts)|postcss\.config\.(js|cjs|mjs|ts)|jest\.config\.(js|ts)|init-options\.json)$/i.test(normalizedPath)) {
        return false;
    }

    return true;
}

function shouldIgnorePath(filePath) {
    const normalized = filePath.split(path.sep).join("/");
    const isIgnoredDir = IGNORED_DIRS.some((dir) => normalized.includes(`/${dir}/`) || normalized.endsWith(`/${dir}`));
    if (isIgnoredDir) return true;

    const ext = path.extname(filePath).toLowerCase();
    return !ALLOWED_EXTENSIONS.includes(ext);
}

function escapeFilterValue(value) {
    return String(value).replace(/\\/g, "\\\\").replace(/"/g, '\\"');
}

function isLikelyGenericFile(filePath, textContent = "") {
    const normalizedPath = (filePath || "").toLowerCase();
    const normalizedText = (textContent || "").toLowerCase();

    const genericTokens = [
        "/app.tsx", "/app.jsx", "/main.tsx", "/main.jsx", "/index.tsx", "/index.jsx",
        "app shell", "main screen", "bootstrap", "entry point", "initializes the app"
    ];

    if (genericTokens.some((token) => normalizedPath.includes(token) || normalizedText.includes(token))) {
        return true;
    }

    return false;
}

function buildExplanation(query, filePath, textContent, score) {
    const normalizedQuery = String(query || "").toLowerCase();
    const normalizedPath = String(filePath || "").toLowerCase();
    const normalizedText = String(textContent || "").toLowerCase();
    const queryTokens = normalizedQuery.split(/[^a-z0-9]+/).filter(Boolean);

    const matchedTokens = queryTokens.filter((token) => token.length >= 3 && (normalizedPath.includes(token) || normalizedText.includes(token)));
    const domainHints = ["procedural", "generator", "generate", "seed", "world", "system", "planner", "engine"].filter((token) => normalizedPath.includes(token) || normalizedText.includes(token));

    let explanation = "Contains relevant code and matches the request topic.";
    if (matchedTokens.length > 0) {
        explanation = `Contains query keywords: ${matchedTokens.slice(0, 4).join(", ")}.`;
    }
    if (domainHints.length > 0) {
        explanation += ` There are also signs of domain logic: ${domainHints.slice(0, 4).join(", ")}.`;
    }
    if (score > 0) {
        explanation += ` Relevance score: ${score}.`;
    }

    return explanation;
}

function rankSearchResults(query, results, excludePaths = [], includePaths = []) {
    const normalizedQuery = String(query || "").toLowerCase();
    const queryTokens = normalizedQuery.split(/[^a-z0-9]+/).filter(Boolean);

    const excludes = (Array.isArray(excludePaths) ? excludePaths : []).map(p => p.toLowerCase());
    const includes = (Array.isArray(includePaths) ? includePaths : []).map(p => p.toLowerCase());

    return results
        .map((res) => {
            const filePath = res.fields?.file_path || "";
            const textContent = res.fields?.text_content || "";
            const combined = `${filePath}\n${textContent}`.toLowerCase();

            let score = 0;
            if (queryTokens.length === 0) return { ...res, score: 0 };

            for (const token of queryTokens) {
                if (token.length < 3) continue;
                if (combined.includes(token)) score += 3;
            }

            const pathBoost = ["procedural", "generator", "generate", "seed", "world", "system", "planner", "engine"].filter((token) => combined.includes(token));
            score += pathBoost.length * 2;

            if (isLikelyGenericFile(filePath, textContent)) score -= 5;
            if (filePath.includes("src/") || filePath.includes("lib/")) score += 1;
            if (combined.includes("function") || combined.includes("class") || combined.includes("export")) score += 1;

            return {
                ...res,
                score,
                explanation: buildExplanation(query, filePath, textContent, score)
            };
        })
        .filter((res) => {
            const filePath = res.fields?.file_path || "";
            const textContent = res.fields?.text_content || "";
            if (!isSupportedFile(filePath)) return false;
            if (isLikelyGenericFile(filePath, textContent)) return false;

            const normalizedFilePath = filePath.toLowerCase();
            if (excludes.some(ex => normalizedFilePath.includes(ex))) {
                return false;
            }
            if (includes.length > 0 && !includes.some(inc => normalizedFilePath.includes(inc))) {
                return false;
            }

            return true;
        })
        .sort((a, b) => b.score - a.score);
}

function removeFileFromIndex(filePath) {
    const resolvedPath = path.resolve(filePath);
    try {
        collection.deleteSync(`file_path == "${escapeFilterValue(resolvedPath)}"`);
    } catch (err) {
        console.error(`[Zvec Bridge] Error deleting from index ${resolvedPath}:`, err.message);
    }
}

async function indexFile(filePath) {
    const resolvedPath = path.resolve(filePath);
    try {
        try {
            collection.deleteSync(`file_path == "${escapeFilterValue(resolvedPath)}"`);
        } catch (e) {}

        if (!fs.existsSync(resolvedPath) || !isSupportedFile(resolvedPath)) return;

        const content = fs.readFileSync(resolvedPath, "utf-8");
        if (!content.trim()) return;

        const chunks = chunkText(content);
        const ext = path.extname(resolvedPath).replace(".", "").toLowerCase() || "txt";
        const relativePath = path.relative(PROJECT_ROOT, resolvedPath) || path.basename(resolvedPath);

        for (let i = 0; i < chunks.length; i++) {
            const chunk = chunks[i];
            const vector = await getEmbedding(chunk);
            
            const safeId = crypto
                .createHash("md5")
                .update(`${relativePath}_${i}`)
                .digest("hex");

            const doc = {
                id: safeId,
                vectors: { "code_embedding": vector },
                fields: {
                    "text_content": chunk,
                    "file_path": resolvedPath,
                    "language": ext
                }
            };

            if (typeof collection.insertSync === 'function') {
                collection.insertSync(doc);
            } else {
                console.error(`[Zvec Bridge] Error: insertSync not found in the collection prototype.`);
            }
        }
    } catch (err) {
        console.error(`[Zvec Bridge] Error indexing ${resolvedPath}:`, err.message);
    }
}

async function indexProject() {
    if (!fs.existsSync(PROJECT_ROOT)) return;

    const walk = async (dirPath) => {
        const entries = fs.readdirSync(dirPath, { withFileTypes: true });
        for (const entry of entries) {
            const entryPath = path.join(dirPath, entry.name);
            if (shouldIgnorePath(entryPath)) continue;

            if (entry.isDirectory()) {
                await walk(entryPath);
            } else if (entry.isFile() && isSupportedFile(entryPath)) {
                await indexFile(entryPath);
            }
        }
    };

    await walk(PROJECT_ROOT);
}

async function initializeKnowledgeBase(forceRebuild = false) {
    ensureCollection();

    if (forceRebuild) {
        try {
            collection.deleteByFilterSync("1=1");
        } catch (err) {
            console.error("[Zvec Bridge] Could not clear the existing index:", err.message);
        }
    }

    console.error("[Zvec Bridge] Starting project indexing...");
    await indexProject();
    console.error("[Zvec Bridge] Project indexing completed.");
    return {
        ok: true,
        dbFile: DB_FILE,
        projectRoot: PROJECT_ROOT
    };
}

async function ensureKnowledgeReady(options = {}) {
    const { waitForCompletion = false, timeoutMs = 0 } = options;

    const hasExistingDatabase = fs.existsSync(DB_FILE);

    if (!initializationStarted) {
        if (hasExistingDatabase) {
            initializationStarted = true;
            initializationState = "ready";
            initializationMessage = "Knowledge base already exists and is available.";
            initializationPromise = Promise.resolve({ ready: true, status: "ready", message: initializationMessage });
        } else {
            try {
                await runInitializationOnce(() => initializeKnowledgeBase(false));
            } catch (err) {
                console.error("[Zvec Bridge] Error initializing knowledge base:", err.message);
                return { ready: false, status: "failed", message: initializationMessage };
            }
        }
    }

    if (!waitForCompletion) {
        return { ready: initializationState === "ready", status: initializationState, message: initializationMessage };
    }

    if (initializationPromise && timeoutMs > 0) {
        const timeoutPromise = new Promise((resolve) => {
            setTimeout(() => resolve({ ready: false, status: "initializing", message: initializationMessage }), timeoutMs);
        });

        const result = await Promise.race([
            initializationPromise.then(() => ({ ready: true, status: "ready", message: "Knowledge base ready." })),
            timeoutPromise
        ]);

        return result;
    }

    if (initializationPromise) {
        await initializationPromise;
    }

    return { ready: initializationState === "ready", status: initializationState, message: initializationMessage };
}

function startWatcher() {
    const watcher = chokidar.watch(PROJECT_ROOT, {
        ignored: (p) => shouldIgnorePath(p),
        persistent: true,
        ignoreInitial: true,
        awaitWriteFinish: {
            stabilityThreshold: 300,
            pollInterval: 100
        }
    });

    watcher
        .on("add", (fp) => { if (isSupportedFile(fp)) indexFile(fp); })
        .on("change", (fp) => { if (isSupportedFile(fp)) indexFile(fp); })
        .on("unlink", (fp) => { if (isSupportedFile(fp)) removeFileFromIndex(fp); });
}

const server = new Server({
    name: "zvec-antigravity-bridge",
    version: "1.0.0"
}, {
    capabilities: { tools: {} }
});

server.setRequestHandler(ListToolsRequestSchema, async () => ({
    tools: [
        {
            name: "search_project_knowledge",
            description: "Search the local Zvec knowledge base for relevant code snippets and implementation examples from the indexed project files. Use this when you need to find existing code patterns, functions, or project-specific logic by semantic query.",
            inputSchema: {
                type: "object",
                properties: {
                    query: {
                        type: "string",
                        description: "Natural-language query to search the indexed project code"
                    },
                    exclude_paths: {
                        type: "array",
                        items: { type: "string" },
                        description: "Optional list of path substrings to exclude from search results (e.g. ['src_old'])"
                    },
                    include_paths: {
                        type: "array",
                        items: { type: "string" },
                        description: "Optional list of path substrings that must be present in search results (e.g. ['apps/client'])"
                    }
                },
                required: ["query"]
            }
        },
        {
            name: "initialize_project_knowledge",
            description: "Create the local Zvec database if it does not exist and index the current project files. Use this when the knowledge base is missing, was deleted, or you want to rebuild it explicitly.",
            inputSchema: {
                type: "object",
                properties: {
                    force_rebuild: {
                        type: "boolean",
                        description: "If true, clear the existing index entries before re-indexing the project"
                    }
                }
            }
        },
        {
            name: "index_file",
            description: "Force indexing or updating a specific file in the Zvec database immediately.",
            inputSchema: {
                type: "object",
                properties: {
                    file_path: {
                        type: "string",
                        description: "Relative or absolute path of the file to index."
                    }
                },
                required: ["file_path"]
            }
        },
        {
            name: "get_knowledge_status",
            description: "Get the current health and statistics of the local Zvec knowledge base (document count, indexes, disk paths, etc.).",
            inputSchema: {
                type: "object",
                properties: {}
            }
        }
    ]
}));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
    if (request.params?.name === "search_project_knowledge") {
        const query = request.params?.arguments?.query;
        const excludePaths = request.params?.arguments?.exclude_paths;
        const includePaths = request.params?.arguments?.include_paths;

        if (typeof query !== "string" || !query.trim()) {
            return { content: [{ type: "text", text: "Empty search query." }] };
        }

        console.error(`[Zvec Bridge] AI is searching: "${query}" (ex: ${JSON.stringify(excludePaths)}, in: ${JSON.stringify(includePaths)})`);

        const readiness = await ensureKnowledgeReady({ waitForCompletion: false });
        if (!readiness.ready) {
            return { content: [{ type: "text", text: readiness.message || "Knowledge base is still initializing. Please retry shortly." }] };
        }

        let queryVector;
        try {
            queryVector = await getEmbedding(query, 15000);
        } catch (err) {
            console.error("[Zvec Bridge] Error preparing query vector:", err);
            return { content: [{ type: "text", text: "Knowledge base is still warming up. The embedding model is loading; please retry shortly." }] };
        }

        const response = await ensureCollection().query({
            fieldName: "code_embedding",
            vector: queryVector,
            topk: 15 // Higher topk allows better candidates for javascript reranking/explanation
        });

        let results = [];
        if (Array.isArray(response)) {
            results = response;
        } else if (response && Array.isArray(response.rows)) {
            results = response.rows;
        } else if (response && Array.isArray(response.results)) {
            results = response.results;
        } else if (response && Array.isArray(response.documents)) {
            results = response.documents;
        } else {
            console.error("[Zvec Bridge] Unknown structure of query response:", JSON.stringify(response));
        }

        if (results.length === 0) {
            return { content: [{ type: "text", text: "Nothing found in the Zvec codebase." }] };
        }

        const rankedResults = rankSearchResults(query, results, excludePaths, includePaths);

        const formattedResults = rankedResults.map(res => `
        ---
        File: ${res.fields?.file_path || 'Unknown'}
        Why selected: ${res.explanation || 'Contains relevant code.'}
        Code:
        ${res.fields?.text_content || ''}
        `).join("\n");

        return { content: [{ type: "text", text: formattedResults || "Nothing found in the Zvec codebase." }] };
    }

    if (request.params?.name === "initialize_project_knowledge") {
        const forceRebuild = request.params?.arguments?.force_rebuild === true;
        const result = await initializeKnowledgeBase(forceRebuild);
        return { content: [{ type: "text", text: `Knowledge base initialized at ${result.dbFile}` }] };
    }

    if (request.params?.name === "index_file") {
        const filePathArg = request.params?.arguments?.file_path;
        if (typeof filePathArg !== "string" || !filePathArg.trim()) {
            return { content: [{ type: "text", text: "Empty file_path." }] };
        }
        const fullPath = path.resolve(PROJECT_ROOT, filePathArg);
        if (!fs.existsSync(fullPath)) {
            return { content: [{ type: "text", text: `File not found: ${fullPath}` }] };
        }
        if (!isSupportedFile(fullPath)) {
            return { content: [{ type: "text", text: `File format/name not supported: ${fullPath}` }] };
        }
        await indexFile(fullPath);
        return { content: [{ type: "text", text: `Successfully indexed file: ${filePathArg}` }] };
    }

    if (request.params?.name === "get_knowledge_status") {
        const col = ensureCollection();
        const stats = col.stats;
        const statusInfo = {
            dbPath: DB_FILE,
            exists: fs.existsSync(DB_FILE),
            docCount: stats?.docCount ?? 0,
            indexCompleteness: stats?.indexCompleteness ?? {},
            initializationState: initializationState,
            projectRoot: PROJECT_ROOT
        };
        return {
            content: [{
                type: "text",
                text: JSON.stringify(statusInfo, null, 2)
            }]
        };
    }

    throw new Error("Tool not found");
});

async function main() {
    const transport = new StdioServerTransport();
    await server.connect(transport);
    await ensureKnowledgeReady({ waitForCompletion: false });
    startWatcher();
}

if (process.env.ZVEC_MCP_SKIP_MAIN !== "1") {
    main().catch((err) => {
        console.error("[Zvec Bridge] Critical error:", err);
        process.exit(1);
    });
}

export { ensureCollection, isSupportedFile, shouldIgnorePath, ensureKnowledgeReady, initializeKnowledgeBase, rankSearchResults, runInitializationOnce, startWatcher };