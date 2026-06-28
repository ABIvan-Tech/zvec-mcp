import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { SSEServerTransport } from "@modelcontextprotocol/sdk/server/sse.js";
import http from "http";
import {
    CallToolRequestSchema,
    ListToolsRequestSchema,
    ListResourcesRequestSchema,
    ListResourceTemplatesRequestSchema,
    ReadResourceRequestSchema,
    ListPromptsRequestSchema,
    GetPromptRequestSchema,
    CompleteRequestSchema,
    SetLevelRequestSchema,
    SubscribeRequestSchema,
    UnsubscribeRequestSchema
} from "@modelcontextprotocol/sdk/types.js";
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
import { execSync, spawn } from "child_process";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Fully disable logs and progress bars in stdout/stderr for MCP safety
if (env.backends && env.backends.onnx) {
    env.backends.onnx.logLevel = 'error';
}

// ─── CONFIGURATION ──────────────────────────────────────────────────────────

const PROJECT_ROOT = path.resolve(process.env.PROJECT_ROOT || process.cwd());
const DB_FILE = path.join(PROJECT_ROOT, ".zvec", "knowledge.db");
const DAEMON_LOCK_FILE = path.join(PROJECT_ROOT, ".zvec", "daemon.lock");
const DAEMON_IDLE_TIMEOUT_MS = 30000; // 30s before shutdown when 0 clients

const HOME_DIR = process.env.HOME || process.env.USERPROFILE;
const MODELS_CACHE = path.join(HOME_DIR, ".cache", "huggingface", "transformers");
env.localModelPath = MODELS_CACHE;
env.cacheDir = MODELS_CACHE;
env.allowRemoteModels = true;

const EMBEDDING_MODEL = process.env.ZVEC_EMBEDDING_MODEL || "Xenova/all-MiniLM-L6-v2";

// --- Auto-detect project type and configure extensions ---

const EXTENSION_PRESETS = {
    js:    [".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"],
    python: [".py", ".pyi", ".pyx"],
    go:    [".go"],
    rust:  [".rs"],
    java:  [".java", ".kt", ".kts"],
    ruby:  [".rb", ".rake", ".erb"],
    php:   [".php"],
    c:     [".c", ".h", ".cpp", ".hpp", ".cc", ".cxx"],
    swift: [".swift"],
    scala: [".scala", ".sc"],
    erlang: [".erl", ".hrl"],
    dotnet: [".cs", ".fs", ".vb"],
};

const DEFAULT_IGNORED_DIRS = [
    "node_modules", ".git", ".zvec", "dist", "build", ".gradle",
    ".cache", ".next", ".turbo", ".vscode", ".idea", "__pycache__",
    ".venv", "venv", "target", "vendor", ".dart_tool", "Pods"
];

const DEFAULT_EXCLUDED_FILE_NAMES = new Set([
    "package.json", "package-lock.json", "pnpm-lock.yaml", "yarn.lock",
    "tsconfig.json", "tsconfig.app.json", "tsconfig.node.json",
    "eslint.config.js", "eslint.config.cjs", "eslint.config.mjs",
    "vite.config.js", "vite.config.ts", "vitest.config.js", "vitest.config.ts",
    "babel.config.js", "next.config.js", "tailwind.config.js",
    "postcss.config.js", "jest.config.js", "init-options.json",
    "Cargo.toml", "Cargo.lock", "go.mod", "go.sum",
    "pom.xml", "build.gradle", "build.gradle.kts",
    "Gemfile", "Gemfile.lock", "composer.json", "composer.lock",
    "requirements.txt", "pyproject.toml", "setup.py", "setup.cfg",
    "Makefile", "CMakeLists.txt", "Dockerfile", "docker-compose.yml",
    ".env", ".env.local", ".env.example"
]);

function detectProjectExtensions(root) {
    const detectors = [
        { file: "package.json",    exts: EXTENSION_PRESETS.js },
        { file: "pyproject.toml",  exts: EXTENSION_PRESETS.python },
        { file: "setup.py",        exts: EXTENSION_PRESETS.python },
        { file: "go.mod",          exts: EXTENSION_PRESETS.go },
        { file: "Cargo.toml",      exts: EXTENSION_PRESETS.rust },
        { file: "pom.xml",         exts: EXTENSION_PRESETS.java },
        { file: "build.gradle",    exts: EXTENSION_PRESETS.java },
        { file: "Gemfile",         exts: EXTENSION_PRESETS.ruby },
        { file: "composer.json",   exts: EXTENSION_PRESETS.php },
        { file: "Package.swift",   exts: EXTENSION_PRESETS.swift },
        { file: "build.sbt",       exts: EXTENSION_PRESETS.scala },
    ];

    for (const { file, exts } of detectors) {
        if (fs.existsSync(path.join(root, file))) return exts;
    }

    try {
        const entries = fs.readdirSync(root, { withFileTypes: true });
        const extCounts = {};
        for (const e of entries) {
            if (!e.isFile()) continue;
            const ext = path.extname(e.name).toLowerCase();
            if (ext) extCounts[ext] = (extCounts[ext] || 0) + 1;
        }
        const sorted = Object.entries(extCounts).sort((a, b) => b[1] - a[1]);
        if (sorted.length > 0 && sorted[0][1] >= 2) return Object.keys(EXTENSION_PRESETS).flat();
    } catch {}

    return Object.keys(EXTENSION_PRESETS).flat();
}

function parseEnvList(value, fallback) {
    if (!value || typeof value !== "string") return fallback;
    try {
        const parsed = JSON.parse(value);
        return Array.isArray(parsed) ? parsed : fallback;
    } catch {
        return value.split(",").map(s => s.trim()).filter(Boolean);
    }
}

const ALLOWED_EXTENSIONS = (() => {
    const envVal = process.env.ZVEC_EXTENSIONS;
    if (envVal) return parseEnvList(envVal, []);
    return detectProjectExtensions(PROJECT_ROOT);
})();

const IGNORED_DIRS = (() => {
    const envVal = process.env.ZVEC_IGNORE_DIRS;
    if (envVal) return parseEnvList(envVal, DEFAULT_IGNORED_DIRS);
    return DEFAULT_IGNORED_DIRS;
})();

const EXCLUDED_FILE_NAMES = (() => {
    const envVal = process.env.ZVEC_EXCLUDE_FILES;
    if (envVal) return new Set(parseEnvList(envVal, [...DEFAULT_EXCLUDED_FILE_NAMES]));
    return DEFAULT_EXCLUDED_FILE_NAMES;
})();

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
let processHandlersRegistered = false;
let totalIndexedFiles = 0;
let serverTransport = null;
const sessionTransports = new Map(); // sessionId → SSEServerTransport (daemon mode)
let idleTimeout = null;
let isDaemonMode = false;

function registerProcessHandlers() {
    if (processHandlersRegistered) return;
    processHandlersRegistered = true;

    process.on('uncaughtException', (err) => {
        console.error("[Zvec Critical Error]", err);
        process.exit(1);
    });

    process.on('unhandledRejection', (reason) => {
        console.error("[Zvec Critical Rejection]", reason);
        process.exit(1);
    });
}

function beginInitialization(initializer) {
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

    return initializationPromise;
}

async function runInitializationOnce(initializer) {
    if (!initializationStarted) {
        return beginInitialization(initializer);
    }

    return initializationPromise;
}

async function rerunInitialization(initializer) {
    if (initializationState === "initializing" && initializationPromise) {
        return initializationPromise;
    }

    return beginInitialization(initializer);
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
        fs.rmSync(DB_FILE, { recursive: true, force: true, maxRetries: 5, retryDelay: 100 });
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

let extractor = null;
async function getEmbedding(text, timeoutMs = 15000) {
    const loadModel = async () => {
        if (!extractor) {
            extractor = await pipeline('feature-extraction', EMBEDDING_MODEL, {
                progress_callback: () => {}
            });
        }

        const output = await extractor(text, { pooling: 'mean', normalize: true });
        return Array.from(output.data);
    };

    try {
        if (timeoutMs <= 0) {
            return await loadModel();
        }

        let timeoutId;
        try {
            return await Promise.race([
                loadModel(),
                new Promise((_, reject) => {
                    timeoutId = setTimeout(() => reject(new Error(`Embedding timed out after ${timeoutMs}ms`)), timeoutMs);
                })
            ]);
        } finally {
            if (timeoutId) clearTimeout(timeoutId);
        }
    } catch (err) {
        console.error("[Zvec Bridge] Error calculating vector:", err);
        throw err;
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

function isExistingDirectory(filePath) {
    if (!filePath || !fs.existsSync(filePath)) return false;

    try {
        return fs.statSync(filePath).isDirectory();
    } catch {
        return false;
    }
}

function isSupportedFile(filePath) {
    if (!filePath || shouldIgnorePath(filePath) || isExistingDirectory(filePath)) return false;

    const normalizedPath = filePath.toLowerCase();
    const baseName = path.basename(filePath).toLowerCase();
    const ext = path.extname(filePath).toLowerCase();

    if (!ALLOWED_EXTENSIONS.includes(ext)) return false;
    if (EXCLUDED_FILE_NAMES.has(baseName)) return false;
    if (/\.d\.ts$/i.test(normalizedPath)) return false;
    if (/(^|\/)(package\.json|package-lock\.json|pnpm-lock\.yaml|yarn\.lock|tsconfig\.json|tsconfig\.app\.json|tsconfig\.node\.json|eslint\.config\.(js|cjs|mjs)|vite\.config\.(js|ts)|vitest\.config\.(js|ts)|babel\.config\.js|next\.config\.(js|mjs|ts)|tailwind\.config\.(js|cjs|mjs|ts)|postcss\.config\.(js|cjs|mjs|ts)|jest\.config\.(js|ts)|init-options\.json)$/i.test(normalizedPath)) {
        return false;
    }

    return true;
}

function shouldIgnorePath(filePath) {
    if (!filePath) return true;

    const normalized = filePath.split(path.sep).join("/");
    const isIgnoredDir = IGNORED_DIRS.some((dir) => normalized.includes(`/${dir}/`) || normalized.endsWith(`/${dir}`));
    if (isIgnoredDir) return true;

    if (isExistingDirectory(filePath)) return false;

    const ext = path.extname(filePath).toLowerCase();
    if (!ext) return false;

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
        const col = ensureCollection();
        col.deleteSync(`file_path == "${escapeFilterValue(resolvedPath)}"`);
    } catch (err) {
        console.error(`[Zvec Bridge] Error deleting from index ${resolvedPath}:`, err.message);
    }
}

function yieldToEventLoop() {
    return new Promise((resolve) => setImmediate(resolve));
}

async function indexFile(filePath) {
    const resolvedPath = path.resolve(filePath);
    try {
        const col = ensureCollection();

        try {
            col.deleteSync(`file_path == "${escapeFilterValue(resolvedPath)}"`);
        } catch (e) {}

        let exists = false;
        try {
            await fs.promises.access(resolvedPath);
            exists = true;
        } catch {}

        if (!exists || !isSupportedFile(resolvedPath)) return;

        const content = await fs.promises.readFile(resolvedPath, "utf-8");
        if (!content.trim()) return;

        const chunks = chunkText(content);
        const ext = path.extname(resolvedPath).replace(".", "").toLowerCase() || "txt";
        const relativePath = path.relative(PROJECT_ROOT, resolvedPath) || path.basename(resolvedPath);

        for (let i = 0; i < chunks.length; i++) {
            const chunk = chunks[i];
            let vector;

            try {
                vector = await getEmbedding(chunk);
            } catch (err) {
                console.error(`[Zvec Bridge] Skipping chunk ${i + 1}/${chunks.length} for ${resolvedPath}:`, err.message);
                continue;
            }
            
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

            if (typeof col.insertSync === 'function') {
                col.insertSync(doc);
            } else {
                console.error(`[Zvec Bridge] Error: insertSync not found in the collection prototype.`);
            }

            await yieldToEventLoop();
        }
    } catch (err) {
        console.error(`[Zvec Bridge] Error indexing ${resolvedPath}:`, err.message);
    }
}

async function indexProject() {
    totalIndexedFiles = 0;
    try { await fs.promises.access(PROJECT_ROOT); } catch { return; }

    const walk = async (dirPath) => {
        const entries = await fs.promises.readdir(dirPath, { withFileTypes: true });
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
    const col = ensureCollection();

    if (forceRebuild) {
        try {
            col.deleteByFilterSync("1=1");
        } catch (err) {
            console.error("[Zvec Bridge] Could not clear the existing index:", err.message);
        }
    }

    console.error("[Zvec Bridge] Starting project indexing...");
    await indexProject();
        console.error(`[Zvec Bridge] Project indexing completed. Files indexed: ${totalIndexedFiles}`);
    return { ok: true, dbFile: DB_FILE, projectRoot: PROJECT_ROOT, filesIndexed: totalIndexedFiles };
}

async function ensureKnowledgeReady(options = {}) {
    const { waitForCompletion = false, timeoutMs = 0 } = options;

    const col = ensureCollection();
    const docCount = col.stats?.docCount ?? 0;
    const hasIndexedDocuments = docCount > 0;
    const shouldStartInitialization = !initializationStarted;

    if (shouldStartInitialization) {
        if (!hasIndexedDocuments) {
            const initPromise = runInitializationOnce(() => initializeKnowledgeBase(false));
            if (!waitForCompletion) {
                initPromise.catch((err) => {
                    console.error("[Zvec Bridge] Error initializing knowledge base:", err.message);
                });
            }
        } else {
            initializationState = "ready";
            initializationMessage = "Knowledge base already exists. Skipping re-index on startup.";
            initializationStarted = true;
            initializationPromise = Promise.resolve({ ok: true, dbFile: DB_FILE });
        }
    }

    if (!waitForCompletion) {
        if (initializationState === "failed") {
            return { ready: false, status: "failed", message: initializationMessage };
        }

        if (initializationState === "initializing") {
            if (hasIndexedDocuments) {
                return {
                    ready: true,
                    status: "ready",
                    message: "Knowledge base already exists and is being refreshed in the background."
                };
            }

            return { ready: false, status: "initializing", message: initializationMessage };
        }

        return { ready: true, status: "ready", message: initializationMessage };
    }

    if (initializationPromise && timeoutMs > 0) {
        const timeoutPromise = new Promise((resolve) => {
            setTimeout(() => {
                if (hasIndexedDocuments) {
                    resolve({
                        ready: true,
                        status: "ready",
                        message: "Knowledge base already exists and is being refreshed in the background."
                    });
                    return;
                }

                resolve({ ready: false, status: "initializing", message: initializationMessage });
            }, timeoutMs);
        });

        const result = await Promise.race([
            initializationPromise
                .then(() => ({ ready: true, status: "ready", message: "Knowledge base ready." }))
                .catch(() => ({ ready: false, status: "failed", message: initializationMessage })),
            timeoutPromise
        ]);

        return result;
    }

    if (initializationPromise) {
        try {
            await initializationPromise;
        } catch (err) {
            console.error("[Zvec Bridge] Error initializing knowledge base:", err.message);
            return { ready: false, status: "failed", message: initializationMessage };
        }
    }

    return { ready: initializationState === "ready", status: initializationState, message: initializationMessage };
}

function startWatcher() {
    const watcher = chokidar.watch(PROJECT_ROOT, {
        ignored: (p) => shouldIgnorePath(p),
        persistent: true,
        ignoreInitial: true,
        awaitWriteFinish: { stabilityThreshold: 300, pollInterval: 100 }
    });

    async function notifyResourceChanged(filePath) {
        const relativePath = path.relative(PROJECT_ROOT, filePath);
        const notification = {
            method: "notifications/resources/updated",
            params: { uri: `zvec://file/${relativePath}` }
        };

        if (isDaemonMode) {
            for (const [sid, transport] of sessionTransports) {
                try { await transport.send(notification); } catch (err) {
                    console.error(`[Zvec Bridge] Failed to notify session ${sid}:`, err.message);
                }
            }
        } else if (serverTransport) {
            try { await serverTransport.send(notification); } catch (err) {
                console.error("[Zvec Bridge] Failed to send resource notification:", err.message);
            }
        }
    }

    watcher
        .on("add", (fp) => { if (isSupportedFile(fp)) { indexFile(fp); notifyResourceChanged(fp); } })
        .on("change", (fp) => { if (isSupportedFile(fp)) { indexFile(fp); notifyResourceChanged(fp); } })
        .on("unlink", (fp) => { if (isSupportedFile(fp)) { removeFileFromIndex(fp); notifyResourceChanged(fp); } });
}

function getProjectInfo() {
    const info = {
        name: path.basename(PROJECT_ROOT),
        root: PROJECT_ROOT,
        dbFile: DB_FILE,
        embeddingModel: EMBEDDING_MODEL,
        allowedExtensions: ALLOWED_EXTENSIONS,
        ignoredDirs: IGNORED_DIRS,
        excludedFileCount: EXCLUDED_FILE_NAMES.size,
    };
    const detectors = [
        { file: "package.json",     type: "Node.js/JavaScript" },
        { file: "pyproject.toml",   type: "Python" },
        { file: "setup.py",         type: "Python" },
        { file: "go.mod",           type: "Go" },
        { file: "Cargo.toml",       type: "Rust" },
        { file: "pom.xml",          type: "Java (Maven)" },
        { file: "build.gradle",     type: "Java (Gradle)" },
        { file: "build.gradle.kts", type: "Kotlin" },
        { file: "Gemfile",          type: "Ruby" },
        { file: "composer.json",    type: "PHP" },
        { file: "Package.swift",    type: "Swift" },
        { file: "build.sbt",        type: "Scala" },
    ];
    for (const { file, type } of detectors) {
        if (fs.existsSync(path.join(PROJECT_ROOT, file))) { info.projectType = type; break; }
    }
    if (!info.projectType) {
        try {
            const entries = fs.readdirSync(PROJECT_ROOT, { withFileTypes: true });
            const extCounts = {};
            for (const e of entries) {
                if (!e.isFile()) continue;
                const ext = path.extname(e.name).toLowerCase();
                if (ext) extCounts[ext] = (extCounts[ext] || 0) + 1;
            }
            const sorted = Object.entries(extCounts).sort((a, b) => b[1] - a[1]);
            if (sorted.length > 0) {
                const topExt = sorted[0][0];
                const knownMap = { ".js": "JavaScript/Node.js", ".ts": "TypeScript/Node.js", ".py": "Python", ".go": "Go", ".rs": "Rust", ".java": "Java", ".rb": "Ruby", ".php": "PHP", ".cs": ".NET", ".swift": "Swift", ".scala": "Scala" };
                info.projectType = knownMap[topExt] || `Unknown (${topExt})`;
            } else {
                info.projectType = "Unknown";
            }
        } catch { info.projectType = "Unknown"; }
    }
    try { info.gitRemote = execSync('git remote get-url origin', { cwd: PROJECT_ROOT, encoding: 'utf-8', timeout: 3000 }).trim(); } catch {}
    return info;
}

const server = new Server({
    name: "zvec-project-knowledge",
    version: "2.0.0"
}, {
    capabilities: {
        tools: {},
        resources: {},
        prompts: {},
        completions: {},
        logging: {}
    }
});

server.setRequestHandler(ListToolsRequestSchema, async () => ({
    tools: [
        {
            name: "search_project_knowledge",
            description: "Semantic search through the indexed project codebase. Returns relevant code snippets with file paths and explanations. Use this as the first step before any broad repository search.",
            inputSchema: {
                type: "object",
                properties: {
                    query: { type: "string", description: "Natural-language or keyword query to search the indexed code" },
                    exclude_paths: { type: "array", items: { type: "string" }, description: "Path substrings to exclude from results" },
                    include_paths: { type: "array", items: { type: "string" }, description: "Path substrings that must be present in results" }
                },
                required: ["query"]
            }
        },
        {
            name: "initialize_project_knowledge",
            description: "Create or rebuild the Zvec knowledge base index. Call this when the index is missing, stale, or corrupted.",
            inputSchema: {
                type: "object",
                properties: {
                    force_rebuild: { type: "boolean", description: "If true, clear the existing index before re-indexing" }
                }
            }
        },
        {
            name: "index_file",
            description: "Force indexing or updating a specific file in the knowledge base immediately.",
            inputSchema: {
                type: "object",
                properties: {
                    file_path: { type: "string", description: "Relative or absolute path of the file to index" }
                },
                required: ["file_path"]
            }
        },
        {
            name: "get_knowledge_status",
            description: "Get health, statistics, and configuration of the knowledge base (document count, init state, project root, detected extensions).",
            inputSchema: { type: "object", properties: {} }
        }
    ]
}));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const { name, arguments: args } = request.params || {};

    if (name === "search_project_knowledge") {
        const query = args?.query;
        const excludePaths = args?.exclude_paths;
        const includePaths = args?.include_paths;

        if (typeof query !== "string" || !query.trim()) {
            return { content: [{ type: "text", text: "Empty search query." }] };
        }

        console.error(`[Zvec Bridge] Searching: "${query}" (exclude: ${JSON.stringify(excludePaths)}, include: ${JSON.stringify(includePaths)})`);

        const readiness = await ensureKnowledgeReady({ waitForCompletion: false });
        if (!readiness.ready) {
            return { content: [{ type: "text", text: readiness.message || "Knowledge base is still initializing. Please retry shortly." }] };
        }

        let queryVector;
        try { queryVector = await getEmbedding(query, 15000); } catch (err) {
            console.error("[Zvec Bridge] Error preparing query vector:", err);
            return { content: [{ type: "text", text: "Knowledge base is still warming up. The embedding model is loading; please retry shortly." }] };
        }

        const response = await ensureCollection().query({
            fieldName: "code_embedding", vector: queryVector, topk: 15
        });

        let results = [];
        if (Array.isArray(response)) results = response;
        else if (response?.rows) results = response.rows;
        else if (response?.results) results = response.results;
        else if (response?.documents) results = response.documents;
        else console.error("[Zvec Bridge] Unknown query response structure:", JSON.stringify(response));

        if (results.length === 0) {
            return { content: [{ type: "text", text: "Nothing found in the Zvec codebase." }] };
        }

        const ranked = rankSearchResults(query, results, excludePaths, includePaths);
        const formatted = ranked.map(res => `
---
File: ${res.fields?.file_path || 'Unknown'}
Why selected: ${res.explanation || 'Contains relevant code.'}
Code:
${res.fields?.text_content || ''}
`).join("\n");

        return { content: [{ type: "text", text: formatted || "Nothing found in the Zvec codebase." }] };
    }

    if (name === "initialize_project_knowledge") {
        const forceRebuild = args?.force_rebuild === true;
        const result = await rerunInitialization(() => initializeKnowledgeBase(forceRebuild));
        return { content: [{ type: "text", text: `Knowledge base initialized at ${result.dbFile}. Files indexed: ${result.filesIndexed}` }] };
    }

    if (name === "index_file") {
        const filePathArg = args?.file_path;
        if (typeof filePathArg !== "string" || !filePathArg.trim()) {
            return { content: [{ type: "text", text: "Empty file_path." }] };
        }
        const fullPath = path.resolve(PROJECT_ROOT, filePathArg);
        if (!fs.existsSync(fullPath)) return { content: [{ type: "text", text: `File not found: ${fullPath}` }] };
        if (!isSupportedFile(fullPath)) return { content: [{ type: "text", text: `File format/name not supported: ${fullPath}` }] };
        await indexFile(fullPath);
        return { content: [{ type: "text", text: `Successfully indexed: ${filePathArg}` }] };
    }

    if (name === "get_knowledge_status") {
        const col = ensureCollection();
        const stats = col.stats;
        const info = getProjectInfo();
        const statusInfo = {
            ...info,
            docCount: stats?.docCount ?? 0,
            indexCompleteness: stats?.indexCompleteness ?? {},
            initializationState,
            totalIndexedFiles
        };
        return { content: [{ type: "text", text: JSON.stringify(statusInfo, null, 2) }] };
    }

    throw new Error(`Tool not found: ${name}`);
});

// ── ListResources ───────────────────────────────────────────────────────────

server.setRequestHandler(ListResourcesRequestSchema, async () => ({
    resources: [
        {
            uri: "zvec://project/info",
            name: "Project Information",
            description: "Metadata about the indexed project: root path, detected type, extensions, git remote.",
            mimeType: "application/json"
        },
        {
            uri: "zvec://knowledge/status",
            name: "Knowledge Base Status",
            description: "Current health and statistics of the Zvec knowledge base.",
            mimeType: "application/json"
        }
    ]
}));

// ── ListResourceTemplates ───────────────────────────────────────────────────

server.setRequestHandler(ListResourceTemplatesRequestSchema, async () => ({
    resourceTemplates: [
        {
            uriTemplate: "zvec://file/{path}",
            name: "Indexed File Content",
            description: "Read the original content of a file that has been indexed in the knowledge base.",
            mimeType: "text/plain"
        }
    ]
}));

// ── ReadResource ────────────────────────────────────────────────────────────

server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
    const { uri } = request.params;

    if (uri === "zvec://project/info") {
        return {
            contents: [{ uri, mimeType: "application/json", text: JSON.stringify(getProjectInfo(), null, 2) }]
        };
    }

    if (uri === "zvec://knowledge/status") {
        const col = ensureCollection();
        const stats = col.stats;
        const status = {
            dbPath: DB_FILE,
            exists: fs.existsSync(DB_FILE),
            docCount: stats?.docCount ?? 0,
            initializationState,
            totalIndexedFiles
        };
        return {
            contents: [{ uri, mimeType: "application/json", text: JSON.stringify(status, null, 2) }]
        };
    }

    if (uri.startsWith("zvec://file/")) {
        const filePath = decodeURIComponent(uri.slice("zvec://file/".length));
        const fullPath = path.resolve(PROJECT_ROOT, filePath);
        if (!fs.existsSync(fullPath)) throw new Error(`File not found: ${filePath}`);
        const content = await fs.promises.readFile(fullPath, "utf-8");
        return {
            contents: [{ uri, mimeType: "text/plain", text: content }]
        };
    }

    throw new Error(`Unknown resource URI: ${uri}`);
});

// ── ListPrompts ─────────────────────────────────────────────────────────────

server.setRequestHandler(ListPromptsRequestSchema, async () => ({
    prompts: [
        {
            name: "explore_codebase",
            description: "Get an overview of the project structure and key files.",
            arguments: [
                { name: "topic", description: "Specific topic or area to focus on (optional)", required: false }
            ]
        },
        {
            name: "find_similar_code",
            description: "Find code similar to a given snippet or pattern.",
            arguments: [
                { name: "code_snippet", description: "Code pattern or description to find similar implementations for", required: true }
            ]
        },
        {
            name: "explain_file",
            description: "Get detailed information about a specific file and its role in the project.",
            arguments: [
                { name: "file_path", description: "Path to the file to explain", required: true }
            ]
        },
        {
            name: "debug_help",
            description: "Search for error handling patterns, logging, and debugging approaches in the project.",
            arguments: [
                { name: "error_description", description: "Description of the error or issue", required: false }
            ]
        }
    ]
}));

// ── GetPrompt ───────────────────────────────────────────────────────────────

server.setRequestHandler(GetPromptRequestSchema, async (request) => {
    const { name, arguments: args } = request.params;

    if (name === "explore_codebase") {
        const topic = args?.topic || "general structure";
        const readiness = await ensureKnowledgeReady({ waitForCompletion: false });
        if (!readiness.ready) {
            return { messages: [{ role: "user", content: { type: "text", text: `Knowledge base is initializing. Please retry shortly to explore: ${topic}` } }] };
        }
        let queryVector;
        try { queryVector = await getEmbedding(`project structure ${topic}`, 15000); } catch {
            return { messages: [{ role: "user", content: { type: "text", text: "Embedding model still loading. Please retry shortly." } }] };
        }
        const response = await ensureCollection().query({ fieldName: "code_embedding", vector: queryVector, topk: 5 });
        let results = Array.isArray(response) ? response : response?.rows || response?.results || response?.documents || [];
        const fileList = results.map(r => r.fields?.file_path).filter(Boolean).join("\n");
        return {
            messages: [{
                role: "user",
                content: { type: "text", text: `Explore the project structure focusing on: ${topic}\n\nKey files found:\n${fileList || "No files found yet."}\n\nPlease provide an overview of the project structure and how these files relate to each other.` }
            }]
        };
    }

    if (name === "find_similar_code") {
        const snippet = args?.code_snippet;
        if (!snippet) return { messages: [{ role: "user", content: { type: "text", text: "Please provide a code snippet or description to search for." } }] };
        const readiness = await ensureKnowledgeReady({ waitForCompletion: false });
        if (!readiness.ready) {
            return { messages: [{ role: "user", content: { type: "text", text: "Knowledge base is initializing. Please retry shortly." } }] };
        }
        let queryVector;
        try { queryVector = await getEmbedding(snippet, 15000); } catch {
            return { messages: [{ role: "user", content: { type: "text", text: "Embedding model still loading. Please retry shortly." } }] };
        }
        const response = await ensureCollection().query({ fieldName: "code_embedding", vector: queryVector, topk: 10 });
        let results = Array.isArray(response) ? response : response?.rows || response?.results || response?.documents || [];
        const ranked = rankSearchResults(snippet, results);
        const found = ranked.slice(0, 5).map(r => `File: ${r.fields?.file_path}\nCode:\n${r.fields?.text_content}`).join("\n---\n");
        return {
            messages: [{
                role: "user",
                content: { type: "text", text: `Find code similar to:\n${snippet}\n\nResults:\n${found || "No similar code found."}\n\nAnalyze these results and explain how they relate to the provided snippet.` }
            }]
        };
    }

    if (name === "explain_file") {
        const filePath = args?.file_path;
        if (!filePath) return { messages: [{ role: "user", content: { type: "text", text: "Please provide a file path." } }] };
        const fullPath = path.resolve(PROJECT_ROOT, filePath);
        let content = "";
        try { content = await fs.promises.readFile(fullPath, "utf-8"); } catch {
            return { messages: [{ role: "user", content: { type: "text", text: `Could not read file: ${filePath}` } }] };
        }
        return {
            messages: [{
                role: "user",
                content: { type: "text", text: `Explain the following file and its role in the project:\n\nFile: ${filePath}\n\n\`\`\`\n${content.slice(0, 8000)}\n\`\`\`` }
            }]
        };
    }

    if (name === "debug_help") {
        const errorDesc = args?.error_description || "general debugging";
        const readiness = await ensureKnowledgeReady({ waitForCompletion: false });
        if (!readiness.ready) {
            return { messages: [{ role: "user", content: { type: "text", text: "Knowledge base is initializing. Please retry shortly." } }] };
        }
        let queryVector;
        try { queryVector = await getEmbedding(`error handling logging ${errorDesc}`, 15000); } catch {
            return { messages: [{ role: "user", content: { type: "text", text: "Embedding model still loading. Please retry shortly." } }] };
        }
        const response = await ensureCollection().query({ fieldName: "code_embedding", vector: queryVector, topk: 5 });
        let results = Array.isArray(response) ? response : response?.rows || response?.results || response?.documents || [];
        const found = results.map(r => `File: ${r.fields?.file_path}\n${r.fields?.text_content}`).join("\n---\n");
        return {
            messages: [{
                role: "user",
                content: { type: "text", text: `Help debug: ${errorDesc}\n\nRelated error handling code:\n${found || "No relevant patterns found."}\n\nAnalyze these patterns and suggest debugging approaches.` }
            }]
        };
    }

    throw new Error(`Prompt not found: ${name}`);
});

// ── Complete (autocomplete) ─────────────────────────────────────────────────

server.setRequestHandler(CompleteRequestSchema, async (request) => {
    const { ref, argument } = request.params;
    let values = [];

    if (ref.type === "ref/prompt") {
        if (ref.name === "explore_codebase") {
            values = ["general structure", "authentication", "database", "API", "testing", "configuration"];
        } else if (ref.name === "find_similar_code") {
            values = ["error handling", "API routes", "database queries", "authentication", "middleware"];
        } else if (ref.name === "explain_file") {
            const col = ensureCollection();
            const stats = col.stats;
            if (stats?.docCount > 0) {
                try {
                    const response = await col.query({
                        fieldName: "code_embedding",
                        vector: await getEmbedding(argument?.value || "", 5000),
                        topk: 10
                    });
                    const results = Array.isArray(response) ? response : response?.rows || [];
                    values = results.map(r => r.fields?.file_path).filter(Boolean);
                } catch {}
            }
        }
    } else if (ref.type === "ref/resource") {
        if (ref.uri === "zvec://file/{path}") {
            const col = ensureCollection();
            if (col.stats?.docCount > 0) {
                try {
                    const response = await col.query({
                        fieldName: "code_embedding",
                        vector: await getEmbedding(argument?.value || "", 5000),
                        topk: 10
                    });
                    const results = Array.isArray(response) ? response : response?.rows || [];
                    values = results.map(r => r.fields?.file_path).filter(Boolean);
                } catch {}
            }
        }
    }

    const filtered = argument?.value
        ? values.filter(v => v.toLowerCase().startsWith(argument.value.toLowerCase()))
        : values;

    return {
        completion: { values: filtered.slice(0, 10), hasMore: filtered.length > 10 }
    };
});

// ── SetLevel (logging) ──────────────────────────────────────────────────────

server.setRequestHandler(SetLevelRequestSchema, async (request) => {
    const level = request.params?.level;
    console.error(`[Zvec Bridge] Log level set to: ${level}`);
    return {};
});

// ── Subscribe/Unsubscribe (resource change notifications) ───────────────────

const subscribedResources = new Set();

server.setRequestHandler(SubscribeRequestSchema, async (request) => {
    const { uri } = request.params;
    subscribedResources.add(uri);
    console.error(`[Zvec Bridge] Client subscribed to: ${uri}`);
    return {};
});

server.setRequestHandler(UnsubscribeRequestSchema, async (request) => {
    const { uri } = request.params;
    subscribedResources.delete(uri);
    console.error(`[Zvec Bridge] Client unsubscribed from: ${uri}`);
    return {};
});

// ─── DAEMON MODE ────────────────────────────────────────────────────────────

function readDaemonLock() {
    try {
        const data = fs.readFileSync(DAEMON_LOCK_FILE, "utf-8");
        const parsed = JSON.parse(data);
        if (parsed && parsed.pid && parsed.port) return parsed;
    } catch {}
    return null;
}

function writeDaemonLock(port) {
    const dir = path.dirname(DAEMON_LOCK_FILE);
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(DAEMON_LOCK_FILE, JSON.stringify({ pid: process.pid, port }));
}

function removeDaemonLock() {
    try { fs.rmSync(DAEMON_LOCK_FILE, { force: true }); } catch {}
}

function scheduleShutdown() {
    if (idleTimeout) clearTimeout(idleTimeout);
    idleTimeout = setTimeout(() => {
        const count = sessionTransports.size;
        if (count > 0) {
            console.error(`[Zvec Bridge] Cancel shutdown — ${count} client(s) still connected`);
            return;
        }
        console.error(`[Zvec Bridge] Idle timeout — ${DAEMON_IDLE_TIMEOUT_MS}ms with 0 clients. Shutting down.`);
        removeDaemonLock();
        process.exit(0);
    }, DAEMON_IDLE_TIMEOUT_MS);
    idleTimeout.unref();
}

function cancelShutdown() {
    if (idleTimeout) { clearTimeout(idleTimeout); idleTimeout = null; }
}

async function startDaemon(port) {
    isDaemonMode = true;

    const existing = readDaemonLock();
    if (existing) {
        try {
            process.kill(existing.pid, 0);
            console.error(`[Zvec Bridge] Daemon already running on port ${existing.port} (PID ${existing.pid}).`);
            process.exit(1);
        } catch {
            console.error(`[Zvec Bridge] Removing stale lock file from PID ${existing.pid}.`);
            removeDaemonLock();
        }
    }

    // Init knowledge base and watcher once
    await ensureKnowledgeReady({ waitForCompletion: false });
    startWatcher();

    const httpServer = http.createServer(async (req, res) => {
        const url = new URL(req.url, `http://localhost:${port}`);

        if (req.method === "GET" && url.pathname === "/sse") {
            const transport = new SSEServerTransport("/message", res);
            const sessionId = transport.sessionId;
            sessionTransports.set(sessionId, transport);
            cancelShutdown();

            transport.onclose = () => {
                sessionTransports.delete(sessionId);
                console.error(`[Zvec Bridge] Client ${sessionId} disconnected (${sessionTransports.size} remaining)`);
                if (sessionTransports.size === 0) scheduleShutdown();
            };

            try {
                await server.connect(transport);
            } catch (err) {
                console.error(`[Zvec Bridge] Error connecting SSE transport for ${sessionId}:`, err);
                sessionTransports.delete(sessionId);
            }
            return;
        }

        if (req.method === "POST" && url.pathname === "/message") {
            const sessionId = url.searchParams.get("sessionId");
            if (!sessionId) {
                res.writeHead(400);
                res.end("Missing sessionId");
                return;
            }
            const transport = sessionTransports.get(sessionId);
            if (!transport) {
                res.writeHead(404);
                res.end("Session not found");
                return;
            }
            await transport.handlePostMessage(req, res);
            return;
        }

        if (req.method === "GET" && url.pathname === "/health") {
            res.writeHead(200, { "Content-Type": "application/json" });
            res.end(JSON.stringify({
                pid: process.pid,
                port,
                clients: sessionTransports.size,
                status: initializationState,
                db: fs.existsSync(DB_FILE),
                uptime: process.uptime()
            }));
            return;
        }

        res.writeHead(404);
        res.end("Not found");
    });

    const shutdown = () => {
        console.error("[Zvec Bridge] Daemon shutting down...");
        removeDaemonLock();
        for (const [sid, transport] of sessionTransports) {
            transport.close().catch(() => {});
        }
        httpServer.close(() => process.exit(0));
    };

    process.on("SIGTERM", shutdown);
    process.on("SIGINT", shutdown);

    httpServer.listen(port, "127.0.0.1", () => {
        writeDaemonLock(port);
        console.error(`[Zvec Bridge] Daemon ready on http://127.0.0.1:${port} (PID ${process.pid})`);
    });
}

// ─── PROXY MODE ─────────────────────────────────────────────────────────────

async function startProxy(port) {
    const url = new URL(`http://127.0.0.1:${port}/sse`);

    // Use SSEClientTransport from SDK
    const { SSEClientTransport } = await import("@modelcontextprotocol/sdk/client/sse.js");
    const transport = new SSEClientTransport(url);

    transport.onmessage = (message) => {
        const line = JSON.stringify(message);
        process.stdout.write(line + "\n");
    };

    transport.onerror = (err) => {
        console.error("[Zvec Proxy] SSE error:", err.message);
        process.exit(1);
    };

    transport.onclose = () => {
        process.exit(0);
    };

    await transport.start();

    // Relay stdin JSON-RPC → daemon
    const rl = (await import("readline")).createInterface({ input: process.stdin });
    for await (const line of rl) {
        if (!line.trim()) continue;
        try {
            const message = JSON.parse(line);
            await transport.send(message);
        } catch (err) {
            console.error("[Zvec Proxy] Error forwarding message:", err.message);
        }
    }

    await transport.close();
    process.exit(0);
}

// ─── MAIN ───────────────────────────────────────────────────────────────────

async function spawnDaemon(port) {
    return new Promise((resolve, reject) => {
        const child = spawn(process.execPath, [process.argv[1], "--daemon", "--port", String(port), ...process.argv.slice(2).filter(a => a !== "--daemon" && a !== "--port" && !/^\d+$/.test(a) && a !== String(port))], {
            stdio: "ignore",
            detached: true
        });
        child.unref();

        // Wait for lock file to appear
        let attempts = 0;
        const interval = setInterval(() => {
            attempts++;
            const lock = readDaemonLock();
            if (lock && lock.port === port) {
                clearInterval(interval);
                console.error(`[Zvec Bridge] Daemon started (PID ${lock.pid})`);
                resolve(lock);
            }
            if (attempts > 20) {
                clearInterval(interval);
                reject(new Error("Daemon start timeout"));
            }
        }, 200);
    });
}

function findFreePort() {
    return new Promise((resolve) => {
        const srv = http.createServer();
        srv.listen(0, "127.0.0.1", () => {
            const port = srv.address().port;
            srv.close(() => resolve(port));
        });
        srv.on("error", () => resolve(3100)); // fallback
    });
}

async function main() {
    registerProcessHandlers();
    console.error(`[Zvec Bridge] Start. DB: ${DB_FILE}`);
    console.error(`[Zvec Bridge] Extensions: ${ALLOWED_EXTENSIONS.join(", ")}`);
    console.error(`[Zvec Bridge] Model: ${EMBEDDING_MODEL}`);

    const args = process.argv.slice(2);
    const daemonIdx = args.indexOf("--daemon");
    const isDaemonRequested = daemonIdx !== -1;

    if (isDaemonRequested) {
        const portIdx = args.indexOf("--port");
        const port = portIdx !== -1 ? parseInt(args[portIdx + 1], 10) : 3100;
        if (isNaN(port) || port < 1 || port > 65535) {
            console.error("[Zvec Bridge] Invalid port. Usage: --daemon --port <number>");
            process.exit(1);
        }
        await startDaemon(port);
        return;
    }

    // Auto-detect or start daemon
    let daemonInfo = readDaemonLock();
    if (daemonInfo) {
        try {
            process.kill(daemonInfo.pid, 0);
            console.error(`[Zvec Bridge] Daemon found on port ${daemonInfo.port} (PID ${daemonInfo.pid}). Connecting via proxy.`);
            await startProxy(daemonInfo.port);
            return;
        } catch {
            console.error(`[Zvec Bridge] Stale daemon lock (PID ${daemonInfo.pid}). Starting new daemon.`);
            removeDaemonLock();
        }
    }

    // No daemon — spawn one automatically on a free port
    const port = await findFreePort();
    console.error(`[Zvec Bridge] No daemon running. Starting one on port ${port}...`);
    try {
        daemonInfo = await spawnDaemon(port);
        await startProxy(daemonInfo.port);
    } catch (err) {
        console.error(`[Zvec Bridge] ${err.message}. Falling back to legacy stdio mode.`);
        const transport = new StdioServerTransport();
        serverTransport = transport;
        await server.connect(transport);
        await ensureKnowledgeReady({ waitForCompletion: false });
        startWatcher();
    }
}

if (process.env.ZVEC_MCP_SKIP_MAIN !== "1") {
    main().catch((err) => {
        console.error("[Zvec Bridge] Critical error:", err);
        process.exit(1);
    });
}

export { ensureCollection, isSupportedFile, shouldIgnorePath, ensureKnowledgeReady, initializeKnowledgeBase, rankSearchResults, runInitializationOnce, startWatcher };