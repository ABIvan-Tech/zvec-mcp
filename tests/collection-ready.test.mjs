import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import {
  ZVecCreateAndOpen,
  ZVecCollectionSchema,
  ZVecDataType,
  ZVecIndexType,
  ZVecMetricType
} from '@zvec/zvec';

process.env.ZVEC_MCP_SKIP_MAIN = '1';

test('reports ready once the collection can be opened even while indexing continues', async () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'zvec-mcp-collection-'));
  process.env.PROJECT_ROOT = tempRoot;

  const dbPath = path.join(tempRoot, '.zvec', 'knowledge.db');
  const schema = new ZVecCollectionSchema({
    name: 'project_code',
    fields: [
      { name: 'text_content', dataType: ZVecDataType.STRING },
      { name: 'file_path', dataType: ZVecDataType.STRING },
      { name: 'language', dataType: ZVecDataType.STRING }
    ],
    vectors: [
      {
        name: 'code_embedding',
        dataType: ZVecDataType.VECTOR_FP32,
        dimension: 384,
        indexParams: {
          indexType: ZVecIndexType.HNSW,
          metricType: ZVecMetricType.COSINE
        }
      }
    ]
  });

  const collection = ZVecCreateAndOpen(dbPath, schema);
  collection.insertSync({
    id: 'existing-doc',
    vectors: { code_embedding: new Array(384).fill(0) },
    fields: {
      text_content: 'existing indexed code',
      file_path: path.join(tempRoot, 'src', 'existing.js'),
      language: 'js'
    }
  });
  collection.closeSync();

  const bridgeUrl = pathToFileURL(path.resolve('zvec-mcp-bridge.js')).href + '?t=' + Date.now();
  const { ensureKnowledgeReady } = await import(bridgeUrl);

  const result = await ensureKnowledgeReady({ waitForCompletion: false });

  assert.equal(result.ready, true);
  assert.equal(result.status, 'ready');
});

test('does not report ready for a fresh empty database until the first scan finishes', async () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'zvec-mcp-fresh-collection-'));
  process.env.PROJECT_ROOT = tempRoot;

  const bridgeUrl = pathToFileURL(path.resolve('zvec-mcp-bridge.js')).href + '?t=' + Date.now();
  const { ensureKnowledgeReady } = await import(bridgeUrl);

  const initial = await ensureKnowledgeReady({ waitForCompletion: false });
  assert.equal(initial.ready, false);
  assert.equal(initial.status, 'initializing');

  const completed = await ensureKnowledgeReady({ waitForCompletion: true, timeoutMs: 1000 });
  assert.equal(completed.ready, true);
  assert.equal(completed.status, 'ready');
});
