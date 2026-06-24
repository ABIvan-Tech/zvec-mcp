import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

process.env.ZVEC_MCP_SKIP_MAIN = '1';

test('reports ready once the collection can be opened even while indexing continues', async () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'zvec-mcp-collection-'));
  process.env.PROJECT_ROOT = tempRoot;
  fs.mkdirSync(path.join(tempRoot, '.zvec', 'knowledge.db'), { recursive: true });

  const bridgeUrl = pathToFileURL(path.resolve('zvec-mcp-bridge.js')).href + '?t=' + Date.now();
  const { ensureKnowledgeReady } = await import(bridgeUrl);

  const result = await ensureKnowledgeReady({ waitForCompletion: false });

  assert.equal(result.ready, true);
  assert.equal(result.status, 'ready');
});
