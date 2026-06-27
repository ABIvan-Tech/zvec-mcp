import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

process.env.ZVEC_MCP_SKIP_MAIN = '1';

test('importing the bridge helpers does not create a knowledge database', async () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'zvec-mcp-import-'));
  const dbPath = path.join(tempRoot, '.zvec', 'knowledge.db');

  process.env.PROJECT_ROOT = tempRoot;

  const bridgeUrl = pathToFileURL(path.resolve('zvec-mcp-bridge.js')).href + '?t=' + Date.now();
  const mod = await import(bridgeUrl);

  assert.equal(typeof mod.ensureKnowledgeReady, 'function');
  assert.equal(fs.existsSync(dbPath), false);
});