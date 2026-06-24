import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

test('auto-recovers from a broken knowledge db directory', async () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'zvec-mcp-recovery-'));
  const projectRoot = path.join(tempRoot, 'project');
  const dbPath = path.join(projectRoot, '.zvec', 'knowledge.db');

  fs.mkdirSync(path.dirname(dbPath), { recursive: true });
  fs.writeFileSync(dbPath, 'not-a-real-zvec-db');

  process.env.PROJECT_ROOT = projectRoot;
  process.env.ZVEC_MCP_SKIP_MAIN = '1';

  const bridgeUrl = pathToFileURL(path.resolve('zvec-mcp-bridge.js')).href + '?t=' + Date.now();
  const mod = await import(bridgeUrl);
  const collection = mod.ensureCollection();

  assert.ok(collection, 'collection should be created after recovery');
  assert.ok(fs.existsSync(dbPath), 'db path should exist after recovery');
});
