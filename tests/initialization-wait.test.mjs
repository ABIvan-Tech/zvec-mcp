import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

process.env.ZVEC_MCP_SKIP_MAIN = '1';

test('waits for a single initialization pass before continuing', async () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'zvec-mcp-init-'));
  process.env.PROJECT_ROOT = tempRoot;

  const bridgeUrl = pathToFileURL(path.resolve('zvec-mcp-bridge.js')).href + '?t=' + Date.now();
  const { runInitializationOnce } = await import(bridgeUrl);

  let started = 0;
  let finished = 0;

  const initializer = async () => {
    started += 1;
    await new Promise((resolve) => setTimeout(resolve, 30));
    finished += 1;
  };

  await Promise.all([runInitializationOnce(initializer), runInitializationOnce(initializer)]);

  assert.equal(started, 1);
  assert.equal(finished, 1);
});
