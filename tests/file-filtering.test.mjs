import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

process.env.ZVEC_MCP_SKIP_MAIN = '1';

const bridgeUrl = pathToFileURL(path.resolve('zvec-mcp-bridge.js')).href + '?t=' + Date.now();
const { shouldIgnorePath } = await import(bridgeUrl);

test('ignores files whose extensions are not in the indexable allowlist', () => {
  assert.equal(shouldIgnorePath('/project/src'), false);
  assert.equal(shouldIgnorePath('/project/src/app.tsx'), false);
  assert.equal(shouldIgnorePath('/project/LICENSE'), false);
  assert.equal(shouldIgnorePath('/project/src/notes.md'), true);
  assert.equal(shouldIgnorePath('/project/src/data.json'), true);
  assert.equal(shouldIgnorePath('/project/src/knowledge.db'), true);
  assert.equal(shouldIgnorePath('/project/.zvec/knowledge.db/chunk.0'), true);
});

test('does not treat extensionless files as supported source files', async () => {
  const bridgeUrl = pathToFileURL(path.resolve('zvec-mcp-bridge.js')).href + '?t=' + Date.now();
  const { isSupportedFile } = await import(bridgeUrl);

  assert.equal(isSupportedFile('/project/LICENSE'), false);
  assert.equal(isSupportedFile('/project/src/script.ts'), true);
});
