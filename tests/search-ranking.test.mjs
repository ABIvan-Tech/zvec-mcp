import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

process.env.ZVEC_MCP_SKIP_MAIN = '1';

const bridgeUrl = pathToFileURL(path.resolve('zvec-mcp-bridge.js')).href + '?t=' + Date.now();
const { rankSearchResults } = await import(bridgeUrl);

test('prefers specific procedural-generation files over generic app files', () => {
  const genericAppResult = {
    fields: {
      file_path: '/project/src/App.tsx',
      text_content: 'This file initializes the app, renders the main screen, and contains broad startup logic for the user interface.'
    }
  };

  const proceduralResult = {
    fields: {
      file_path: '/project/src/proceduralGeneration.ts',
      text_content: 'This module generates procedural planets and seeds for the game world based on the requested generation rules.'
    }
  };

  const ranked = rankSearchResults('procedural generation seed', [genericAppResult, proceduralResult]);

  assert.equal(ranked[0].fields.file_path.endsWith('proceduralGeneration.ts'), true);
  assert.ok(ranked[0].fields.file_path.includes('procedural'));
});
