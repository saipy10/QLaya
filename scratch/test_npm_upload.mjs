import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import cp from 'node:child_process';

const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'test-qlaya-npm-'));
console.log('Testing in:', tempDir);

try {
  cp.execSync('npm init -y', { cwd: tempDir, stdio: 'ignore' });
  const pkgPath = path.join(tempDir, 'package.json');
  const pkg = JSON.parse(fs.readFileSync(pkgPath, 'utf8'));
  pkg.type = 'module';
  fs.writeFileSync(pkgPath, JSON.stringify(pkg, null, 2));

  console.log('Installing qlaya from public npm registry...');
  cp.execSync('npm install qlaya@latest', { cwd: tempDir, stdio: 'inherit' });

  const testScript = `
import { VERSION, QLAYA_MODEL_IDS, QLAYA_MODELS, normaliseName, Agent } from 'qlaya';

console.log('✓ Successfully imported qlaya from public npm registry!');
console.log('✓ Published Version:', VERSION);
console.log('✓ Models Count:', QLAYA_MODEL_IDS.length);
console.log('✓ First Model ID:', QLAYA_MODEL_IDS[0]);
console.log('✓ Model Spec Name:', QLAYA_MODELS[QLAYA_MODEL_IDS[0]].name);
console.log('✓ Normalised Router Name (ml):', normaliseName('ml'));
console.log('✓ Agent Class Type:', typeof Agent);
`;

  fs.writeFileSync(path.join(tempDir, 'test.mjs'), testScript);
  console.log('\nRunning test consumer script:');
  const out = cp.execSync('node test.mjs', { cwd: tempDir, encoding: 'utf8' });
  console.log(out);
  console.log('=== TEST RESULT: PASSED ===');
} finally {
  fs.rmSync(tempDir, { recursive: true, force: true });
  console.log('Cleaned up temp directory.');
}
