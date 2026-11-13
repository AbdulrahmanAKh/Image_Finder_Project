import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { rank } from '../docs/search.js';
const root = new URL('../', import.meta.url);
const file = p => new URL(p, root);
const m = JSON.parse(fs.readFileSync(file('docs/data/manifest.json')));
const bytes = fs.readFileSync(file('docs/data/embeddings.f32'));
const vectors = new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
assert.equal(m.count, 2000); assert.equal(m.dimensions, 512);
assert.equal(vectors.length, m.count * m.dimensions);
assert.equal(new Set(m.items.map(item => item.id)).size, m.count);
const categories = new Map();
for (let i = 0; i < m.count; i++) {
  const item = m.items[i];
  assert(fs.existsSync(file('docs/data/' + item.image)));
  categories.set(item.category, (categories.get(item.category) || 0) + 1);
  const v = vectors.slice(i * 512, (i + 1) * 512);
  assert(v.every(Number.isFinite)); assert(Math.abs(Math.hypot(...v) - 1) < 1e-5);
}
assert.equal(categories.size, 200); assert([...categories.values()].every(n => n === 10));
for (const sample of m.samples) {
  assert(!m.items.some(item => item.id === sample.id));
  assert(fs.existsSync(file('docs/data/' + sample.image)));
  const matches = rank(sample.vector, vectors, m.count, 512, 6);
  assert.equal(matches.length, 6);
  assert(matches.every((item, index) => index === 0 || item.score <= matches[index - 1].score));
}
const q = vectors.slice(0, 512);
assert.equal(rank(q, vectors, m.count, 512, 1)[0].index, 0);
assert(!rank(q, vectors, m.count, 512, 12, 0).some(x => x.index === 0));
assert.throws(() => rank([0], vectors, m.count, 512, 1));
assert.throws(() => rank(new Array(512).fill(0), vectors, m.count, 512, 1));
const nb = JSON.parse(fs.readFileSync(file('Image_Finder_CLIP.ipynb')));
assert(nb.cells.filter(c => c.cell_type === 'code').every(c => c.execution_count !== null));
assert(!nb.cells.some(c => c.outputs?.some(o => o.output_type === 'error')));
assert(nb.cells.some(c => c.outputs?.some(o => o.data?.['image/png'])));
for (const name of ['app.js','search.js','model-worker.js','style.css']) assert(fs.existsSync(file('docs/'+name)));
console.log('Verified 2,000 images, 200 balanced categories, normalized finite embeddings, held-out samples, sorted cosine rankings, self-exclusion, and executed notebook figures.');
console.log(m.evaluation);
