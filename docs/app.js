import { rank } from './search.js';
const $ = id => document.getElementById(id);
let manifest, embeddings, selection, currentVector, worker, busy = false, uploadURL, detailIndex;
const asset = path => new URL(`data/${path}`, document.baseURI).href;
function status(message, error = false) { $('status').textContent = message; $('status').classList.toggle('error', error); }
function setBusy(value) {
  busy = value;
  $('search').disabled = value || !selection;
  $('upload').disabled = value;
  $('clear').disabled = value;
  $('encode-live').disabled = value;
  $('model-stage').hidden = !value;
  $('dropzone').classList.toggle('busy', value);
  $('search').textContent = value ? 'Finding visual neighbors…' : 'Find similar images ⌕';
  document.querySelectorAll('.sample').forEach(button => button.disabled = value);
}
function choose(next) {
  if (busy) return;
  if (uploadURL && next.url !== uploadURL) { URL.revokeObjectURL(uploadURL); uploadURL = undefined; }
  selection = next;
  currentVector = undefined;
  $('query-image').src = next.url;
  $('query-image').hidden = false;
  $('upload-prompt').hidden = true;
  $('query-caption').textContent = next.caption;
  $('clear').hidden = false;
  $('search').disabled = false;
  $('encode-live').hidden = !next.vector;
  document.querySelectorAll('.sample').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.id === String(next.id))));
  status('Ready to search the Tiny ImageNet gallery.');
}
function render() {
  if (!currentVector) return;
  const start = performance.now();
  const matches = rank(currentVector, embeddings, manifest.count, manifest.dimensions, Number($('limit').value), selection.excludeIndex);
  $('results').replaceChildren();
  for (const [position, match] of matches.entries()) {
    const item = manifest.items[match.index];
    const button = document.createElement('button');
    button.className = 'result';
    button.setAttribute('aria-label', `Rank ${position + 1}, ${item.label}, cosine ${match.score.toFixed(3)}. View image.`);
    const figure = document.createElement('figure');
    const image = document.createElement('img'); image.src = asset(item.image); image.alt = item.label; image.loading = 'lazy';
    const badge = document.createElement('span'); badge.className = 'rank'; badge.textContent = `#${position + 1}`;
    figure.append(image, badge);
    const caption = document.createElement('div'); caption.className = 'result-name';
    const name = document.createElement('strong'); name.textContent = item.label;
    const score = document.createElement('span'); score.textContent = match.score.toFixed(3);
    caption.append(name, score);
    const track = document.createElement('div'); track.className = 'score-track'; track.setAttribute('aria-hidden', 'true');
    const bar = document.createElement('i'); bar.style.width = `${Math.max(0, match.score) * 100}%`; track.append(bar);
    button.append(figure, caption, track);
    button.addEventListener('click', () => {
      detailIndex = match.index;
      $('detail-image').src = asset(item.image); $('detail-image').alt = item.label;
      $('detail-label').textContent = item.label;
      $('detail-score').textContent = `Rank #${position + 1} · Cosine similarity ${match.score.toFixed(4)} · Dataset row ${item.id}`;
      $('detail').showModal();
    });
    $('results').append(button);
  }
  $('empty').hidden = true;
  $('score-note').hidden = false;
  $('result-summary').textContent = `${matches.length} nearest neighbors from ${manifest.count.toLocaleString()} dataset images`;
  $('time').textContent = `Search ${(performance.now() - start).toFixed(0)} ms`;
  status(selection.vector ? 'Results ready · Using a cached CLIP embedding.' : 'Results ready · This image was encoded locally with CLIP.');
}
async function encodeUpload(url) {
  if (!worker) worker = new Worker(new URL('./model-worker.js', import.meta.url), { type: 'module' });
  return new Promise((resolve, reject) => {
    worker.onmessage = ({ data }) => {
      if (data.type === 'progress') status(data.message);
      if (data.type === 'result') { cleanup(); resolve(data.vector); }
      if (data.type === 'error') { cleanup(); reject(new Error(data.message)); }
    };
    worker.onerror = event => { event.preventDefault(); cleanup(); worker.terminate(); worker = undefined; reject(new Error('The image model could not start. Check your connection and allow downloads from Hugging Face and jsDelivr.')); };
    const timeout = setTimeout(() => { cleanup(); worker.terminate(); worker = undefined; reject(new Error('Model loading timed out. Check your connection and try again, or use a sample.')); }, 300000);
    function cleanup() { clearTimeout(timeout); if (worker) { worker.onmessage = null; worker.onerror = null; } }
    worker.postMessage({ url });
  });
}
$('search').addEventListener('click', async () => {
  if (!selection || busy) return;
  setBusy(true);
  try {
    if (selection.vector) currentVector = selection.vector;
    else currentVector = await encodeUpload(selection.url);
    render();
  } catch (error) { status(`Search failed: ${error.message} You can retry or choose a ready-made sample.`, true); }
  finally { setBusy(false); }
});
$('encode-live').addEventListener('click', () => {
  if (!selection || busy) return;
  selection = { ...selection, vector: undefined };
  $('encode-live').hidden = true;
  $('search').click();
});
async function acceptFile(file) {
  if (busy || !manifest || !file) return;
  if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) { status('Choose a JPG, PNG, or WebP image.', true); return; }
  if (file.size > 10 * 1024 * 1024) { status('This image is larger than 10 MB. Choose a smaller image.', true); return; }
  const url = URL.createObjectURL(file);
  try {
    const image = new Image(); image.src = url; await image.decode();
    if (image.naturalWidth * image.naturalHeight > 40000000) throw new Error('Choose an image smaller than 40 megapixels.');
    choose({ url, caption: file.name, excludeIndex: -1 }); uploadURL = url;
  } catch (error) { URL.revokeObjectURL(url); status(`Could not read this image. ${error.message}`, true); }
}
$('upload').addEventListener('change', event => acceptFile(event.target.files[0]));
['dragenter','dragover'].forEach(type => $('dropzone').addEventListener(type, event => { event.preventDefault(); if (!busy) $('dropzone').classList.add('drag'); }));
['dragleave','drop'].forEach(type => $('dropzone').addEventListener(type, event => { event.preventDefault(); $('dropzone').classList.remove('drag'); if (type === 'drop') acceptFile(event.dataTransfer.files[0]); }));
$('limit').addEventListener('change', render);
$('clear').addEventListener('click', () => {
  if (busy) return;
  selection = currentVector = undefined;
  if (uploadURL) { URL.revokeObjectURL(uploadURL); uploadURL = undefined; }
  $('upload').value = ''; $('query-image').hidden = true; $('query-image').removeAttribute('src'); $('upload-prompt').hidden = false;
  $('query-caption').textContent = 'Start with your photo or a sample below.';
  $('clear').hidden = true; $('search').disabled = true; $('results').replaceChildren(); $('empty').hidden = false;
  $('encode-live').hidden = true;
  $('score-note').hidden = true; $('time').textContent = ''; $('result-summary').textContent = 'Choose an image to begin exploring.';
  document.querySelectorAll('.sample').forEach(button => button.setAttribute('aria-pressed', 'false'));
  status('Collection ready. Choose a sample or upload an image.');
});
$('close-detail').addEventListener('click', () => $('detail').close());
$('detail').addEventListener('click', event => { if (event.target === $('detail')) $('detail').close(); });
$('search-detail').addEventListener('click', () => {
  const item = manifest.items[detailIndex];
  choose({ id: item.id, url: asset(item.image), caption: `${item.label} · Gallery query; this image is excluded from results.`, vector: Array.from(embeddings.slice(detailIndex * 512, (detailIndex + 1) * 512)), excludeIndex: detailIndex });
  $('detail').close(); $('search').click();
});
async function init() {
  try {
    const responses = await Promise.all([fetch('data/manifest.json'), fetch('data/embeddings.f32')]);
    if (responses.some(response => !response.ok)) throw new Error('Gallery files are unavailable.');
    manifest = await responses[0].json(); embeddings = new Float32Array(await responses[1].arrayBuffer());
    if (embeddings.length !== manifest.count * manifest.dimensions) throw new Error('The gallery index is incomplete.');
    for (const sample of manifest.samples) {
      const button = document.createElement('button'); button.className = 'sample'; button.dataset.id = sample.id; button.type = 'button';
      button.title = `Search with ${sample.label}`; button.setAttribute('aria-label', `Use ${sample.label} sample`); button.setAttribute('aria-pressed', 'false');
      const img = document.createElement('img'); img.src = asset(sample.image); img.alt = sample.label; button.append(img);
      button.addEventListener('click', () => { choose({ ...sample, url: asset(sample.image), caption: `${sample.label} · Held-out dataset sample`, excludeIndex: -1 }); $('search').click(); });
      $('samples').append(button);
    }
    [0, 1, 3].forEach((index, position) => {
      const sample = manifest.samples[index]; $('hero-' + ['a','b','c'][position]).src = asset(sample.image);
      const image = document.createElement('img'); image.src = asset(sample.image); image.alt = ''; $('empty-tiles').append(image);
    });
    const table = document.createElement('table'); table.className = 'metric-table'; table.innerHTML = '<caption>Held-out class relevance</caption><thead><tr><th scope="col">Method</th><th scope="col">Precision@5</th><th scope="col">Hit@5</th></tr></thead>';
    const body = document.createElement('tbody');
    for (const [name, values] of Object.entries(manifest.evaluation)) {
      const row = document.createElement('tr');
      for (const value of [name, values['Precision@5'].toFixed(3), values['Hit@5'].toFixed(3)]) { const cell = document.createElement('td'); cell.textContent = value; row.append(cell); }
      body.append(row);
    }
    table.append(body); $('metrics').replaceChildren(table);
    $('gallery-count').textContent = `${manifest.count.toLocaleString()} images`;
    status('Collection ready. Choose a sample or upload an image.');
  } catch (error) { $('upload').disabled = true; status(`Collection could not load: ${error.message} Refresh to try again.`, true); }
}
init();
