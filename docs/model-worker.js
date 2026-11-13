// Model work stays off the UI thread so progress and animation stay responsive.
import { AutoProcessor, CLIPVisionModelWithProjection, RawImage, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/dist/transformers.min.js';
env.allowLocalModels = false;
env.backends.onnx.wasm.numThreads = 1; // GitHub Pages does not supply cross-origin isolation headers.
let encoder;
async function loadEncoder() {
  if (!encoder) encoder = Promise.all([
    AutoProcessor.from_pretrained('Xenova/clip-vit-base-patch32', { revision: '6b7c523703ae26e39be4b03774966f8571c7f523' }),
    CLIPVisionModelWithProjection.from_pretrained('Xenova/clip-vit-base-patch32', {
      dtype: 'q8', device: 'wasm', revision: '6b7c523703ae26e39be4b03774966f8571c7f523',
      progress_callback: p => {
        if (p.status === 'progress' && p.file?.includes('vision_model')) self.postMessage({ type: 'progress', message: `Downloading image model… ${Math.round(p.progress)}%` });
      },
    }),
  ]).catch(error => { encoder = undefined; throw error; });
  return encoder;
}
self.onmessage = async ({ data }) => {
  try {
    self.postMessage({ type: 'progress', message: 'Loading the image model. First upload downloads ~89 MB; later searches reuse the cache.' });
    const [processor, model] = await loadEncoder();
    self.postMessage({ type: 'progress', message: 'Encoding your image with CLIP…' });
    const image = await RawImage.read(data.url);
    const inputs = await processor(image);
    const { image_embeds } = await model(inputs);
    const vector = Array.from(image_embeds.data);
    const norm = Math.hypot(...vector);
    self.postMessage({ type: 'result', vector: vector.map(x => x / Math.max(norm, 1e-12)) });
  } catch (error) {
    self.postMessage({ type: 'error', message: error.message || String(error) });
  }
};
