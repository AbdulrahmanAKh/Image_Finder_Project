# Image Finder

An interactive visual atlas: upload an image and retrieve related images from a **2,000-image Tiny ImageNet gallery**, using **CLIP ViT-B/32 and cosine similarity**.

[![Live demo](https://img.shields.io/badge/Explore-Live_demo-174bc5?style=for-the-badge)](https://AbdulrahmanAKh.github.io/Image_Finder_Project/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbdulrahmanAKh/Image_Finder_Project/blob/main/Image_Finder_CLIP.ipynb)
[![GitHub](https://img.shields.io/badge/GitHub-Source-102b69?style=for-the-badge&logo=github)](https://github.com/AbdulrahmanAKh/Image_Finder_Project)

## Try it

Choose one of six held-out sample images for an instant search, or upload a JPG, PNG, or WebP image up to 10 MB. Uploaded images are encoded on your device in a Web Worker. The first upload downloads the ~89 MB model from Hugging Face; the model and browser runtime need an internet connection and enough device memory. Subsequent searches reuse the browser cache where available. Sample searches use cached CLIP vectors and do not download the model. Select **Run CLIP on this image** to process a sample through the same file validation, decoding, and inference pipeline as an upload.

Results are existing dataset images, not generated images or web search results. Click a result to inspect its source row and score, or use it as a new query (its own gallery entry is excluded). Choose 6, 12, or 24 results. The processing animation follows actual model loading and inference, with no simulated detection delay. Keyboard focus, accessible controls, mobile layouts, and reduced-motion preferences are supported.

## Notebook

Open [Image_Finder_CLIP.ipynb](Image_Finder_CLIP.ipynb) in Colab and run the cells in order. It downloads the validation parquet (~15 MB), builds a deterministic gallery, extracts normalized 512-dimensional embeddings, displays query/result strips, evaluates retrieval, accepts uploaded images, and exports all website data. Real execution outputs and figures are retained in the committed notebook.

The implementation uses an ONNX quantized CLIP vision encoder and the official CLIP image processor. The website uses the same pinned model revision through Transformers.js 3.8.1 with CPU/WASM inference, which works without GitHub Pages cross-origin isolation headers. Each image is encoded independently in Python and in the browser: dynamic q8 quantization can otherwise change embeddings with batch composition. Minor Python/JavaScript resizing or runtime differences may still affect near-tied scores. Sample retrieval uses the same exported embeddings on both platforms.

### Improvements over the supplied notebook

| Original behavior | Revised behavior |
| --- | --- |
| Resize and tensor conversion without pretrained normalization | Official CLIP resize, crop, RGB conversion, and normalization |
| Flattened EfficientNet spatial feature maps | Compact pretrained CLIP image projection (512 dimensions) |
| PCA to 300 dimensions without evaluation | No PCA; direct normalized cosine search |
| Unseeded random gallery | Seeded balanced gallery across 200 classes |
| Query removed by positional slicing | Explicit self-exclusion for gallery queries |
| One random query, no metrics | Held-out queries, labeled results, Precision@5 and Hit@5 |
| Cosine scores displayed as percentages | Raw cosine scores with a clear non-probability explanation |
| Notebook-only experience | Executed notebook plus interactive static browser demo |

CLIP is selected for its semantic image representation and practical browser support. This project does **not** establish superiority over a correctly implemented EfficientNetV2 baseline. The measured comparison is against a simple normalized 8 × 8 RGB descriptor on identical gallery/query partitions. See the notebook and the demo for the actual measured values.

## Dataset and evaluation

- Dataset: [Tiny ImageNet, hosted by zh-plus](https://huggingface.co/datasets/zh-plus/tiny-imagenet), revision `5a77092c28e51558c5586e9c5eb71a7e17a5e43f`.
- Gallery: 10 images per class, **2,000 total**, from the validation split.
- Evaluation queries: one additional image per class, **200 total**, disjoint from the gallery, using seed 42.
- Duplicate RGB image content is removed before sampling.
- Precision@5: fraction of the top five sharing the query class. Hit@5: fraction of queries with at least one same-class result in the top five.
- Class labels approximate semantic relevance, not exact object identity. This split is an educational experiment, not an official retrieval benchmark. No fine-tuning or hyperparameter selection occurs on the queries. Pretraining overlap cannot be excluded.
- The six ready-made samples are held-out queries, not gallery members. They are illustrative examples; the metrics cover all 200 queries.

Tiny ImageNet broadens the original 101-category collection to 200 categories. Its **64 × 64 resolution** limits fine-detail search; a higher-resolution, appropriately licensed dataset is recommended for production. Arbitrary uploads always have nearest neighbors, even when no relevant match exists. CLIP may reflect biases in its pretraining data.

## Local use

The exported website needs no build step, backend, API key, or npm dependencies:

```bash
python -m http.server 8000 --directory docs
```

Open `http://localhost:8000`. Do not open `index.html` using `file://`, which restricts module loading and workers.

To regenerate the gallery, install `requirements.txt` and run every notebook cell. Commit the resulting notebook, `docs/data/manifest.json`, `docs/data/embeddings.f32`, and `docs/data/images/` together. Changing the encoder requires rebuilding the entire index. Model and dataset revision identifiers and SHA-256 checksums are stored in the manifest.

```bash
python -m pip install -r requirements.txt
npm run check
```

The verification script checks finite normalized embeddings, all local image assets, balanced classes, disjoint samples, sorted cosine rankings, explicit self-exclusion, invalid query handling, and real notebook figures. It runs automatically before Pages deployment.

## GitHub Pages

In repository **Settings → Pages**, choose **GitHub Actions** as the source. The included `.github/workflows/pages.yml` verifies the project and publishes `docs/` on pushes to `main`. Static paths are relative, so the project works under `/Image_Finder_Project/`. The model is fetched from Hugging Face instead of committing a large binary to GitHub. GitHub and Colab action buttons point to this repository and notebook.

## Project files

```text
Image_Finder_CLIP.ipynb     # Simple end-to-end notebook with outputs
requirements.txt          # Python notebook dependencies
docs/index.html           # Accessible, responsive interface
docs/style.css            # Visual atlas theme and processing animation
docs/app.js               # Uploads, samples, ranking, result inspector
docs/model-worker.js      # CLIP inference away from the UI thread
docs/search.js            # Exact cosine ranking
docs/data/                # Gallery, embeddings, samples, evaluation metadata
scripts/check.mjs         # Integrity and retrieval verification
.github/workflows/pages.yml
```

## Design references and licenses

The interface follows a visual atlas concept: navy `#102b69`, cobalt `#174bc5`, ice `#edf3fd`, paper `#f8faff`, aqua `#7ee1db`, and rose `#ffbcb2`; Manrope headings and DM Sans body text. A query panel faces a ranked contact sheet, with restrained surrounding content. Animation responds to inference work and respects reduced motion.

Design guidance consulted: [Anthropic frontend-design skill](https://github.com/anthropics/skills/blob/main/skills/frontend-design/SKILL.md). No external skill code is bundled. Fonts load from Google Fonts; inference code loads from jsDelivr; model assets load from Hugging Face. Uploaded image contents are not sent to those services.

Original project code is MIT licensed. **The code license does not cover dataset images, model weights, fonts, or third-party libraries.** See [THIRD_PARTY.md](THIRD_PARTY.md). Dataset images remain subject to [ImageNet noncommercial educational/research terms and original owners’ rights](https://huggingface.co/datasets/zh-plus/tiny-imagenet/blob/main/README.md).

## Author

- GitHub: [@AbdulrahmanAKh](https://github.com/AbdulrahmanAKh)
- LinkedIn: [Abdulrahman Alkhatib](https://www.linkedin.com/in/abdulrahman-alkhatib/)
- Email: [aboodkhatib90@gmail.com](mailto:aboodkhatib90@gmail.com)
- Portfolio: [abdulrahmanakh.github.io](https://abdulrahmanakh.github.io/)

---

<p align="center">
  <b>Developed &amp; Engineered by Abdulrahman Alkhatib</b><br>
  <i>Computer Science Graduate | Software &amp; AI Engineer</i>
</p>
