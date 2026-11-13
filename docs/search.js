// Pure cosine retrieval shared with the Node verification script.
export function rank(vector, embeddings, count, dimensions, limit, excludeIndex = -1) {
  if (vector.length !== dimensions || embeddings.length !== count * dimensions) throw new Error('Index dimensions do not match the query.');
  const norm = Math.hypot(...vector);
  if (!Number.isFinite(norm) || norm < 1e-12) throw new Error('The query embedding is invalid.');
  const scores = [];
  for (let i = 0; i < count; i++) {
    if (i === excludeIndex) continue;
    let score = 0;
    for (let j = 0; j < dimensions; j++) score += vector[j] / norm * embeddings[i * dimensions + j];
    scores.push({ index: i, score: Math.max(-1, Math.min(1, score)) });
  }
  scores.sort((a, b) => b.score - a.score || a.index - b.index);
  return scores.slice(0, Math.min(limit, scores.length));
}
