import type { Metadata } from "next";

export const metadata: Metadata = { title: "Model Visualizations · echo" };

// The images are served by FastAPI from backend/static/
export default function Visualizations() {
  return (
    <div className="panel">
      <h3>Model Visualizations (1,656 Commercial Songs)</h3>
      <p className="panel-lead">The 512-dimensional acoustic universe discovered by CnnMusicEncoder projected into 2D.</p>
      <div className="viz-grid">
        <div>
          <h4 style={{ color: "var(--teal)" }}>2D t-SNE Embedding Cluster Map</h4>
          <img src="/static/tsne.png" alt="t-SNE Embeddings" />
        </div>
        <div>
          <h4 style={{ color: "var(--accent)" }}>Cross-Genre Acoustic Similarity Heatmap</h4>
          <img src="/static/genre_similarity.png" alt="Genre Similarity" />
        </div>
        <div>
          <h4 style={{ color: "#e5e7eb" }}>Training &amp; Validation Loss Convergence</h4>
          <img src="/static/training_curves.png" alt="Training Curves" />
        </div>
      </div>
    </div>
  );
}
