import React from "react";
import { DEFAULT_CONF_THRESHOLD, DEFAULT_CMI_THRESHOLD } from "../services/pipelineService";

export function Sidebar({
  confThreshold,
  setConfThreshold,
  cmiThreshold,
  setCmiThreshold,
  onResetControls,
}) {
  return (
    <aside className="sidebar" aria-label="Routing Controls & Empirical Benchmarks">
      {/* Routing Controls Card */}
      <div className="card">
        <h2 className="card-title">
          <span>⚙️</span> Routing & Triage Controls
        </h2>

        <div className="control-group">
          <div className="control-header">
            <label htmlFor="conf-slider">Confidence Threshold (Triage)</label>
            <span className="control-value">{(confThreshold * 100).toFixed(0)}%</span>
          </div>
          <input
            id="conf-slider"
            type="range"
            min="0.20"
            max="0.90"
            step="0.05"
            value={confThreshold}
            onChange={(e) => setConfThreshold(parseFloat(e.target.value))}
            className="slider-input"
            aria-label="Confidence threshold"
          />
          <div className="control-hint">
            Predictions below this level are escalated to <em>"Human agent (low confidence)"</em>.
          </div>
        </div>

        <div className="control-group">
          <div className="control-header">
            <label htmlFor="cmi-slider">Bilingual Queue CMI Threshold</label>
            <span className="control-value">{cmiThreshold.toFixed(0)}%</span>
          </div>
          <input
            id="cmi-slider"
            type="range"
            min="10.0"
            max="60.0"
            step="5.0"
            value={cmiThreshold}
            onChange={(e) => setCmiThreshold(parseFloat(e.target.value))}
            className="slider-input"
            aria-label="Code-Mixing Index threshold"
          />
          <div className="control-hint">
            Queries with CMI &ge; threshold are flagged for bilingual tier routing (Empirical 75th percentile = 30.0%).
          </div>
        </div>

        <button
          type="button"
          onClick={onResetControls}
          className="btn-secondary"
          style={{ width: "100%", marginTop: "12px", textAlign: "center" }}
        >
          Reset to Empirical Defaults
        </button>
      </div>

      {/* Empirical Highlights Card */}
      <div className="card">
        <h2 className="card-title">
          <span>📊</span> Empirical Benchmarks (15 Folds)
        </h2>
        <div className="stat-list">
          <div className="stat-item">
            <span>LID Test Accuracy (LinCE)</span>
            <span style={{ color: "var(--success)" }}>98.42% (F1 0.9805)</span>
          </div>
          <div className="stat-item">
            <span>Homograph Disambiguation</span>
            <span style={{ color: "var(--success)" }}>+10.79% (95% CI)</span>
          </div>
          <div className="stat-item">
            <span>Annotator Agreement (Kappa)</span>
            <span>&kappa; = 0.9399</span>
          </div>
          <div className="stat-item">
            <span>Setup B (Subwords, No LID)</span>
            <span style={{ color: "var(--success)" }}>0.9308 &plusmn; 0.0272</span>
          </div>
          <div className="stat-item">
            <span>Setup C (LID-Aware Normalization)</span>
            <span>0.9185 &plusmn; 0.0258</span>
          </div>
          <div className="stat-item">
            <span>Setup A (English Baseline)</span>
            <span style={{ color: "var(--danger)" }}>0.8975 &plusmn; 0.0280</span>
          </div>
          <div className="stat-item" style={{ background: "var(--hi-bg)", color: "var(--hi-color)", fontWeight: 700 }}>
            <span>Refund Intent Win (C vs B)</span>
            <span>0.9606 vs 0.9555</span>
          </div>
          <div className="stat-item" style={{ background: "var(--hi-bg)", color: "var(--hi-color)", fontWeight: 700 }}>
            <span>Cancel Order Win (C vs B)</span>
            <span>0.9723 vs 0.9697</span>
          </div>
        </div>
      </div>

      {/* Pipeline Specifications Card */}
      <div className="card" style={{ fontSize: "0.82rem", color: "var(--text-muted)" }}>
        <h2 className="card-title" style={{ fontSize: "0.88rem" }}>
          <span>🛡️</span> Architecture Safeguards
        </h2>
        <ul style={{ paddingLeft: "16px", display: "flex", flexDirection: "column", gap: "6px" }}>
          <li><strong>Zero Data Leakage:</strong> Vectorizers fit strictly inside CV folds.</li>
          <li><strong>Zero Circular Dependency:</strong> LID tagger never trained on intent data.</li>
          <li><strong>Sentence Partitioning:</strong> 80/10/10 split prevented intra-sentential token leaks.</li>
          <li><strong>Deterministic:</strong> Seed=42 locked across all 15 folds &amp; 5,000 bootstrap resamples.</li>
        </ul>
      </div>
    </aside>
  );
}
