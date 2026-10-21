import React, { useState } from "react";
import { TokenBadge } from "./TokenBadge";
import { AuditTable } from "./AuditTable";
import { IntentProbabilityBar } from "./IntentProbabilityBar";
import { RoutingCard } from "./RoutingCard";

export function PipelineStages({ analysis, onCopyText }) {
  const [showAuditTable, setShowAuditTable] = useState(false);

  if (!analysis) {
    return (
      <div className="stage-block" style={{ textAlign: "center", padding: "40px 20px" }}>
        <span style={{ fontSize: "2rem" }}>👆</span>
        <h3 style={{ margin: "10px 0 6px", fontWeight: 700 }}>No Analysis Yet</h3>
        <p style={{ color: "var(--text-muted)", fontSize: "0.9rem" }}>
          Type a support query above or click any benchmark sample to run the linguistic pipeline.
        </p>
      </div>
    );
  }

  const {
    cleaned_text,
    tokens,
    tags,
    tag_explanations,
    cmi,
    text_C,
    audit_rows,
    intent_results,
    routing,
  } = analysis;

  const hiCount = tags.filter((t) => t === "HI").length;
  const enCount = tags.filter((t) => t === "EN").length;
  const otherCount = tags.filter((t) => t === "OTHER").length;

  const topIntent = intent_results.Setup_C.intent;
  const topConfidence = intent_results.Setup_C.confidence;
  const probabilities = intent_results.Setup_C.probabilities;

  return (
    <div>
      {/* ------------------------------------------------------------------ */}
      {/* Stage 1: Preprocessing & Masking */}
      {/* ------------------------------------------------------------------ */}
      <section className="stage-block" aria-labelledby="stage-1-title">
        <div className="stage-header">
          <h2 id="stage-1-title" className="stage-title">
            <span className="stage-step-num">1</span> Tokenization &amp; Deterministic Masking
          </h2>
          <span style={{ fontSize: "0.82rem", color: "var(--text-muted)" }}>
            Entity regex masking &amp; character collapse
          </span>
        </div>

        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: "12px", flexWrap: "wrap", background: "var(--bg-surface-sunken)", padding: "10px 14px", borderRadius: "var(--radius-md)" }}>
          <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.9rem", wordBreak: "break-all" }}>
            <strong>Cleaned Text:</strong> {cleaned_text}
          </div>
          <div style={{ fontSize: "0.8rem", color: "var(--accent-primary)", fontWeight: 700 }}>
            {tokens.length} Total Tokens
          </div>
        </div>
      </section>

      {/* ------------------------------------------------------------------ */}
      {/* Stage 2: Word-Level LID Tagging */}
      {/* ------------------------------------------------------------------ */}
      <section className="stage-block" aria-labelledby="stage-2-title">
        <div className="stage-header">
          <h2 id="stage-2-title" className="stage-title">
            <span className="stage-step-num">2</span> Word-Level Language ID (Production M3 Context Model)
          </h2>
          <span style={{ fontSize: "0.82rem", color: "var(--text-muted)" }}>
            LinCE benchmark test accuracy: 98.42%
          </span>
        </div>

        {/* Tokens Container */}
        <div className="tokens-container">
          {tokens.map((tok, idx) => (
            <TokenBadge
              key={`${tok}-${idx}`}
              token={tok}
              tag={tags[idx]}
              explanation={tag_explanations[idx]}
              index={idx + 1}
            />
          ))}
        </div>

        {/* Metrics Row */}
        <div className="metrics-row">
          <div className="metric-card">
            <div className="metric-value" style={{ color: "var(--hi-color)" }}>{hiCount}</div>
            <div className="metric-label">Hindi (HI) Tokens</div>
          </div>
          <div className="metric-card">
            <div className="metric-value" style={{ color: "var(--en-color)" }}>{enCount}</div>
            <div className="metric-label">English (EN) Tokens</div>
          </div>
          <div className="metric-card">
            <div className="metric-value" style={{ color: "var(--other-color)" }}>{otherCount}</div>
            <div className="metric-label">Other Tokens (Punctuation)</div>
          </div>
          <div className="metric-card" style={{ border: cmi >= 30 ? "1px solid var(--warning)" : "none" }}>
            <div className="metric-value" style={{ color: cmi >= 30 ? "var(--warning)" : "var(--accent-primary)" }}>
              {cmi.toFixed(1)}%
            </div>
            <div className="metric-label">Code-Mixing Index (CMI)</div>
          </div>
        </div>
      </section>

      {/* ------------------------------------------------------------------ */}
      {/* Stage 3: Language-Aware Normalization */}
      {/* ------------------------------------------------------------------ */}
      <section className="stage-block" aria-labelledby="stage-3-title">
        <div className="stage-header">
          <h2 id="stage-3-title" className="stage-title">
            <span className="stage-step-num">3</span> Language-Aware Processing &amp; Intent Protection
          </h2>
          <button
            type="button"
            className="btn-secondary"
            onClick={() => setShowAuditTable(!showAuditTable)}
            style={{ fontSize: "0.8rem", padding: "4px 10px" }}
          >
            {showAuditTable ? "▲ Hide Audit Trail" : "▼ View Normalization Audit Trail"}
          </button>
        </div>

        {showAuditTable && (
          <div style={{ marginBottom: "16px" }}>
            <AuditTable auditRows={audit_rows} />
          </div>
        )}

        {/* Setup C Output Box */}
        <div style={{ background: "var(--bg-surface-sunken)", padding: "12px 16px", borderRadius: "var(--radius-md)", border: "1px solid var(--border-color)", display: "flex", justifyContent: "space-between", alignItems: "center", gap: "10px", flexWrap: "wrap" }}>
          <div>
            <div style={{ fontSize: "0.74rem", textTransform: "uppercase", fontWeight: 700, color: "var(--text-muted)" }}>
              Setup C Normalized Query String (Classifier Input)
            </div>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.95rem", fontWeight: 600, color: "var(--text-primary)", marginTop: "2px" }}>
              {text_C || "(empty)"}
            </div>
          </div>

          <button
            type="button"
            className="btn-secondary"
            onClick={() => onCopyText(text_C, "Normalized query copied!")}
            title="Copy normalized text to clipboard"
          >
            📋 Copy Text
          </button>
        </div>
      </section>

      {/* ------------------------------------------------------------------ */}
      {/* Stage 4: Support Intent Classification */}
      {/* ------------------------------------------------------------------ */}
      <section className="stage-block" aria-labelledby="stage-4-title">
        <div className="stage-header">
          <h2 id="stage-4-title" className="stage-title">
            <span className="stage-step-num">4</span> Support Intent Classification (Setup C Model)
          </h2>
          <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
            <span style={{ fontSize: "0.82rem", color: "var(--text-muted)" }}>Confidence:</span>
            <span
              style={{
                fontFamily: "var(--font-mono)",
                fontWeight: 800,
                color: topConfidence >= 0.5 ? "var(--success)" : "var(--danger)",
                fontSize: "0.95rem",
              }}
            >
              {(topConfidence * 100).toFixed(1)}%
            </span>
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "12px", marginBottom: "14px", flexWrap: "wrap" }}>
          <div style={{ fontSize: "1.1rem", fontWeight: 700 }}>
            Predicted Intent:{" "}
            <span
              style={{
                background: "var(--accent-light)",
                color: "var(--accent-primary)",
                padding: "3px 12px",
                borderRadius: "var(--radius-sm)",
                fontFamily: "var(--font-mono)",
              }}
            >
              {topIntent}
            </span>
          </div>
          {topConfidence < 0.5 && (
            <span style={{ color: "var(--danger)", fontSize: "0.84rem", fontWeight: 600 }}>
              ⚠️ Low Confidence Warning (&lt; 50%)
            </span>
          )}
        </div>

        {/* Probability Bars */}
        <IntentProbabilityBar
          probabilities={probabilities}
          topIntent={topIntent}
        />
      </section>

      {/* ------------------------------------------------------------------ */}
      {/* Stage 5: Enterprise Queue Routing */}
      {/* ------------------------------------------------------------------ */}
      <section className="stage-block" aria-labelledby="stage-5-title">
        <div className="stage-header">
          <h2 id="stage-5-title" className="stage-title">
            <span className="stage-step-num">5</span> Enterprise Queue Routing &amp; Bilingual Triage
          </h2>
          <span style={{ fontSize: "0.82rem", color: "var(--text-muted)" }}>
            6 Operational queues • Data-derived threshold CMI &ge; 30%
          </span>
        </div>

        <RoutingCard routing={routing} />
      </section>
    </div>
  );
}
