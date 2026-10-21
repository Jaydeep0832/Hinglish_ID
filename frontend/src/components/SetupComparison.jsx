import React from "react";

export function SetupComparison({ analysis, onCopyText }) {
  if (!analysis) {
    return (
      <div className="card" style={{ textAlign: "center", padding: "40px 20px" }}>
        <p style={{ color: "var(--text-muted)" }}>Run an analysis above to see live 3-way setup comparisons.</p>
      </div>
    );
  }

  const { text_A, text_B, text_C, intent_results } = analysis;

  const resultA = intent_results.Setup_A;
  const resultB = intent_results.Setup_B;
  const resultC = intent_results.Setup_C;

  return (
    <div>
      <div style={{ marginBottom: "16px" }}>
        <h2 style={{ fontSize: "1.2rem", fontWeight: 800 }}>
          ⚖️ Live 3-Way Setup Comparison (Fair Isolation Benchmark)
        </h2>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.88rem", marginTop: "4px" }}>
          Observe how each representation pipeline transforms the identical customer support utterance and how the underlying classifier evaluates it.
        </p>
      </div>

      <div className="comparison-grid">
        {/* Setup A */}
        <div className="comp-card">
          <span className="comp-badge" style={{ background: "var(--danger-bg)", color: "var(--danger)" }}>
            Setup A • Monolingual English Baseline
          </span>
          <h3 style={{ fontSize: "1rem", fontWeight: 700, margin: "4px 0" }}>
            Blind English Preprocessing
          </h3>
          <div style={{ fontSize: "0.78rem", color: "var(--text-muted)" }}>
            Lowercased + English Stopwords + Porter Stemmer + Word TF-IDF (1, 2)
          </div>

          <div className="comp-code-box">
            {text_A || "(empty after stopword filtering)"}
          </div>

          <button
            type="button"
            className="btn-secondary"
            onClick={() => onCopyText(text_A, "Setup A text copied!")}
            style={{ fontSize: "0.76rem", alignSelf: "flex-start", marginBottom: "12px" }}
          >
            📋 Copy
          </button>

          <div style={{ marginTop: "auto", borderTop: "1px solid var(--border-color)", paddingTop: "12px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "6px" }}>
              <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>Prediction:</span>
              <span style={{ fontFamily: "var(--font-mono)", fontWeight: 700 }}>
                {resultA.intent} ({(resultA.confidence * 100).toFixed(1)}%)
              </span>
            </div>
            <div style={{ padding: "8px 10px", background: "var(--bg-surface-sunken)", borderRadius: "var(--radius-sm)", fontSize: "0.78rem" }}>
              <div><strong>15-Fold Macro-F1:</strong> 0.8975 &plusmn; 0.0280 (89.82%)</div>
              <div style={{ color: "var(--danger)", marginTop: "4px" }}>
                &bull; Drops Hindi imperative verbs like <em>"do"</em> (give).<br />
                &bull; Porter stemmer mutilates Romanized Hindi roots.
              </div>
            </div>
          </div>
        </div>

        {/* Setup B */}
        <div className="comp-card highlight">
          <span className="comp-badge" style={{ background: "var(--success-bg)", color: "var(--success)" }}>
            Setup B • Generic Subword TF-IDF
          </span>
          <h3 style={{ fontSize: "1rem", fontWeight: 700, margin: "4px 0" }}>
            Unguided Subwords (No LID)
          </h3>
          <div style={{ fontSize: "0.78rem", color: "var(--text-muted)" }}>
            Clean Text with Entity Masks + char_wb TF-IDF (1, 4), NO Language ID
          </div>

          <div className="comp-code-box">
            {text_B || "(empty)"}
          </div>

          <button
            type="button"
            className="btn-secondary"
            onClick={() => onCopyText(text_B, "Setup B text copied!")}
            style={{ fontSize: "0.76rem", alignSelf: "flex-start", marginBottom: "12px" }}
          >
            📋 Copy
          </button>

          <div style={{ marginTop: "auto", borderTop: "1px solid var(--border-color)", paddingTop: "12px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "6px" }}>
              <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>Prediction:</span>
              <span style={{ fontFamily: "var(--font-mono)", fontWeight: 700, color: "var(--success)" }}>
                {resultB.intent} ({(resultB.confidence * 100).toFixed(1)}%)
              </span>
            </div>
            <div style={{ padding: "8px 10px", background: "var(--success-bg)", borderRadius: "var(--radius-sm)", fontSize: "0.78rem", color: "var(--success)" }}>
              <div><strong>15-Fold Macro-F1:</strong> 0.9308 &plusmn; 0.0272 (93.06%) [Best Overall]</div>
              <div style={{ marginTop: "4px" }}>
                &bull; Subwords naturally absorb spelling noise (<em>kro/karo</em> share <code>kr</code>, <code>ro</code>).<br />
                &bull; Strongest generalist baseline across 15 folds.
              </div>
            </div>
          </div>
        </div>

        {/* Setup C */}
        <div className="comp-card">
          <span className="comp-badge" style={{ background: "var(--accent-light)", color: "var(--accent-primary)" }}>
            Setup C • LID-Aware Normalization (Ours)
          </span>
          <h3 style={{ fontSize: "1rem", fontWeight: 700, margin: "4px 0" }}>
            LID-Aware Normalized Representation
          </h3>
          <div style={{ fontSize: "0.78rem", color: "var(--text-muted)" }}>
            224 Phonetic Rules + Intent Protection + char_wb (1, 4)
          </div>

          <div className="comp-code-box">
            {text_C || "(empty)"}
          </div>

          <button
            type="button"
            className="btn-secondary"
            onClick={() => onCopyText(text_C, "Setup C text copied!")}
            style={{ fontSize: "0.76rem", alignSelf: "flex-start", marginBottom: "12px" }}
          >
            📋 Copy
          </button>

          <div style={{ marginTop: "auto", borderTop: "1px solid var(--border-color)", paddingTop: "12px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "6px" }}>
              <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>Prediction:</span>
              <span style={{ fontFamily: "var(--font-mono)", fontWeight: 700, color: "var(--accent-primary)" }}>
                {resultC.intent} ({(resultC.confidence * 100).toFixed(1)}%)
              </span>
            </div>
            <div style={{ padding: "8px 10px", background: "var(--accent-light)", borderRadius: "var(--radius-sm)", fontSize: "0.78rem", color: "var(--accent-primary)" }}>
              <div><strong>15-Fold Macro-F1:</strong> 0.9185 &plusmn; 0.0258 (91.85%)</div>
              <div style={{ marginTop: "4px", fontWeight: 700 }}>
                &bull; Peak Accuracy on High-Stakes Grievances:<br />
                &nbsp;&nbsp;- Refund: 0.9606 (C) &gt; 0.9555 (B) &gt; 0.8992 (A)<br />
                &nbsp;&nbsp;- Cancel: 0.9723 (C) &gt; 0.9697 (B) &gt; 0.9560 (A)
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
