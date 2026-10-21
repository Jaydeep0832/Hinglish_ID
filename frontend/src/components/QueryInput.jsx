import React from "react";
import { SAMPLE_QUERIES } from "../data/samples";

export function QueryInput({
  query,
  setQuery,
  onAnalyze,
  onSelectSample,
  activeSampleId,
  isAnalyzing,
}) {
  const handleKeyDown = (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      onAnalyze();
    }
  };

  const isEmpty = !query.trim();

  return (
    <section className="input-section" aria-label="Customer Query Input">
      {/* Sample Selectors */}
      <div className="sample-buttons-wrapper">
        <span className="sample-label">Benchmark Samples:</span>
        {SAMPLE_QUERIES.map((sample) => (
          <button
            key={sample.id}
            type="button"
            className={`btn-sample ${activeSampleId === sample.id ? "active" : ""}`}
            onClick={() => onSelectSample(sample)}
            title={`${sample.text} (Expected: ${sample.expected})`}
            aria-label={`Load ${sample.label}`}
          >
            {sample.id}. {sample.shortLabel}
          </button>
        ))}
      </div>

      {/* Textarea container */}
      <div className="textarea-container">
        <textarea
          className="query-textarea"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type or paste a customer support query in romanized Hindi-English (Hinglish)... e.g. Mera order deliver nahi hua, please refund karo"
          aria-label="Customer Support Query Input"
        />

        <div className="textarea-footer">
          <div className="char-counter">
            <span>{query.length} characters</span>
            {isEmpty && <span style={{ color: "var(--warning)", marginLeft: "8px" }}>⚠️ Enter text to analyze</span>}
          </div>

          <div className="textarea-actions">
            {query.length > 0 && (
              <button
                type="button"
                className="btn-secondary"
                onClick={() => setQuery("")}
                title="Clear current text"
              >
                Clear
              </button>
            )}

            <button
              type="button"
              className="btn-primary"
              onClick={onAnalyze}
              disabled={isEmpty || isAnalyzing}
              title="Execute full linguistic pipeline analysis (Ctrl+Enter)"
            >
              {isAnalyzing ? "Processing..." : "Analyze & Route Query ➔"}
            </button>
          </div>
        </div>
      </div>
    </section>
  );
}
