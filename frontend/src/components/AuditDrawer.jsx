import React from "react";

export function AuditDrawer({ isOpen, onClose, analysis, onCopyText }) {
  if (!isOpen) return null;

  const jsonString = analysis ? JSON.stringify(analysis, null, 2) : "{}";

  const handleDownload = () => {
    if (!analysis) return;
    const blob = new Blob([jsonString], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `hinglish_analysis_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="drawer-backdrop" onClick={onClose} role="dialog" aria-modal="true" aria-labelledby="drawer-title">
      <div className="drawer-panel" onClick={(e) => e.stopPropagation()}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid var(--border-color)", paddingBottom: "14px" }}>
          <h2 id="drawer-title" style={{ fontSize: "1.15rem", fontWeight: 800 }}>
            📊 Pipeline Inspector &amp; JSON Export
          </h2>
          <button type="button" className="btn-icon" onClick={onClose} aria-label="Close drawer">
            ✕
          </button>
        </div>

        {/* Action Buttons */}
        <div style={{ display: "flex", gap: "8px" }}>
          <button
            type="button"
            className="btn-primary"
            onClick={() => onCopyText(jsonString, "Analysis JSON copied to clipboard!")}
            style={{ fontSize: "0.82rem" }}
          >
            📋 Copy JSON
          </button>
          <button
            type="button"
            className="btn-secondary"
            onClick={handleDownload}
            style={{ fontSize: "0.82rem" }}
          >
            💾 Download File
          </button>
        </div>

        {/* JSON Preview Box */}
        <div style={{ flex: 1, minHeight: "260px", background: "var(--bg-surface-sunken)", border: "1px solid var(--border-color)", borderRadius: "var(--radius-md)", padding: "12px", overflow: "auto" }}>
          <pre style={{ fontFamily: "var(--font-mono)", fontSize: "0.78rem", color: "var(--text-primary)" }}>
            {jsonString}
          </pre>
        </div>

        {/* API Integration Snippet */}
        <div style={{ borderTop: "1px solid var(--border-color)", paddingTop: "14px" }}>
          <h3 style={{ fontSize: "0.9rem", fontWeight: 700, marginBottom: "6px" }}>
            🚀 FastAPI Production Integration
          </h3>
          <p style={{ fontSize: "0.78rem", color: "var(--text-muted)", marginBottom: "8px" }}>
            Query this endpoint directly via cURL or Python requests:
          </p>
          <pre style={{ background: "var(--bg-surface-sunken)", padding: "8px 10px", borderRadius: "var(--radius-sm)", fontSize: "0.74rem", fontFamily: "var(--font-mono)" }}>
{`curl -X POST "http://localhost:8000/api/analyze" \\
  -H "Content-Type: application/json" \\
  -d '{"query": "${analysis ? analysis.raw_query.slice(0, 30) : "Mera refund do"}..."}'`}
          </pre>
        </div>
      </div>
    </div>
  );
}
