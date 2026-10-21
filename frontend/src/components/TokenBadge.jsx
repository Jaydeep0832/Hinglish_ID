import React, { useState } from "react";
import { AMBIGUOUS_SET } from "../data/ambiguousWords";

export function TokenBadge({ token, tag, explanation, index }) {
  const [showTooltip, setShowTooltip] = useState(false);
  const isAmbiguous = AMBIGUOUS_SET.has(token.toLowerCase());

  return (
    <div
      style={{ position: "relative", display: "inline-block" }}
      onMouseEnter={() => setShowTooltip(true)}
      onMouseLeave={() => setShowTooltip(false)}
      onClick={() => setShowTooltip(!showTooltip)}
    >
      <span
        className={`token-badge token-${tag} ${isAmbiguous ? "token-amb" : ""}`}
        tabIndex="0"
        role="button"
        aria-label={`Token ${token}, Tag ${tag}`}
      >
        <span>{token}</span>
        <small>
          {tag}
          {isAmbiguous ? " ⚑" : ""}
        </small>
      </span>

      {showTooltip && (
        <div
          style={{
            position: "absolute",
            bottom: "calc(100% + 6px)",
            left: "50%",
            transform: "translateX(-50%)",
            backgroundColor: "var(--bg-surface-elevated)",
            color: "var(--text-primary)",
            padding: "8px 12px",
            borderRadius: "var(--radius-md)",
            border: "1px solid var(--border-color)",
            boxShadow: "var(--shadow-lg)",
            fontSize: "0.78rem",
            whiteSpace: "nowrap",
            zIndex: 40,
            pointerEvents: "none",
            display: "flex",
            flexDirection: "column",
            gap: "3px",
          }}
        >
          <div style={{ fontWeight: 700, display: "flex", gap: "6px", alignItems: "center" }}>
            <span>Token #{index}: <code>{token}</code></span>
            <span style={{ color: tag === "HI" ? "var(--hi-color)" : tag === "EN" ? "var(--en-color)" : "var(--other-color)" }}>
              [{tag}]
            </span>
          </div>
          <div style={{ color: "var(--text-secondary)" }}>
            {explanation || "Assigned via character n-gram representation."}
          </div>
          {isAmbiguous && (
            <div style={{ color: "var(--accent-primary)", fontWeight: 600 }}>
              ⚑ Pre-frozen cross-lingual homograph disambiguated via local context.
            </div>
          )}
        </div>
      )}
    </div>
  );
}
