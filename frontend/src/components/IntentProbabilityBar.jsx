import React from "react";

const INTENT_ICONS = {
  refund: "💰",
  cancel_order: "❌",
  order_not_delivered: "📦",
  delivery_delay: "⏳",
  payment_failed: "💳",
  other: "💬",
};

export function IntentProbabilityBar({ probabilities, topIntent }) {
  if (!probabilities || probabilities.length === 0) return null;

  return (
    <div className="prob-list" aria-label="Support Intent Probability Distribution">
      {probabilities.map(({ intent, prob }) => {
        const isTop = intent === topIntent;
        const percentage = Math.round(prob * 100);

        return (
          <div key={intent} className={`prob-row ${isTop ? "top" : ""}`}>
            <div className="prob-intent-name">
              <span>{INTENT_ICONS[intent] || "🏷️"}</span>
              <span style={{ fontWeight: isTop ? 800 : 600 }}>{intent}</span>
            </div>

            <div className="prob-bar-track" role="progressbar" aria-valuenow={percentage} aria-valuemin="0" aria-valuemax="100">
              <div
                className="prob-bar-fill"
                style={{ width: `${Math.max(percentage, 3)}%` }}
              />
            </div>

            <div className="prob-pct" style={{ color: isTop ? "var(--accent-primary)" : "var(--text-muted)" }}>
              {percentage}%
            </div>
          </div>
        );
      })}
    </div>
  );
}
