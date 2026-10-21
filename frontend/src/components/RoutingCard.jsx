import React from "react";

export function RoutingCard({ routing }) {
  if (!routing) return null;

  const {
    queue,
    flag_heavy_mix,
    escalated_to_human,
    routing_reason,
    cmi,
    cmi_threshold,
  } = routing;

  let cardType = "normal";
  if (escalated_to_human) {
    cardType = "human-triage";
  } else if (flag_heavy_mix) {
    cardType = "escalated";
  }

  return (
    <div className={`routing-card ${cardType}`} aria-label="Support Queue Routing Information">
      <div className="queue-badge-row">
        <div>
          <span style={{ fontSize: "0.78rem", textTransform: "uppercase", fontWeight: 700, color: "var(--text-muted)", letterSpacing: "0.05em" }}>
            Destination Queue
          </span>
          <div className="queue-name">➔ {queue}</div>
        </div>

        <div>
          {flag_heavy_mix && (
            <span
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "6px",
                padding: "4px 12px",
                borderRadius: "var(--radius-full)",
                background: "var(--warning-bg)",
                border: "1px solid var(--warning-border)",
                color: "var(--warning)",
                fontSize: "0.78rem",
                fontWeight: 700,
              }}
            >
              ⚠️ Bilingual Support Flag Active
            </span>
          )}
          {escalated_to_human && (
            <span
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "6px",
                padding: "4px 12px",
                borderRadius: "var(--radius-full)",
                background: "var(--danger-bg)",
                border: "1px solid var(--danger-border)",
                color: "var(--danger)",
                fontSize: "0.78rem",
                fontWeight: 700,
                marginLeft: "6px",
              }}
            >
              🚨 Low Confidence Safeguard
            </span>
          )}
        </div>
      </div>

      <div className="routing-desc">
        <div>
          <strong>Routing Reason:</strong> {routing_reason}
        </div>
        <div style={{ marginTop: "4px" }}>
          <strong>Linguistic Code-Mixing Triage:</strong>{" "}
          {flag_heavy_mix ? (
            <span style={{ color: "var(--warning)", fontWeight: 600 }}>
              Heavy mixing detected (CMI {cmi.toFixed(1)}% &ge; {cmi_threshold.toFixed(0)}% threshold). Dispatched to specialized bilingual English-Hindi tier.
            </span>
          ) : (
            <span style={{ color: "var(--text-secondary)" }}>
              Standard monolingual or low-mixing workflow (CMI {cmi.toFixed(1)}% &lt; {cmi_threshold.toFixed(0)}% threshold).
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
