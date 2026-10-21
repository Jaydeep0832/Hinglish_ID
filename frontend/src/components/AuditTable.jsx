import React, { useState } from "react";

export function AuditTable({ auditRows }) {
  const [search, setSearch] = useState("");
  const [filterType, setFilterType] = useState("all"); // "all", "modified", "dropped"

  const filteredRows = auditRows.filter((row) => {
    const matchesSearch =
      row.token.toLowerCase().includes(search.toLowerCase()) ||
      row.action.toLowerCase().includes(search.toLowerCase()) ||
      row.tag.toLowerCase().includes(search.toLowerCase());

    if (!matchesSearch) return false;
    if (filterType === "modified") return row.normalized !== "-" && row.normalized !== row.token.toLowerCase();
    if (filterType === "dropped") return row.isDropped;
    return true;
  });

  return (
    <div>
      {/* Table Controls */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: "12px", marginBottom: "10px", flexWrap: "wrap" }}>
        <input
          type="text"
          placeholder="Filter tokens or actions..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          style={{
            padding: "6px 12px",
            borderRadius: "var(--radius-sm)",
            border: "1px solid var(--border-color)",
            background: "var(--bg-surface)",
            fontSize: "0.82rem",
            maxWidth: "240px",
          }}
          aria-label="Filter normalization table"
        />

        <div style={{ display: "flex", gap: "6px" }}>
          <button
            type="button"
            className={`btn-secondary ${filterType === "all" ? "active" : ""}`}
            onClick={() => setFilterType("all")}
            style={{ fontSize: "0.78rem", padding: "4px 10px" }}
          >
            All ({auditRows.length})
          </button>
          <button
            type="button"
            className={`btn-secondary ${filterType === "modified" ? "active" : ""}`}
            onClick={() => setFilterType("modified")}
            style={{ fontSize: "0.78rem", padding: "4px 10px" }}
          >
            Phonetic Normalized
          </button>
          <button
            type="button"
            className={`btn-secondary ${filterType === "dropped" ? "active" : ""}`}
            onClick={() => setFilterType("dropped")}
            style={{ fontSize: "0.78rem", padding: "4px 10px" }}
          >
            Dropped Stopwords
          </button>
        </div>
      </div>

      {/* Table */}
      <div className="table-wrapper">
        <table className="audit-table">
          <thead>
            <tr>
              <th style={{ width: "60px" }}>#</th>
              <th>Token</th>
              <th style={{ width: "90px" }}>LID Tag</th>
              <th>Normalized Form</th>
              <th>Action Applied</th>
            </tr>
          </thead>
          <tbody>
            {filteredRows.length > 0 ? (
              filteredRows.map((row) => (
                <tr key={row.index}>
                  <td style={{ color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
                    {row.index}
                  </td>
                  <td style={{ fontWeight: 600, fontFamily: "var(--font-mono)" }}>
                    {row.token}
                  </td>
                  <td>
                    <span className={`token-badge token-${row.tag}`} style={{ padding: "2px 8px", fontSize: "0.78rem" }}>
                      {row.tag}
                    </span>
                  </td>
                  <td style={{ fontFamily: "var(--font-mono)", color: row.isDropped ? "var(--text-muted)" : "var(--text-primary)" }}>
                    {row.normalized}
                  </td>
                  <td>
                    <span style={{
                      color: row.isDropped ? "var(--danger)" : row.action.includes("Normalized") ? "var(--warning)" : "var(--text-secondary)",
                      fontWeight: row.isDropped ? 400 : 600,
                    }}>
                      {row.action}
                    </span>
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan="5" style={{ textAlign: "center", padding: "20px", color: "var(--text-muted)" }}>
                  No matching tokens found for this filter.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
