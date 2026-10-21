import React, { useState } from "react";

export function ScientificFindings() {
  const [openSection, setOpenSection] = useState("sec-1");

  const toggleSection = (id) => {
    setOpenSection(openSection === id ? null : id);
  };

  return (
    <div aria-label="Scientific Findings and Limitations">
      <div style={{ marginBottom: "16px" }}>
        <h2 style={{ fontSize: "1.2rem", fontWeight: 800 }}>
          🔬 Scientific Findings &amp; Academic Limitations
        </h2>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.88rem", marginTop: "4px" }}>
          In the interest of empirical integrity, we openly document our pre-registered hypotheses, statistical tests, negative results, and failure taxonomy.
        </p>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
        {/* Section 1 */}
        <div className="card">
          <div
            onClick={() => toggleSection("sec-1")}
            style={{ display: "flex", justifyContent: "space-between", alignItems: "center", cursor: "pointer" }}
          >
            <h3 style={{ fontSize: "1rem", fontWeight: 700, display: "flex", gap: "8px", alignItems: "center" }}>
              <span>1.</span> Why Generic Subwords (Setup B) Are So Competitive on Hinglish
            </h3>
            <span style={{ fontSize: "1.1rem", color: "var(--text-muted)" }}>
              {openSection === "sec-1" ? "−" : "+"}
            </span>
          </div>

          {openSection === "sec-1" && (
            <div style={{ marginTop: "12px", fontSize: "0.88rem", color: "var(--text-secondary)", lineHeight: 1.6, borderTop: "1px solid var(--border-color)", paddingTop: "12px" }}>
              <p>
                Our 15 locked CV folds demonstrated that generic character boundary n-grams (<code>char_wb</code> 1–4) achieve <strong>93.08% Macro-F1</strong>, decisively outperforming the monolingual English pipeline (+3.33% gain, 95% CI $[+0.0184, +0.0484]$, $p &lt; 0.001$).
              </p>
              <p style={{ marginTop: "8px" }}>
                Because subword n-grams decompose words into overlapping character slices, phonetic spelling noise naturally clusters in subword space:
              </p>
              <div style={{ background: "var(--bg-surface-sunken)", padding: "10px 14px", borderRadius: "var(--radius-sm)", fontFamily: "var(--font-mono)", fontSize: "0.82rem", margin: "8px 0" }}>
                &bull; <code>kro</code> &rarr; [' k', 'kr', 'kro', 'ro ']<br />
                &bull; <code>karo</code> &rarr; [' k', 'ka', 'kar', 'karo', 'ar', 'aro', 'ro ']
              </div>
              <p>
                Both variants share the boundary subwords <code>[' k', 'ro ']</code>, allowing the linear model to assign high predictive weights without requiring an upstream phonetic lexicon.
              </p>
            </div>
          )}
        </div>

        {/* Section 2 */}
        <div className="card">
          <div
            onClick={() => toggleSection("sec-2")}
            style={{ display: "flex", justifyContent: "space-between", alignItems: "center", cursor: "pointer" }}
          >
            <h3 style={{ fontSize: "1rem", fontWeight: 700, display: "flex", gap: "8px", alignItems: "center" }}>
              <span>2.</span> The Statistical Reality of Word Normalization (Setup C' vs Setup A)
            </h3>
            <span style={{ fontSize: "1.1rem", color: "var(--text-muted)" }}>
              {openSection === "sec-2" ? "−" : "+"}
            </span>
          </div>

          {openSection === "sec-2" && (
            <div style={{ marginTop: "12px", fontSize: "0.88rem", color: "var(--text-secondary)", lineHeight: 1.6, borderTop: "1px solid var(--border-color)", paddingTop: "12px" }}>
              <p>
                In an isolated apples-to-apples word-level comparison, LID-aware normalization with word TF-IDF (Setup C': 88.99% Macro-F1) versus the raw word baseline (Setup A: 89.75% Macro-F1) yielded a paired difference of:
              </p>
              <div style={{ background: "var(--bg-surface-sunken)", padding: "10px 14px", borderRadius: "var(--radius-sm)", fontFamily: "var(--font-mono)", fontSize: "0.84rem", margin: "8px 0", color: "var(--accent-primary)", fontWeight: 700 }}>
                &Delta; Macro-F1 = -0.0076 &bull; 95% Bootstrap CI: [-0.0163, +0.0007] &bull; p = 0.117
              </div>
              <p>
                <strong>The 95% bootstrap confidence interval crosses zero.</strong> We report honestly that on moderate-sized domain corpora ($N=360$), word-level phonetic standardization alone yields no statistically significant advantage over raw unnormalized words.
              </p>
            </div>
          )}
        </div>

        {/* Section 3 */}
        <div className="card">
          <div
            onClick={() => toggleSection("sec-3")}
            style={{ display: "flex", justifyContent: "space-between", alignItems: "center", cursor: "pointer" }}
          >
            <h3 style={{ fontSize: "1rem", fontWeight: 700, display: "flex", gap: "8px", alignItems: "center" }}>
              <span>3.</span> Where LID-Aware Processing Provides Actionable Value
            </h3>
            <span style={{ fontSize: "1.1rem", color: "var(--text-muted)" }}>
              {openSection === "sec-3" ? "−" : "+"}
            </span>
          </div>

          {openSection === "sec-3" && (
            <div style={{ marginTop: "12px", fontSize: "0.88rem", color: "var(--text-secondary)", lineHeight: 1.6, borderTop: "1px solid var(--border-color)", paddingTop: "12px" }}>
              <ol style={{ paddingLeft: "18px", display: "flex", flexDirection: "column", gap: "8px" }}>
                <li>
                  <strong>Asymmetric Financial Grievance Resolution:</strong> Setup C surpasses Setup B on the two most costly customer complaints: <code>refund</code> (<strong>0.9606</strong> vs 0.9555) and <code>cancel_order</code> (<strong>0.9723</strong> vs 0.9697).
                </li>
                <li>
                  <strong>Homograph Collision Prevention:</strong> Resolves destructive collisions where English stopwords like <em>'do'</em> or <em>'me'</em> would otherwise be pruned, protecting operative Hindi imperatives (e.g. <em>'paise wapas do'</em>).
                </li>
                <li>
                  <strong>Data-Derived Queue Dispatch:</strong> The Code-Mixing Index (CMI) accurately routes 37.8% of queries displaying substantial syntactic mixing to specialized bilingual call-center agents.
                </li>
              </ol>
            </div>
          )}
        </div>

        {/* Section 4 */}
        <div className="card">
          <div
            onClick={() => toggleSection("sec-4")}
            style={{ display: "flex", justifyContent: "space-between", alignItems: "center", cursor: "pointer" }}
          >
            <h3 style={{ fontSize: "1rem", fontWeight: 700, display: "flex", gap: "8px", alignItems: "center" }}>
              <span>4.</span> Model Failure Taxonomy &amp; Edge Cases
            </h3>
            <span style={{ fontSize: "1.1rem", color: "var(--text-muted)" }}>
              {openSection === "sec-4" ? "−" : "+"}
            </span>
          </div>

          {openSection === "sec-4" && (
            <div style={{ marginTop: "12px", fontSize: "0.88rem", color: "var(--text-secondary)", lineHeight: 1.6, borderTop: "1px solid var(--border-color)", paddingTop: "12px" }}>
              <ul style={{ paddingLeft: "18px", display: "flex", flexDirection: "column", gap: "8px" }}>
                <li>
                  <strong>Sparse Context Failure:</strong> Isolated homographs with zero surrounding context (e.g., typing just <em>'is'</em>) lack surrounding words for voting and fall back to the unigram prior.
                </li>
                <li>
                  <strong>Novel Slang / Dialects:</strong> Informal Romanizations outside our 224-rule dictionary rely purely on subword character n-grams.
                </li>
                <li>
                  <strong>Brand Entity Confusion:</strong> Unseen commercial brand names (<em>Zomato, Zepto, Paytm</em>) can occasionally be categorized as Hindi or English rather than OTHER.
                </li>
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
