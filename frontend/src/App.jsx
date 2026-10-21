import React, { useState, useEffect, useCallback } from "react";
import { Navbar } from "./components/Navbar";
import { Sidebar } from "./components/Sidebar";
import { QueryInput } from "./components/QueryInput";
import { PipelineStages } from "./components/PipelineStages";
import { SetupComparison } from "./components/SetupComparison";
import { ScientificFindings } from "./components/ScientificFindings";
import { AuditDrawer } from "./components/AuditDrawer";
import { Toast } from "./components/Toast";
import {
  analyzeHinglishQuery,
  DEFAULT_CONF_THRESHOLD,
  DEFAULT_CMI_THRESHOLD,
} from "./services/pipelineService";
import { SAMPLE_QUERIES } from "./data/samples";

export function App() {
  // Theme state with localStorage persistence
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem("hidr-theme") || "dark";
  });

  // Query & Thresholds
  const [query, setQuery] = useState(SAMPLE_QUERIES[0].text);
  const [activeSampleId, setActiveSampleId] = useState(SAMPLE_QUERIES[0].id);
  const [confThreshold, setConfThreshold] = useState(DEFAULT_CONF_THRESHOLD);
  const [cmiThreshold, setCmiThreshold] = useState(DEFAULT_CMI_THRESHOLD);

  // Analysis Result
  const [analysis, setAnalysis] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // Tabs & Drawers
  const [activeTab, setActiveTab] = useState("stages"); // "stages", "comparison", "evidence"
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  const [toasts, setToasts] = useState([]);

  // Sync theme attribute to <html> element
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("hidr-theme", theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme((prev) => (prev === "dark" ? "light" : "dark"));
  };

  // Toast notification helper
  const addToast = useCallback((message, icon = "✅") => {
    const id = Date.now() + Math.random();
    setToasts((prev) => [...prev, { id, message, icon }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 3200);
  }, []);

  const dismissToast = (id) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  };

  // Copy helper
  const handleCopyText = (text, message = "Copied to clipboard!") => {
    if (!text) return;
    navigator.clipboard.writeText(text);
    addToast(message, "📋");
  };

  // Analysis runner
  const runAnalysis = useCallback(async (textToAnalyze, options = {}) => {
    if (!textToAnalyze || !textToAnalyze.trim()) return;

    setIsAnalyzing(true);
    try {
      const result = await analyzeHinglishQuery(textToAnalyze, {
        confThreshold: options.confThreshold ?? confThreshold,
        cmiThreshold: options.cmiThreshold ?? cmiThreshold,
        useApi: true,
      });
      setAnalysis(result);
    } catch (err) {
      addToast("Error during analysis. Please check input.", "⚠️");
    } finally {
      setIsAnalyzing(false);
    }
  }, [confThreshold, cmiThreshold, addToast]);

  // Initial analysis on load
  useEffect(() => {
    runAnalysis(query);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Re-run routing when sliders change without re-tokenizing
  useEffect(() => {
    if (analysis && query.trim()) {
      runAnalysis(query, { confThreshold, cmiThreshold });
    }
  }, [confThreshold, cmiThreshold]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleSelectSample = (sample) => {
    setQuery(sample.text);
    setActiveSampleId(sample.id);
    runAnalysis(sample.text);
    addToast(`Loaded ${sample.shortLabel}`, "💡");
  };

  const handleResetControls = () => {
    setConfThreshold(DEFAULT_CONF_THRESHOLD);
    setCmiThreshold(DEFAULT_CMI_THRESHOLD);
    addToast("Controls reset to empirical defaults.", "🔄");
  };

  return (
    <div className="app-container">
      {/* Top Navigation */}
      <Navbar
        theme={theme}
        onToggleTheme={toggleTheme}
        onOpenDrawer={() => setIsDrawerOpen(true)}
        latencyMs={analysis?.latency_ms}
        source={analysis?.source}
      />

      {/* Main Body */}
      <div className="main-layout">
        {/* Left Sidebar: Controls & Benchmarks */}
        <Sidebar
          confThreshold={confThreshold}
          setConfThreshold={setConfThreshold}
          cmiThreshold={cmiThreshold}
          setCmiThreshold={setCmiThreshold}
          onResetControls={handleResetControls}
        />

        {/* Center / Right Content Area */}
        <main className="content-area">
          {/* Query Input Section */}
          <QueryInput
            query={query}
            setQuery={(val) => {
              setQuery(val);
              setActiveSampleId(null);
            }}
            onAnalyze={() => runAnalysis(query)}
            onSelectSample={handleSelectSample}
            activeSampleId={activeSampleId}
            isAnalyzing={isAnalyzing}
          />

          {/* Navigation Tabs */}
          <div className="tabs-header" role="tablist">
            <button
              type="button"
              role="tab"
              aria-selected={activeTab === "stages"}
              className={`tab-btn ${activeTab === "stages" ? "active" : ""}`}
              onClick={() => setActiveTab("stages")}
            >
              <span>🔍</span> End-to-End Pipeline Stages
            </button>

            <button
              type="button"
              role="tab"
              aria-selected={activeTab === "comparison"}
              className={`tab-btn ${activeTab === "comparison" ? "active" : ""}`}
              onClick={() => setActiveTab("comparison")}
            >
              <span>⚖️</span> 3-Way Setup Comparison (A vs B vs C)
            </button>

            <button
              type="button"
              role="tab"
              aria-selected={activeTab === "evidence"}
              className={`tab-btn ${activeTab === "evidence" ? "active" : ""}`}
              onClick={() => setActiveTab("evidence")}
            >
              <span>📖</span> Scientific Findings &amp; Limitations
            </button>
          </div>

          {/* Tab Views */}
          {activeTab === "stages" && (
            <PipelineStages
              analysis={analysis}
              onCopyText={handleCopyText}
            />
          )}

          {activeTab === "comparison" && (
            <SetupComparison
              analysis={analysis}
              onCopyText={handleCopyText}
            />
          )}

          {activeTab === "evidence" && (
            <ScientificFindings />
          )}
        </main>
      </div>

      {/* Slide-out Inspector Drawer */}
      <AuditDrawer
        isOpen={isDrawerOpen}
        onClose={() => setIsDrawerOpen(false)}
        analysis={analysis}
        onCopyText={handleCopyText}
      />

      {/* Floating Status Toasts */}
      <Toast toasts={toasts} onDismiss={dismissToast} />
    </div>
  );
}

export default App;
