import React from "react";

export class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error("ErrorBoundary caught an error:", error, errorInfo);
  }

  handleReload = () => {
    localStorage.clear();
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      return (
        <div style={{
          minHeight: "100vh",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: "var(--bg-main, #0a0f1d)",
          color: "var(--text-main, #f1f5f9)",
          fontFamily: "Inter, -apple-system, sans-serif",
          padding: "24px"
        }}>
          <div style={{
            maxWidth: "520px",
            background: "#161e31",
            border: "1px solid #334155",
            borderRadius: "16px",
            padding: "32px",
            textAlign: "center",
            boxShadow: "0 20px 40px rgba(0,0,0,0.5)"
          }}>
            <div style={{ fontSize: "48px", marginBottom: "16px" }}>⚠️</div>
            <h2 style={{ fontSize: "20px", fontWeight: "700", marginBottom: "8px", color: "#f8fafc" }}>
              Application Render Notice
            </h2>
            <p style={{ fontSize: "14px", color: "#94a3b8", lineHeight: 1.6, marginBottom: "20px" }}>
              The application encountered an unexpected runtime state. Click below to reload the workspace.
            </p>
            {this.state.error?.message && (
              <pre style={{
                background: "#0d1322",
                padding: "12px",
                borderRadius: "8px",
                fontSize: "12px",
                color: "#f87171",
                textAlign: "left",
                overflowX: "auto",
                marginBottom: "20px"
              }}>
                {this.state.error.message}
              </pre>
            )}
            <button
              onClick={this.handleReload}
              style={{
                background: "linear-gradient(135deg, #4f46e5, #06b6d4)",
                color: "#ffffff",
                border: "none",
                borderRadius: "8px",
                padding: "10px 24px",
                fontWeight: "600",
                fontSize: "14px",
                cursor: "pointer"
              }}
            >
              🔄 Refresh & Reload
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
