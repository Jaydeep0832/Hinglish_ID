import React from "react";

export function Toast({ toasts, onDismiss }) {
  if (!toasts || toasts.length === 0) return null;

  return (
    <div className="toast-container" role="status" aria-live="polite">
      {toasts.map((toast) => (
        <div
          key={toast.id}
          className="toast"
          onClick={() => onDismiss(toast.id)}
          role="alert"
        >
          <span>{toast.icon || "ℹ️"}</span>
          <span>{toast.message}</span>
        </div>
      ))}
    </div>
  );
}
