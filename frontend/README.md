# HinglishID-Router web interface

This directory contains the React 19 and Vite interface for exploring the Hinglish customer-support analysis pipeline. The UI includes sample messages, routing thresholds, token-level language tags, normalization details, intent scores, comparison views, and an audit drawer.

## Run locally

Requirements: Node.js 20.19+ or 22.12+ and npm, matching the Vite 8 engine requirement declared in <code>package.json</code>.

    npm ci
    npm run dev

Open the URL reported by Vite, normally <code>http://localhost:5173/</code>. The interface tries the optional API at <code>http://localhost:8000/api/analyze</code>; when it cannot connect, it runs the browser-side demo engine.

## Build and lint

    npm run build
    npm run lint

The Python pipeline, API setup, model generation, research results, and data notes are documented in the repository-root README.
