# Sports Identifier Display

Electron app that displays real-time sports/commercial detection status from the WebSocket server.

## Setup

1. Install dependencies:
```bash
npm install
```

2. Run in development mode:
```bash
npm run electron:dev
```

This will:
- Start the Vite dev server
- Launch the Electron app
- Connect to WebSocket server at `ws://localhost:8765`

## Build

```bash
npm run build
npm run electron
```

## Features

- Real-time status updates via WebSocket
- Visual indicator showing SPORT (green) or COMMERCIAL (orange)
- Confidence percentage display
- Connection status indicator
- Auto-reconnect on disconnect
- Transparent, always-on-top window

## Configuration

The app connects to `ws://localhost:8765` by default. Make sure the Python WebSocket server is running.
