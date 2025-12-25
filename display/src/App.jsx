import { useState, useEffect } from "react";
import "./App.css";

function App() {
  const [status, setStatus] = useState({
    label: "Unknown",
    isSport: false,
    isCommercial: false,
    confidence: 0,
    connected: false,
  });

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8765");

    ws.onopen = () => {
      console.log("WebSocket connected");
      setStatus((prev) => ({ ...prev, connected: true }));
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.type === "status_update" || data.type === "label_changed") {
          setStatus({
            label: data.label || "Unknown",
            isSport: data.is_sport || false,
            isCommercial: data.is_commercial || false,
            confidence: data.confidence || 0,
            connected: true,
          });
        }
      } catch (error) {
        console.error("Error parsing WebSocket message:", error);
      }
    };

    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
      setStatus((prev) => ({ ...prev, connected: false }));
    };

    ws.onclose = () => {
      console.log("WebSocket disconnected");
      setStatus((prev) => ({ ...prev, connected: false }));

      setTimeout(() => {
        const newWs = new WebSocket("ws://localhost:8765");
        newWs.onopen = () =>
          setStatus((prev) => ({ ...prev, connected: true }));
        newWs.onmessage = ws.onmessage;
        newWs.onerror = ws.onerror;
        newWs.onclose = ws.onclose;
      }, 3000);
    };

    return () => {
      ws.close();
    };
  }, []);

  const getStatusColor = () => {
    if (!status.connected) return "rgba(150, 150, 150, 0.5)";
    if (status.isSport) return "rgba(100, 255, 100, 0.5)";
    if (status.isCommercial) return "rgba(255, 100, 100, 0.5)";
    return "rgba(150, 150, 150, 0.5)";
  };

  const getStatusText = () => {
    if (!status.connected) return "Disconnected";
    if (status.isSport) return "SPORT";
    if (status.isCommercial) return "COMMERCIAL";
    return "UNKNOWN";
  };

  return <div className="app" style={{ borderColor: getStatusColor() }}></div>;
}

export default App;
