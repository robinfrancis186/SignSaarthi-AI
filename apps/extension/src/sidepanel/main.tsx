import React from "react";
import { createRoot } from "react-dom/client";
import { SidePanelApp } from "./SidePanelApp";
import "../styles/panel.css";

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <SidePanelApp />
  </React.StrictMode>
);
