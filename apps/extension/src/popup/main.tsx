import React from "react";
import { createRoot } from "react-dom/client";
import { PopupApp } from "./PopupApp";
import "../styles/popup.css";

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <PopupApp />
  </React.StrictMode>
);
