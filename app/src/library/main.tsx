import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { LibraryApp } from "./LibraryApp";
import "../styles.css";
import "./library.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <LibraryApp />
  </StrictMode>,
);
