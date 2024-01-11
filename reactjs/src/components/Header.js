import React from "react";
import "bootstrap/dist/css/bootstrap.min.css";

function Header() {
  return (
    <header className="header">
      <nav className="navbar navbar-dark bg-dark navbar-expand-lg">
        <a className="navbar-brand custom-brand" href="/">
          Stable Diffusion
        </a>
        <button
          className="navbar-toggler"
          type="button"
          data-toggle="collapse"
          data-target="#navbarNav"
          aria-controls="navbarNav"
          aria-expanded="false"
          aria-label="Toggle navigation"
        >
          <span className="navbar-toggler-icon"></span>
        </button>
        <div className="collapse navbar-collapse float-right" id="navbarNav">
          <ul className="navbar-nav ml-auto">
            <li className="nav-item">
              <a className="nav-link" href="/txt2img">
                txt2img
              </a>
            </li>
            <li className="nav-item">
              <a className="nav-link" href="/img2img">
                img2img
              </a>
            </li>
            <li className="nav-item">
              <a className="nav-link" href="/">
                Extras
              </a>
            </li>
            <li className="nav-item">
              <a className="nav-link" href="/">
                Settings
              </a>
            </li>
          </ul>
        </div>
      </nav>
    </header>
  );
}

export default Header;
