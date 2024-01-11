import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";

import Header from "./components/Header";
import Img2Img from "./components/Img2Img";
import Txt2Img from "./components/Txt2Img";

function App() {
  return (
    <Router>
      <div>
        <Header />
        <Routes>
          <Route path="/" element={<Txt2Img />} />{" "}
          <Route path="/txt2img" element={<Txt2Img />} />
          <Route path="/img2img" element={<Img2Img />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
