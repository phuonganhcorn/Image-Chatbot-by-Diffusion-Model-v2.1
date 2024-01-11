import React, { useState } from "react";
import "./Img2Img.css";

function Img2Img() {
  const [width, setWidth] = useState(100);
  const [height, setHeight] = useState(100);
  const [cfgScale, setCfgScale] = useState(100);
  const [selectedCheckpoint, setSelectedCheckpoint] = useState("checkpoint1");
  const [selectedSamplingMethod, setSelectedSamplingMethod] =
    useState("method1");
  const [samplingSteps, setSamplingSteps] = useState(1);
  const [batchCount, setBatchCount] = useState(1);
  const [batchSize, setBatchSize] = useState(50);
  const [restoreFaces, setRestoreFaces] = useState(false);
  const [tiling, setTiling] = useState(false);
  const [hiresFix, setHiresFix] = useState(false);
  const [seed, setSelectedSeed] = useState("seed1");
  const [extras, setExtras] = useState(false);
  const [script, setSelectedScript] = useState("none");

  const handleSamplingMethodChange = (event) => {
    setSelectedSamplingMethod(event.target.value);
  };

  const handleSamplingStepsChange = (event) => {
    setSamplingSteps(parseInt(event.target.value));
  };
  const handleCheckpointChange = (event) => {
    setSelectedCheckpoint(event.target.value);
  };
  const handleBatchCountChange = (event) => {
    setBatchCount(parseInt(event.target.value));
  };

  const handleBatchSizeChange = (event) => {
    setBatchSize(parseInt(event.target.value));
  };

  const handleWidthChange = (event) => {
    setWidth(event.target.value);
  };

  const handleHeightChange = (event) => {
    setHeight(event.target.value);
  };

  const handleCfgScaleChange = (event) => {
    setCfgScale(event.target.value);
  };
  const handleSeedChange = (event) => {
    setSelectedSeed(event.target.value);
  };
  const handleScriptChange = (event) => {
    setSelectedScript(event.target.value);
  };

  const [uploadedImage, setUploadedImage] = useState(null);

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    setUploadedImage(URL.createObjectURL(file));
  };

  const handleSaveImage = () => {
    // Logic to save the image
  };

  const handleDeleteImage = () => {
    // Logic to delete the image
  };

  const handleShareImage = () => {
    // Logic to share the image
  };

  return (
    <div className="img2img-container">
      <div className="checkpoint-select">
        <label htmlFor="checkpoint-select">Select Checkpoint</label>
        <select
          id="checkpoint-select"
          value={selectedCheckpoint}
          onChange={handleCheckpointChange}
        >
          <option key="checkpoint1" value="checkpoint1">
            Checkpoint 1
          </option>
          <option key="checkpoint2" value="checkpoint2">
            Checkpoint 2
          </option>
          <option key="checkpoint3" value="checkpoint3">
            Checkpoint 3
          </option>
        </select>
      </div>
      <div className="img2img-content">
        <div className="image-upload">
          <label htmlFor="image-upload">Upload Image</label>
          <input
            type="file"
            id="image-upload"
            accept="image/*"
            onChange={handleImageUpload}
          />
        </div>
        <div className="button-generate-image">
          <button>Generate Image</button>
        </div>
      </div>
      <div className="img2img-wrapper">
        <div className="generate-container">
          <div className="sampling-container">
            <div className="sampling-method-select">
              <label htmlFor="sampling-method-select">Sampling Method</label>
              <select
                id="sampling-method-select"
                value={selectedSamplingMethod}
                onChange={handleSamplingMethodChange}
              >
                <option key="method1" value="method1">
                  Method 1
                </option>
                <option key="method2" value="method2">
                  Method 2
                </option>
                <option key="method3" value="method3">
                  Method 3
                </option>
              </select>
            </div>
            <div className="sampling-steps-range">
              <label htmlFor="sampling-steps-range">Sampling Steps</label>
              <div className="range-setting">
                <input
                  type="range"
                  min="1"
                  max="50"
                  value={samplingSteps}
                  onChange={handleSamplingStepsChange}
                />
                <input
                  type="number"
                  min="1"
                  max="50"
                  value={samplingSteps}
                  onChange={handleSamplingStepsChange}
                />
              </div>
            </div>
          </div>
          <div className="checkboxes">
            <label>
              <input
                type="checkbox"
                checked={restoreFaces}
                onChange={() => setRestoreFaces(!restoreFaces)}
              />
              Restore Faces
            </label>
            <label>
              <input
                type="checkbox"
                checked={tiling}
                onChange={() => setTiling(!tiling)}
              />
              Tiling
            </label>
            <label>
              <input
                type="checkbox"
                checked={hiresFix}
                onChange={() => setHiresFix(!hiresFix)}
              />
              Hires.fix
            </label>
          </div>
          <div className="width-height-batch">
            <div className="width-height-range">
              <label htmlFor="width-range" className="width-label">
                Width
              </label>
              <div className="range-setting">
                <input
                  type="range"
                  min="50"
                  max="200"
                  value={width}
                  onChange={handleWidthChange}
                />
                <input
                  type="number"
                  min="50"
                  max="200"
                  value={width}
                  onChange={handleWidthChange}
                />
              </div>

              <label htmlFor="height-range" className="height-label">
                Height
              </label>
              <div className="range-setting">
                <input
                  type="range"
                  min="50"
                  max="200"
                  value={height}
                  onChange={handleHeightChange}
                />
                <input
                  type="number"
                  min="50"
                  max="200"
                  value={height}
                  onChange={handleHeightChange}
                />
              </div>
            </div>
            <div className="batch-range">
              <label htmlFor="batch-count-range" className="batch-count-label">
                Batch Count
              </label>
              <div className="range-setting">
                <input
                  type="range"
                  min="1"
                  max="100"
                  value={batchCount}
                  onChange={handleBatchCountChange}
                />
                <input
                  type="number"
                  min="1"
                  max="100"
                  value={batchCount}
                  onChange={handleBatchCountChange}
                />
              </div>

              <label htmlFor="batch-size-range" className="batch-size-label">
                Batch Size
              </label>
              <div className="range-setting">
                <input
                  type="range"
                  min="1"
                  max="100"
                  value={batchSize}
                  onChange={handleBatchSizeChange}
                />
                <input
                  type="number"
                  min="1"
                  max="100"
                  value={batchSize}
                  onChange={handleBatchSizeChange}
                />
              </div>
            </div>
          </div>

          <div className="cfg-scale-range">
            <label htmlFor="cfg-scale-range">CFG Scale</label>
            <div className="range-setting">
              <input
                type="range"
                min="50"
                max="200"
                value={cfgScale}
                onChange={handleCfgScaleChange}
              />
              <input
                type="number"
                min="50"
                max="200"
                value={cfgScale}
                onChange={handleCfgScaleChange}
              />
            </div>
          </div>
          <div className="seed-select-option">
            <div className="seed-select">
              <label htmlFor="seed-select">Seed</label>
              <label>
                <input
                  type="checkbox"
                  checked={extras}
                  onChange={() => setExtras(!extras)}
                />
                Extras
              </label>
              <select id="seed-select" value={seed} onChange={handleSeedChange}>
                <option key="seed1" value="seed1">
                  Seed 1
                </option>
                <option key="seed2" value="seed2">
                  Seed 2
                </option>
                <option key="seed3" value="seed3">
                  Seed 3
                </option>
              </select>
            </div>
          </div>
          <div className="script-select">
            <label htmlFor="script-select">Script</label>
            <select
              id="script-select"
              value={script}
              onChange={handleScriptChange}
            >
              <option key="none" value="none">
                None
              </option>
            </select>
          </div>

          <div className="button-row">
            <button onClick={handleSaveImage}>Save</button>
            <button onClick={handleDeleteImage}>Delete</button>
            <button onClick={handleShareImage}>Share</button>
            <button>Zip</button>
            <button>Send to img2img</button>
            <button>Send to extras</button>
          </div>
        </div>
        <div className="generated-images">
          {uploadedImage && <img src={uploadedImage} alt="Generated Image" />}
        </div>
      </div>
    </div>
  );
}

export default Img2Img;
