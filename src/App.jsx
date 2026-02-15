import { useState } from 'react'
import Galaxy from './components/Galaxy.jsx'
import './App.css'

function App() {
  const [detection, setDetection] = useState("A"); // Placeholder for detected letter

  return (
    <div className="app-container">
      
      {/* 1. The Background Layer */}
      <div className="galaxy-background">
        <Galaxy 
          mouseRepulsion
          mouseInteraction
          transparent={false}
          density={1}
          glowIntensity={0.3}
          saturation={0}
          hueShift={140}
          twinkleIntensity={0.3}
          rotationSpeed={0.1}
          repulsionStrength={2}
          autoCenterRepulsion={0}
          starSpeed={0.5}
          speed={1}
        />
      </div>

      {/* 2. The Main UI Layer (Centered) */}
      <div className="main-content">
        <div className="translator-card">
          
          {/* Left Side: Video Feed */}
          <div className="video-section">
            <div className="video-placeholder">
              {/* This is where your <video> or <canvas> tag will go */}
              <p>Camera Feed Active</p>
              <div className="recording-dot"></div>
            </div>
          </div>

          {/* Right Side: Detection Output */}
          <div className="output-section">
            <h2 className="output-title">Detected Letter</h2>
            <div className="letter-display">
              {detection}
            </div>
            <p className="confidence-text">Confidence: 98%</p>
          </div>

        </div>
      </div>

    </div>
  )
}

export default App