import { useState, useEffect, useRef } from 'react'
import Galaxy from './components/Galaxy.jsx'
import './App.css'

function App() {
  // State
  const [detection, setDetection] = useState('');
  const [confidence, setConfidence] = useState(0);
  const [isActive, setIsActive] = useState(false);
  const [topPredictions, setTopPredictions] = useState([]);
  const [error, setError] = useState('');
  const [stream, setStream] = useState(null);
  
  // Refs
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const sessionId = useRef(Math.random().toString(36).substr(2, 9));
  
  // Loop Control
  const isProcessing = useRef(false);
  const shouldLoop = useRef(false);

  // Backend URL
  const BACKEND_URL = 'http://localhost:5000';

  // Start webcam
  const startWebcam = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { 
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user'
        }
      });

      setStream(mediaStream);
      setIsActive(true); // This will trigger the useEffect below
      setError('');

    } catch (err) {
      console.error('Error accessing webcam:', err);
      setError('Could not access webcam. Please grant camera permission.');
    }
  };

  // Stop webcam
  const stopWebcam = () => {
    setIsActive(false); // This will trigger the useEffect to stop loop
    
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
    }
    setStream(null);
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  };

  // --- NEW: THE LOOPER EFFECT ---
  // This watches 'isActive'. When it becomes TRUE, it kicks off the loop.
  // When it becomes FALSE, it kills the loop.
  useEffect(() => {
    if (isActive) {
      shouldLoop.current = true;
      processLoop();
    } else {
      shouldLoop.current = false;
    }
  }, [isActive]);

  // --- THE SMART LOOP ---
  const processLoop = async () => {
    // 1. Kill switch
    if (!shouldLoop.current) return;

    // 2. Check if busy (Simple Lock)
    if (isProcessing.current) {
        // Retry in 20ms
        setTimeout(processLoop, 20); 
        return;
    }

    // 3. Lock & Process
    isProcessing.current = true;
    
    try {
        await captureAndPredict();
    } catch (e) {
        console.error("Loop crashed:", e);
    } finally {
        // 4. Unlock
        isProcessing.current = false;
        
        // 5. Schedule Next Frame (200ms = ~5 FPS)
        // Only schedule if we are still supposed to be running
        if (shouldLoop.current) {
            setTimeout(processLoop, 200); 
        }
    }
  };

  // Capture frame and send to backend
  const captureAndPredict = async () => {
    // Note: We check shouldLoop.current instead of isActive to avoid stale state issues
    if (!videoRef.current || !canvasRef.current || !shouldLoop.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    
    // Check if video is actually playing
    if (video.readyState !== video.HAVE_ENOUGH_DATA) return;

    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // 0.6 quality is plenty for ML
    const imageData = canvas.toDataURL('image/jpeg', 0.6);

    try {
      const response = await fetch(`${BACKEND_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image: imageData,
          session_id: sessionId.current
        })
      });

      if (!response.ok) return;

      const result = await response.json();

      if (result.success) {
        if (result.updated) {
          setDetection(result.letter);
          setConfidence(result.confidence);
        }
        if (result.top_predictions) {
          setTopPredictions(result.top_predictions);
        }
        setError('');
      }
    } catch (err) {
       // Ignore fetch errors
    }
  };

  // Reset session
  const resetSession = async () => {
    try {
      await fetch(`${BACKEND_URL}/reset`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId.current })
      });
      setDetection('');
      setConfidence(0);
      setTopPredictions([]);
    } catch (err) {
      console.error('Reset error:', err);
    }
  };

  // Assign stream to video when ready
  useEffect(() => {
    if (isActive && videoRef.current && stream) {
      videoRef.current.srcObject = stream;
    }
  }, [isActive, stream]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      shouldLoop.current = false;
      stopWebcam();
    };
  }, []);

  return (
    <div className="app-container">
      <div className="galaxy-background">
        <Galaxy 
          mouseRepulsion mouseInteraction transparent={false} 
          density={1} glowIntensity={0.3} saturation={0} 
          hueShift={140} twinkleIntensity={0.3} rotationSpeed={0.1} 
          repulsionStrength={2} autoCenterRepulsion={0} starSpeed={0.5} speed={1}
        />
      </div>

      <div className="main-content">
        <div className="translator-card">
          
          <div className="video-section">
            <div className="video-container">
              {isActive ? (
                <>
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    className="video-feed"
                  />
                  
                  <div className="recording-indicator">
                    <div className="recording-dot"></div>
                    <span>AI ACTIVE</span>
                  </div>
                </>
              ) : (
                <div className="video-placeholder">
                  <p>{error || 'Camera Ready'}</p>
                  {!error && <div className="camera-icon">📷</div>}
                </div>
              )}
              
              <canvas ref={canvasRef} style={{ display: 'none' }} />
              
              <div className="video-controls">
                {!isActive ? (
                  <button className="control-btn start-btn" onClick={startWebcam}>
                    Start Camera
                  </button>
                ) : (
                  <>
                    <button className="control-btn stop-btn" onClick={stopWebcam}>
                      Stop Camera
                    </button>
                    <button className="control-btn reset-btn" onClick={resetSession}>
                      Reset
                    </button>
                  </>
                )}
              </div>
              
              {error && (
                <div className="error-message">
                  {error}
                </div>
              )}
            </div>
          </div>

          <div className="output-section">
            <h2 className="output-title">Detected Letter</h2>
            <div className="letter-display">
              {detection || '-'}
            </div>
            <p className="confidence-text" style={{
              color: confidence > 85 ? '#28a745' : confidence > 60 ? '#ffc107' : '#dc3545'
            }}>
              Confidence: {confidence ? `${confidence.toFixed(1)}%` : '--'}
            </p>
            
            {topPredictions.length > 0 && (
              <div className="top-predictions">
                <h3 className="predictions-title">Top Predictions</h3>
                {topPredictions.map((pred, idx) => (
                  <div 
                    key={idx} 
                    className={`prediction-item ${idx === 0 ? 'primary' : ''}`}
                  >
                    <span className="prediction-letter">
                      {idx + 1}. {pred.letter}
                    </span>
                    <span className="prediction-confidence">
                      {pred.confidence.toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>

        </div>
      </div>
    </div>
  )
}

export default App