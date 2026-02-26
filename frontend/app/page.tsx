'use client'

import { useState, useCallback } from 'react'
import HeroSection from './components/HeroSection'
import CameraView from './components/CameraView'
import PredictionDisplay from './components/PredictionDisplay'
import StatusBar from './components/StatusBar'
import { SignPrediction } from './lib/api'

export default function Home() {
  const [predictions, setPredictions] = useState<SignPrediction[]>([])
  const [fps, setFps] = useState(0)
  const [bufferFill, setBufferFill] = useState(0)
  const [connected, setConnected] = useState(false)

  const handlePredictions = useCallback((preds: SignPrediction[]) => {
    setPredictions(preds)
    setConnected(true)
  }, [])

  return (
    <main className="min-h-screen flex flex-col" style={{ background: 'var(--bg-primary)' }}>
      <HeroSection />

      <div className="flex-1 max-w-7xl mx-auto w-full px-4 py-6 flex flex-col gap-4">
        {/* Status bar */}
        <StatusBar fps={fps} bufferFill={bufferFill} connected={connected} />

        {/* Main content: camera + predictions */}
        <div className="flex flex-col lg:flex-row gap-6 flex-1">
          {/* Camera panel */}
          <div className="flex-1 min-w-0">
            <div
              className="rounded-xl border p-4 h-full"
              style={{ background: 'var(--bg-card)', borderColor: 'var(--border-dim)' }}
            >
              <CameraView
                onPredictions={handlePredictions}
                onFpsUpdate={setFps}
                onBufferFill={setBufferFill}
              />
            </div>
          </div>

          {/* Prediction panel */}
          <div className="lg:w-80 xl:w-96">
            <div
              className="rounded-xl border p-4 h-full"
              style={{ background: 'var(--bg-card)', borderColor: 'var(--border-dim)' }}
            >
              <PredictionDisplay predictions={predictions} />
            </div>
          </div>
        </div>
      </div>

      <footer className="py-4 text-center text-xs" style={{ color: 'var(--text-muted)' }}>
        GelacioFSL &mdash; Real-time Sign Language Recognition &nbsp;|&nbsp; LandmarkTransformerV4
      </footer>
    </main>
  )
}
