'use client'

import { SignPrediction } from '../lib/api'
import { getNeonColor, formatPct } from '../lib/utils'

interface Props {
  predictions: SignPrediction[]
}

export default function PredictionDisplay({ predictions }: Props) {
  const top = predictions[0]

  return (
    <div className="flex flex-col gap-4 h-full">
      <span
        className="text-xs font-semibold uppercase tracking-widest"
        style={{ color: 'var(--neon-cyan)' }}
      >
        Predictions
      </span>

      {/* Top sign — large display */}
      <div
        className="rounded-lg p-4 text-center"
        style={{ background: 'rgba(0,255,204,0.05)', border: '1px solid rgba(0,255,204,0.2)' }}
      >
        {top ? (
          <>
            <p className="text-xs uppercase tracking-widest mb-1" style={{ color: 'var(--text-muted)' }}>
              Detected Sign
            </p>
            <p
              className="text-3xl font-black uppercase tracking-wide text-glow-cyan"
              style={{ color: 'var(--neon-cyan)' }}
            >
              {top.sign}
            </p>
            <p className="text-lg font-bold mt-1" style={{ color: 'var(--neon-green)' }}>
              {formatPct(top.probability)}
            </p>
          </>
        ) : (
          <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
            Waiting for detection…
          </p>
        )}
      </div>

      {/* Top-5 bars */}
      <div className="flex flex-col gap-3 flex-1">
        {predictions.length > 0 ? (
          predictions.map((pred, i) => {
            const color = getNeonColor(i)
            const pct   = pred.probability * 100
            return (
              <div key={pred.sign} className="flex flex-col gap-1">
                <div className="flex justify-between text-xs">
                  <span className="font-semibold" style={{ color: i === 0 ? 'var(--neon-cyan)' : 'var(--text-primary)' }}>
                    {i === 0 ? '🏆 ' : `${i + 1}.  `}{pred.sign}
                  </span>
                  <span style={{ color: 'var(--text-muted)' }}>{formatPct(pred.probability)}</span>
                </div>
                <div
                  className="w-full rounded-full overflow-hidden"
                  style={{ height: 6, background: 'rgba(255,255,255,0.05)' }}
                >
                  <div
                    className="h-full rounded-full neon-bar transition-all duration-500"
                    style={{
                      width: `${pct}%`,
                      background: color,
                      boxShadow: `0 0 6px ${color}`,
                    }}
                  />
                </div>
              </div>
            )
          })
        ) : (
          // Placeholder skeleton bars
          Array.from({ length: 5 }, (_, i) => (
            <div key={i} className="flex flex-col gap-1">
              <div className="flex justify-between text-xs">
                <span style={{ color: 'var(--border-dim)' }}>—</span>
                <span style={{ color: 'var(--border-dim)' }}>0%</span>
              </div>
              <div
                className="w-full rounded-full"
                style={{ height: 6, background: 'rgba(255,255,255,0.04)' }}
              />
            </div>
          ))
        )}
      </div>

      {/* Model info */}
      <div className="pt-3" style={{ borderTop: '1px solid var(--border-dim)' }}>
        <div className="flex flex-col gap-1 text-xs" style={{ color: 'var(--text-muted)' }}>
          <div className="flex justify-between">
            <span>Model</span>
            <span>LandmarkTfmV4</span>
          </div>
          <div className="flex justify-between">
            <span>Classes</span>
            <span>105</span>
          </div>
        </div>
      </div>
    </div>
  )
}
