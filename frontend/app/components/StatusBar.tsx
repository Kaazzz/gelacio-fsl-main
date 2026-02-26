'use client'

interface Props {
  fps:        number
  bufferFill: number   // 0–100
  connected:  boolean
}

export default function StatusBar({ fps, bufferFill, connected }: Props) {
  return (
    <div
      className="flex items-center gap-4 px-4 py-2 rounded-lg text-xs flex-wrap"
      style={{ background: 'var(--bg-card)', border: '1px solid var(--border-dim)' }}
    >
      {/* FPS */}
      <div className="flex items-center gap-1.5">
        <span style={{ color: 'var(--text-muted)' }}>FPS</span>
        <span
          className="font-mono font-bold w-8 text-right"
          style={{ color: fps > 20 ? 'var(--neon-green)' : fps > 10 ? '#ffcc00' : 'var(--neon-pink)' }}
        >
          {fps}
        </span>
      </div>

      <div style={{ width: 1, height: 14, background: 'var(--border-dim)' }} />

      {/* Buffer fill */}
      <div className="flex items-center gap-2">
        <span style={{ color: 'var(--text-muted)' }}>Buffer</span>
        <div
          className="w-20 rounded-full overflow-hidden"
          style={{ height: 5, background: 'rgba(255,255,255,0.06)' }}
        >
          <div
            className="h-full rounded-full transition-all duration-300"
            style={{
              width: `${bufferFill}%`,
              background: 'var(--neon-cyan)',
              boxShadow: '0 0 4px var(--neon-cyan)',
            }}
          />
        </div>
        <span className="font-mono" style={{ color: 'var(--neon-cyan)' }}>
          {bufferFill}%
        </span>
      </div>

      <div style={{ width: 1, height: 14, background: 'var(--border-dim)' }} />

      {/* Connection status */}
      <div className="flex items-center gap-1.5">
        <span
          className="w-2 h-2 rounded-full"
          style={{
            background: connected ? 'var(--neon-green)' : 'var(--text-muted)',
            boxShadow: connected ? '0 0 6px var(--neon-green)' : 'none',
          }}
        />
        <span style={{ color: connected ? 'var(--neon-green)' : 'var(--text-muted)' }}>
          {connected ? 'Connected' : 'Connecting…'}
        </span>
      </div>

      <div style={{ width: 1, height: 14, background: 'var(--border-dim)' }} />

      {/* Live badge */}
      <div
        className="flex items-center gap-1 px-2 py-0.5 rounded-full"
        style={{ background: 'rgba(57,255,20,0.1)', border: '1px solid rgba(57,255,20,0.3)' }}
      >
        <span
          className="w-1.5 h-1.5 rounded-full animate-pulse"
          style={{ background: 'var(--neon-green)' }}
        />
        <span style={{ color: 'var(--neon-green)', fontWeight: 700 }}>LIVE</span>
      </div>
    </div>
  )
}
