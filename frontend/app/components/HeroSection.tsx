'use client'

export default function HeroSection() {
  return (
    <header
      className="relative overflow-hidden py-5 px-6"
      style={{
        background: 'linear-gradient(135deg, #0f0f1a 0%, #13132a 50%, #0f0f1a 100%)',
        borderBottom: '1px solid var(--border-dim)',
      }}
    >
      {/* Subtle neon grid overlay */}
      <div
        className="absolute inset-0 opacity-5"
        style={{
          backgroundImage:
            'linear-gradient(var(--neon-cyan) 1px, transparent 1px), linear-gradient(90deg, var(--neon-cyan) 1px, transparent 1px)',
          backgroundSize: '40px 40px',
        }}
      />

      <div className="relative max-w-7xl mx-auto flex items-center justify-between">
        {/* Left: Brand */}
        <div className="flex items-center gap-4">
          <div
            className="w-10 h-10 rounded-lg flex items-center justify-center text-lg font-black glow-cyan"
            style={{ background: 'rgba(0,255,204,0.1)', border: '1px solid var(--neon-cyan)' }}
          >
            <span style={{ color: 'var(--neon-cyan)' }}>G</span>
          </div>
          <div>
            <h1
              className="text-xl font-bold tracking-wide text-glow-cyan"
              style={{ color: 'var(--neon-cyan)' }}
            >
              GelacioFSL
            </h1>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
              Sign Language Recognition
            </p>
          </div>
        </div>

        {/* Right: Tags */}
        <div className="hidden sm:flex items-center gap-2 text-xs">
          {['105 Signs', 'MediaPipe', 'Real-time'].map((tag) => (
            <span
              key={tag}
              className="px-3 py-1 rounded-full"
              style={{
                background: 'rgba(0,255,204,0.08)',
                border: '1px solid rgba(0,255,204,0.3)',
                color: 'var(--neon-cyan)',
              }}
            >
              {tag}
            </span>
          ))}
        </div>
      </div>
    </header>
  )
}
