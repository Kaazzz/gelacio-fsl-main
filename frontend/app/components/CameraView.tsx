'use client'

import { useEffect, useRef, useCallback, useState } from 'react'
import { LandmarkBuffer } from '../lib/landmarkBuffer'
import { predictLandmarks, SignPrediction } from '../lib/api'

// ── Constants ────────────────────────────────────────────────────────────────
const SEQ_LEN = 60
const MIN_FRAMES = 10
const POLL_INTERVAL_MS = 500

// MediaPipe colours
const POSE_COLOR   = '#00ffcc'
const HAND_COLOR_L = '#39ff14'
const HAND_COLOR_R = '#ff007f'
const POINT_RADIUS = 3

// ── Types ────────────────────────────────────────────────────────────────────
interface Props {
  onPredictions: (preds: SignPrediction[]) => void
  onFpsUpdate:   (fps: number) => void
  onBufferFill:  (pct: number) => void   // 0–100
}

// ── Landmark extraction helpers ──────────────────────────────────────────────

/**
 * Training feature layout — 159 dims/frame (before velocity concat):
 *   right_hand : 21 × xyz = 63
 *   left_hand  : 21 × xyz = 63
 *   pose (11)  : 11 × xyz = 33
 *   Total = 159  →  ×2 after velocity concat in backend = 318
 *
 * Pose indices match POSE_IDXS_11 from the training notebook:
 *   0=nose, 11-16=shoulders/elbows/wrists, 23-26=hips/knees
 */
const POSE_IDXS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26]  // 11 landmarks

/** Extract [x, y, z] for every landmark in the list (zeros if absent). */
function extractXYZ(
  list: { x: number; y: number; z?: number }[] | undefined | null,
  count: number,
): number[] {
  const out: number[] = []
  for (let i = 0; i < count; i++) {
    const lm = list?.[i]
    out.push(lm?.x ?? 0, lm?.y ?? 0, lm?.z ?? 0)
  }
  return out
}

/** Extract [x, y, z] for a specific subset of pose landmark indices. */
function extractPoseSelected(
  list: { x: number; y: number; z?: number }[] | undefined | null,
  indices: number[],
): number[] {
  const out: number[] = []
  for (const idx of indices) {
    const lm = list?.[idx]
    out.push(lm?.x ?? 0, lm?.y ?? 0, lm?.z ?? 0)
  }
  return out
}

/**
 * Build the 159-dim feature vector matching the Python training extraction:
 *   right_hand(63) + left_hand(63) + pose_selected(33)
 * Returns null when no landmarks detected (frame marked invalid in buffer).
 */
function buildFeatureVector(results: any): number[] | null {
  if (!results) return null

  const rightHand = extractXYZ(results.rightHandLandmarks, 21)          // 63
  const leftHand  = extractXYZ(results.leftHandLandmarks,  21)          // 63
  const pose      = extractPoseSelected(results.poseLandmarks, POSE_IDXS) // 33

  const valid =
    results.poseLandmarks != null ||
    results.leftHandLandmarks != null ||
    results.rightHandLandmarks != null

  return valid ? [...rightHand, ...leftHand, ...pose] : null
}

// ── Drawing helpers ──────────────────────────────────────────────────────────

const POSE_CONNECTIONS = [
  [11,12],[11,13],[13,15],[12,14],[14,16],
  [11,23],[12,24],[23,24],[23,25],[24,26],[25,27],[26,28],
  [0,1],[1,2],[2,3],[3,7],[0,4],[4,5],[5,6],[6,8],
]

const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
  [5,9],[9,13],[13,17],
]

function drawLandmarks(
  ctx: CanvasRenderingContext2D,
  landmarks: { x: number; y: number }[] | null | undefined,
  connections: number[][],
  color: string,
  w: number,
  h: number,
) {
  if (!landmarks) return
  ctx.strokeStyle = color
  ctx.fillStyle   = color
  ctx.lineWidth   = 1.5

  for (const [a, b] of connections) {
    const la = landmarks[a]
    const lb = landmarks[b]
    if (!la || !lb) continue
    ctx.beginPath()
    ctx.moveTo(la.x * w, la.y * h)
    ctx.lineTo(lb.x * w, lb.y * h)
    ctx.stroke()
  }
  for (const lm of landmarks) {
    ctx.beginPath()
    ctx.arc(lm.x * w, lm.y * h, POINT_RADIUS, 0, 2 * Math.PI)
    ctx.fill()
  }
}

// ── Component ────────────────────────────────────────────────────────────────

export default function CameraView({ onPredictions, onFpsUpdate, onBufferFill }: Props) {
  const videoRef  = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const bufferRef = useRef(new LandmarkBuffer(SEQ_LEN))
  const pollRef   = useRef<ReturnType<typeof setInterval> | null>(null)
  const holisticRef = useRef<any>(null)
  const cameraRef   = useRef<any>(null)

  const fpsCountRef = useRef(0)
  const fpsTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const [cameraError, setCameraError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  // ── FPS counter ────────────────────────────────────────────────────────────
  useEffect(() => {
    fpsTimerRef.current = setInterval(() => {
      onFpsUpdate(fpsCountRef.current)
      fpsCountRef.current = 0
    }, 1000)
    return () => { if (fpsTimerRef.current) clearInterval(fpsTimerRef.current) }
  }, [onFpsUpdate])

  // ── Inference polling ──────────────────────────────────────────────────────
  const runInference = useCallback(async () => {
    const buf = bufferRef.current
    const validCount = buf.validFrameCount()
    onBufferFill(Math.round((buf.size() / SEQ_LEN) * 100))

    if (validCount < MIN_FRAMES) return

    const { sequence, mask } = buf.getSequence()
    try {
      const result = await predictLandmarks(sequence, mask)
      if (result?.top_predictions) {
        onPredictions(result.top_predictions)
      }
    } catch {
      // network errors are silent — connection status shown in StatusBar
    }
  }, [onPredictions, onBufferFill])

  // ── MediaPipe result handler ───────────────────────────────────────────────
  const onResults = useCallback((results: any) => {
    fpsCountRef.current++

    const canvas = canvasRef.current
    const video  = videoRef.current
    if (!canvas || !video) return

    const w = canvas.width  = video.videoWidth  || 640
    const h = canvas.height = video.videoHeight || 480

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Mirror: flip horizontally so it feels like a mirror
    ctx.save()
    ctx.scale(-1, 1)
    ctx.translate(-w, 0)

    // Draw video frame
    ctx.drawImage(results.image, 0, 0, w, h)

    // Darken slightly for contrast
    ctx.fillStyle = 'rgba(0,0,0,0.35)'
    ctx.fillRect(0, 0, w, h)

    // Draw skeleton overlays
    drawLandmarks(ctx, results.poseLandmarks,      POSE_CONNECTIONS,  POSE_COLOR,   w, h)
    drawLandmarks(ctx, results.leftHandLandmarks,  HAND_CONNECTIONS,  HAND_COLOR_L, w, h)
    drawLandmarks(ctx, results.rightHandLandmarks, HAND_CONNECTIONS,  HAND_COLOR_R, w, h)

    ctx.restore()

    // Extract features and push to buffer
    const frame = buildFeatureVector(results)
    bufferRef.current.push(frame)   // null → invalid frame (masked)
  }, [])

  // ── MediaPipe initialisation ───────────────────────────────────────────────
  useEffect(() => {
    let destroyed = false

    async function init() {
      try {
        // Dynamically import to avoid SSR issues
        const mp = await import('@mediapipe/holistic')
        const camUtils = await import('@mediapipe/camera_utils')

        if (destroyed) return

        const holistic = new mp.Holistic({
          locateFile: (file: string) =>
            `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`,
        })

        holistic.setOptions({
          modelComplexity: 1,
          smoothLandmarks: true,
          enableSegmentation: false,
          smoothSegmentation: false,
          refineFaceLandmarks: false,
          minDetectionConfidence: 0.5,
          minTrackingConfidence: 0.5,
        })

        holistic.onResults(onResults)
        holisticRef.current = holistic

        const video = videoRef.current!
        const camera = new camUtils.Camera(video, {
          onFrame: async () => {
            if (holisticRef.current) {
              await holisticRef.current.send({ image: video })
            }
          },
          width:  640,
          height: 480,
        })

        await camera.start()
        cameraRef.current = camera

        if (!destroyed) setLoading(false)
      } catch (err: any) {
        if (!destroyed) {
          setCameraError(err?.message ?? 'Camera or MediaPipe failed to load')
          setLoading(false)
        }
      }
    }

    init()

    // Start polling
    pollRef.current = setInterval(runInference, POLL_INTERVAL_MS)

    return () => {
      destroyed = true
      if (pollRef.current) clearInterval(pollRef.current)
      cameraRef.current?.stop?.()
      holisticRef.current?.close?.()
    }
  }, [onResults, runInference])

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center gap-2 mb-1">
        <span
          className="text-xs font-semibold uppercase tracking-widest"
          style={{ color: 'var(--neon-cyan)' }}
        >
          Camera Feed
        </span>
        {!loading && !cameraError && (
          <span
            className="inline-block w-2 h-2 rounded-full animate-pulse"
            style={{ background: 'var(--neon-green)' }}
          />
        )}
      </div>

      <div className="camera-container glow-cyan" style={{ border: '1px solid rgba(0,255,204,0.3)' }}>
        {/* Hidden video element — camera feed drawn onto canvas */}
        <video
          ref={videoRef}
          className="opacity-0"
          playsInline
          muted
          autoPlay
        />
        <canvas ref={canvasRef} />

        {/* Loading overlay */}
        {loading && (
          <div
            className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-sm"
            style={{ background: 'rgba(15,15,26,0.9)' }}
          >
            <div
              className="w-8 h-8 border-2 border-t-transparent rounded-full animate-spin"
              style={{ borderColor: 'var(--neon-cyan)', borderTopColor: 'transparent' }}
            />
            <span style={{ color: 'var(--text-muted)' }}>Loading MediaPipe…</span>
          </div>
        )}

        {/* Error overlay */}
        {cameraError && (
          <div
            className="absolute inset-0 flex flex-col items-center justify-center gap-2 p-4 text-center"
            style={{ background: 'rgba(15,15,26,0.95)' }}
          >
            <span className="text-2xl">📷</span>
            <p className="text-sm font-semibold" style={{ color: 'var(--neon-pink)' }}>
              Camera Error
            </p>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
              {cameraError}
            </p>
          </div>
        )}
      </div>

      <p className="text-xs text-center" style={{ color: 'var(--text-muted)' }}>
        Pose: <span style={{ color: POSE_COLOR }}>■</span>&nbsp;
        Left hand: <span style={{ color: HAND_COLOR_L }}>■</span>&nbsp;
        Right hand: <span style={{ color: HAND_COLOR_R }}>■</span>
      </p>
    </div>
  )
}
