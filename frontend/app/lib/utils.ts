import { type ClassValue, clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/** Format a 0–1 probability as a percentage string. */
export function formatPct(p: number): string {
  return `${(p * 100).toFixed(1)}%`
}

/** Map prediction rank (0-indexed) to a neon colour. */
export function getNeonColor(rank: number): string {
  const palette = ['#00ffcc', '#39ff14', '#00bfff', '#ffcc00', '#ff007f']
  return palette[rank] ?? palette[palette.length - 1]
}

/**
 * Normalise a buffer in-place so each frame's features are in [0, 1].
 * This is a simple min-max normalisation across the entire buffer.
 * Used for display purposes only; the real normalisation happens server-side.
 */
export function normalizeBuffer(buf: number[][]): number[][] {
  if (buf.length === 0) return buf
  const flat = buf.flat()
  const min = Math.min(...flat)
  const max = Math.max(...flat)
  const range = max - min || 1
  return buf.map((frame) => frame.map((v) => (v - min) / range))
}
