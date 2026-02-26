/**
 * Ring-buffer for landmark frames.
 *
 * Holds the last `capacity` frames. Null frames (no person detected) are
 * stored as zero vectors and marked invalid in the mask.
 */
export class LandmarkBuffer {
  private capacity: number
  private frames: (number[] | null)[]   // null = invalid/empty frame
  private head: number                  // index of the next write position
  private count: number                 // number of frames pushed so far (≤ capacity)
  private featureDim: number            // inferred from first non-null frame

  constructor(capacity: number) {
    this.capacity   = capacity
    this.frames     = new Array(capacity).fill(null)
    this.head       = 0
    this.count      = 0
    this.featureDim = 0
  }

  /** Push a new frame. Pass null if no landmarks were detected this frame. */
  push(frame: number[] | null): void {
    if (frame !== null && this.featureDim === 0) {
      this.featureDim = frame.length
    }
    this.frames[this.head] = frame
    this.head  = (this.head + 1) % this.capacity
    this.count = Math.min(this.count + 1, this.capacity)
  }

  /** Number of frames currently stored (up to capacity). */
  size(): number {
    return this.count
  }

  /** Number of valid (non-null) frames. */
  validFrameCount(): number {
    return this.frames.filter((f) => f !== null).length
  }

  /**
   * Return a padded sequence of length `capacity` ready for the API.
   *
   * Frames are ordered oldest→newest.
   * Padding is prepended (at the start) using zero vectors with mask=0.
   */
  getSequence(): { sequence: number[][]; mask: number[] } {
    const dim = this.featureDim || 159   // fallback dim: 159 = rh(63)+lh(63)+pose(33)

    // Reconstruct in chronological order
    const ordered: (number[] | null)[] = []
    if (this.count < this.capacity) {
      // Buffer not yet full — frames are at indices 0..count-1 in insertion order
      for (let i = 0; i < this.count; i++) {
        ordered.push(this.frames[i])
      }
    } else {
      // Buffer full — oldest frame is at `head`
      for (let i = 0; i < this.capacity; i++) {
        ordered.push(this.frames[(this.head + i) % this.capacity])
      }
    }

    // Pad at the front to reach `capacity` length
    const padCount = this.capacity - ordered.length
    const zeros    = new Array(dim).fill(0)

    const sequence: number[][] = []
    const mask: number[]       = []

    for (let i = 0; i < padCount; i++) {
      sequence.push([...zeros])
      mask.push(0)
    }
    for (const frame of ordered) {
      sequence.push(frame !== null ? frame : [...zeros])
      mask.push(frame !== null ? 1 : 0)
    }

    return { sequence, mask }
  }

  /** Clear the buffer. */
  reset(): void {
    this.frames     = new Array(this.capacity).fill(null)
    this.head       = 0
    this.count      = 0
    this.featureDim = 0
  }
}
