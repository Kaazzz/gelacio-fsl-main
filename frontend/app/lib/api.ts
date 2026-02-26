import axios from 'axios'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: { 'Content-Type': 'application/json' },
  timeout: 5000,
})

export interface SignPrediction {
  sign: string
  probability: number
}

export interface PredictResponse {
  top_predictions: SignPrediction[]
  top_sign: string
  top_probability: number
  num_classes: number
  mock?: boolean
}

/**
 * Send a landmark buffer to the backend for sign language inference.
 *
 * @param landmarks  (T × F) array — T frames of F landmark features
 * @param mask       (T,) array — 1.0 for valid frames, 0.0 for padding
 */
export async function predictLandmarks(
  landmarks: number[][],
  mask: number[],
): Promise<PredictResponse> {
  const response = await api.post<PredictResponse>('/api/predict-landmarks', {
    landmarks,
    mask,
  })
  return response.data
}

export async function healthCheck() {
  const response = await api.get('/api/health')
  return response.data
}

export default api
