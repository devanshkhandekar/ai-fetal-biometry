import { keycloak } from './auth'

export const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export async function apiFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  await keycloak.updateToken(30)
  const response = await fetch(`${API}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...(init.headers || {}),
      Authorization: `Bearer ${keycloak.token}`,
    },
  })
  if (!response.ok) {
    const text = await response.text()
    throw new Error(`${response.status} ${response.statusText}: ${text}`)
  }
  if (response.status === 204) return undefined as T
  return response.json()
}

export async function uploadFileToPresignedUrl(url: string, file: File, requiredHeaders: Record<string,string>) {
  const response = await fetch(url, { method: 'PUT', body: file, headers: requiredHeaders })
  if (!response.ok) throw new Error(`Upload failed: ${response.status} ${await response.text()}`)
}
