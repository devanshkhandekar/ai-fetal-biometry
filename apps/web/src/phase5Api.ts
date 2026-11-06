import { apiFetch } from './api'

import type {
  ReportLanguage,
} from './lib/reportLanguages'


export async function generateReportNarrative(
  reportId: string,
  language: ReportLanguage,
): Promise<any> {
  return apiFetch(
    `/v1/reports/${reportId}/explain`,
    {
      method: 'POST',
      body: JSON.stringify({
        language,
      }),
    },
  )
}


export function extractNarrativeText(
  response: any,
): string {
  if (!response) return ''

  if (typeof response === 'string') {
    return response
  }

  const candidates = [
    response.narrative,
    response.explanation,
    response.text,
    response.content,
    response.summary,
    response.result?.text,
    response.result?.explanation,
  ]

  for (const candidate of candidates) {
    if (
      typeof candidate === 'string' &&
      candidate.trim()
    ) {
      return candidate
    }
  }

  if (response.download_url) {
    return (
      'Explanation generated successfully. ' +
      'A downloadable narrative artifact is available.'
    )
  }

  return JSON.stringify(
    response,
    null,
    2,
  )
}
