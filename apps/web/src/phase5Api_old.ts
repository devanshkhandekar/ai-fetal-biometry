import { apiFetch } from './api'

import type {
  ReportLanguage,
} from './lib/reportLanguages'


export async function generateReportNarrative(
  reportId: string,
  language: ReportLanguage,
): Promise<any> {

  const action =
    language === 'English'
      ? 'explain'
      : 'translate'

  return apiFetch(
    `/v1/reports/${reportId}/${action}`,
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

  if (!response) {
    return ''
  }

  if (typeof response === 'string') {
    return response
  }

  const candidates = [
    response.explanation,
    response.text,
    response.content,
    response.narrative,
    response.summary,
    response.result?.text,
    response.result?.explanation,
  ]

  for (
    const candidate
    of candidates
  ) {
    if (
      typeof candidate === 'string' &&
      candidate.trim()
    ) {
      return candidate
    }
  }

  /*
   * Some implementations persist the generated
   * explanation as an artifact and return only
   * metadata/download_url.
   */
  if (response.download_url) {
    return (
      'Explanation generated successfully. ' +
      'Use the generated explanation download ' +
      'link returned by the server.'
    )
  }

  return JSON.stringify(
    response,
    null,
    2,
  )
}
