import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useParams } from 'react-router-dom'
import { apiFetch, uploadFileToPresignedUrl } from '../api'
import type { Study } from '../types'
import { useEffect, useState } from 'react'
import RulerCalibration from '../components/RulerCalibration'
import EfwMethodCard from '../components/EfwMethodCard'
const terminal = new Set(['COMPLETE', 'FAILED'])


import PatientExplanationPanel
  from '../components/PatientExplanationPanel'
import ReportLanguagePicker
  from '../components/ReportLanguagePicker'
import type { ReportLanguage }
  from '../lib/reportLanguages'

type Anatomy = 'HEAD' | 'ABDOMEN' | 'FEMUR'
type CalibrationSource = 'manual' | 'ruler' | 'dicom'

async function generateReportNarrative(
  reportId: string,
  language: ReportLanguage,
): Promise<any> {
  return apiFetch<any>(
    `/v1/reports/${reportId}/explain`,
    {
      method: 'POST',
      body: JSON.stringify({ language }),
    },
  )
}

function reportLanguageCode(
  language: ReportLanguage,
): string {
  const codes: Record<ReportLanguage, string> = {
    English: 'en',
    Hindi: 'hi',
    Gujarati: 'gu',
    Tamil: 'ta',
    Telugu: 'te',
    Arabic: 'ar',
    French: 'fr',
    Spanish: 'es',
  }

  return codes[language]
}

function extractNarrativeText(response: any): string {
  if (!response) return ''
  if (typeof response === 'string') return response

  const candidates = [
    response.explanation,
    response.text,
    response.content,
    response.narrative,
    response.summary,
    response.result?.text,
    response.result?.explanation,
  ]

  for (const candidate of candidates) {
    if (typeof candidate === 'string' && candidate.trim()) {
      return candidate
    }
  }

  if (response.download_url) {
    return 'Explanation generated successfully. A downloadable narrative artifact is available.'
  }

  return JSON.stringify(response, null, 2)
}

function MeasurementTable({ m }: { m: any }) {
  if (!m || Object.keys(m).length === 0) return null

  return (
    <table className="mt-3 w-full text-sm">
      <tbody>
        {Object.entries(m).map(([k, v]: any) => (
          <tr key={k} className="border-t">
            <td className="py-2 font-medium">{k}</td>
            <td>
              {typeof v === 'number' ? v : v.value}{' '}
              {typeof v === 'object' ? v.unit : ''}
            </td>
            <td>{typeof v === 'object' ? v.status : ''}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function EvidenceImage({ label, url }: { label: string; url?: string }) {
  if (!url) return null

  return (
    <figure>
      <div className="mb-1 text-xs font-medium text-slate-500">{label}</div>
      <img
        className="w-full rounded border bg-black object-contain"
        style={{ maxHeight: 320 }}
        src={url}
        alt={label}
      />
    </figure>
  )
}

function VisualEvidence({ scan }: { scan: any }) {
  const overlay = scan.result?.artifact_urls?.overlay
  const mask = scan.result?.artifact_urls?.mask
  const original = scan.original_url

  if (!original && !overlay && !mask) return null

  return (
    <div className="mt-4">
      <div className="mb-2 text-sm font-semibold">Visual evidence</div>

      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
        <EvidenceImage label="Original" url={original} />
        <EvidenceImage label="AI overlay" url={overlay} />
        <EvidenceImage label="Predicted mask" url={mask} />
      </div>

      {!overlay && (
        <div className="mt-2 rounded bg-slate-50 p-2 text-xs text-slate-500">
          No derived overlay was produced by the current{' '}
          {scan.anatomy.toLowerCase()} model adapter for this run.
        </div>
      )}
    </div>
  )
}

export default function StudyPage() {
  const { id } = useParams()
  const qc = useQueryClient()
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string>()
  const [explanation, setExplanation] = useState<string | null>(null)
  const [reportLanguage, setReportLanguage] =
    useState<ReportLanguage>('English')

  const study = useQuery({
    queryKey: ['study', id],
    queryFn: () => apiFetch<Study>(`/v1/studies/${id}`),
    refetchInterval: (q) => {
      const d = q.state.data as Study | undefined
      return d?.jobs?.some((j) => !terminal.has(j.status)) ? 2000 : false
    },
  })

  const results = useQuery({
    queryKey: ['study-results', id],
    queryFn: () => apiFetch<any>(`/v1/studies/${id}/results`),
    refetchInterval: () =>
      study.data?.jobs?.some((j) => !terminal.has(j.status)) ? 2000 : false,
  })

  const reports = useQuery({
    queryKey: ['reports', id],
    queryFn: () => apiFetch<any[]>(`/v1/studies/${id}/reports`),
    refetchInterval: 3000,
  })

  const completedReport =
    reports.data
      ?.filter(
        (r: any) =>
          r.status === 'COMPLETE'
      )
      ?.sort(
        (a: any, b: any) =>
          String(
            b.completed_at ||
            b.created_at ||
            ''
          ).localeCompare(
            String(
              a.completed_at ||
              a.created_at ||
              ''
            )
          )
      )?.[0]



  const start = useMutation({
    mutationFn: () =>
      apiFetch(`/v1/studies/${id}/inference`, {
        method: 'POST',
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['study', id] })
      qc.invalidateQueries({ queryKey: ['study-results', id] })
    },
  })

  const generateReport = useMutation({
    mutationFn: () =>
      apiFetch(`/v1/studies/${id}/reports`, {
        method: 'POST',
        body: JSON.stringify({
          report_type: 'DOCTOR',
          language: reportLanguageCode(reportLanguage),
        }),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['reports', id] }),
  })

  async function upload(
    anatomy: Anatomy,
    file: File,
    sx?: string,
    sy?: string,
    roi?: any,
  ) {
    setBusy(true)
    setError(undefined)

    try {
      /*
       * IMPORTANT:
       * Keep the existing Phase-4 upload contract unchanged.
       * Ruler calibration only populates sx/sy in the UI; the backend still
       * receives pixel_spacing_x / pixel_spacing_y exactly as before.
       */
      const init = await apiFetch<any>(`/v1/studies/${id}/uploads/initiate`, {
        method: 'POST',
        body: JSON.stringify({
          anatomy,
          filename: file.name,
          content_type: file.type || 'application/octet-stream',
          pixel_spacing_x: sx ? Number(sx) : null,
          pixel_spacing_y: sy ? Number(sy) : null,
          roi: roi || null,
        }),
      })

      await uploadFileToPresignedUrl(
        init.upload_url,
        file,
        init.required_headers,
      )

      await apiFetch(`/v1/studies/${id}/uploads/complete`, {
        method: 'POST',
        body: JSON.stringify({ scan_id: init.scan_id }),
      })

      await qc.invalidateQueries({ queryKey: ['study', id] })
      await qc.invalidateQueries({ queryKey: ['study-results', id] })
    } catch (e: any) {
      setError(String(e))
    } finally {
      setBusy(false)
    }
  }

  if (study.isLoading) return <p>Loading…</p>
  if (!study.data) return <p>Study not found.</p>

  const s = study.data
  const aggregate = results.data?.study_aggregate
  const displayScans = results.data?.scans || s.scans

  const allMeasurements =
  results.data?.measurements || {}

  const measurementValue = (
    name: string,
  ): number | null => {
    const item =
      allMeasurements?.[name]

    if (typeof item === 'number') {
      return item
    }

    if (
      item &&
      typeof item === 'object' &&
      typeof item.value === 'number'
    ) {
      return item.value
    }

    return null
  }

  const efwMeasurement =
    allMeasurements?.efw_g

  const efwValue =
    measurementValue('efw_g')

  const hcValue =
    measurementValue('hc_mm')

  const acValue =
    measurementValue('ac_mm')

  const flValue =
    measurementValue('fl_mm')

  const efwStatus =
    typeof efwMeasurement === 'object'
      ? efwMeasurement?.status
      : aggregate?.status

  const efwFormula =
    aggregate?.result?.provenance?.formula

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Study {s.id.slice(0, 8)}</h1>
        <p className="text-slate-500">
          Patient: {s.patient.display_name || s.patient.external_id} · Status:{' '}
          {s.status}
        </p>
      </div>

      {error && (
        <div className="rounded bg-red-50 p-3 text-red-700">{error}</div>
      )}

      <div className="grid gap-4 md:grid-cols-3">
        {(['HEAD', 'ABDOMEN', 'FEMUR'] as const).map((a) => (
          <UploadCard
            key={a}
            anatomy={a}
            disabled={busy}
            onUpload={upload}
          />
        ))}
      </div>

      <div className="flex flex-col gap-3 lg:flex-row lg:items-end">
        <button
          onClick={() => start.mutate()}
          disabled={start.isPending || busy}
          className="rounded bg-emerald-600 px-4 py-2 text-white disabled:opacity-50"
        >
          Start Analysis
        </button>

        <div className="min-w-[220px]">
          <ReportLanguagePicker
            value={reportLanguage}
            onChange={setReportLanguage}
            disabled={generateReport.isPending}
            label="PDF language"
          />
        </div>

        <button
          onClick={() => generateReport.mutate()}
          disabled={generateReport.isPending}
          className="rounded bg-slate-900 px-4 py-2 text-white disabled:opacity-50"
        >
          {generateReport.isPending
            ? 'Generating PDF…'
            : `Generate Doctor PDF (${reportLanguage})`}
        </button>
      </div>

      {s.jobs.length > 0 && (
        <div className="rounded-xl border bg-white p-4">
          <h2 className="font-semibold">Processing</h2>

          {s.jobs.map((j) => (
            <div
              key={j.id}
              className="mt-2 flex justify-between border-t pt-2 text-sm"
            >
              <span>
                {s.scans.find((x) => x.id === j.scan_id)?.anatomy || j.scan_id}
              </span>

              <span>
                {j.progress_step} · {j.status}
                {j.error ? ` · ${j.error}` : ''}
              </span>
            </div>
          ))}
        </div>
      )}

      {results.data && (
        <div className="rounded-xl border bg-white p-4">
          <div className="flex items-center justify-between">
            <h2 className="font-semibold">Overall Study Measurements</h2>
            <span className="text-sm text-slate-500">
              Aggregation: {aggregate?.status || '—'}
            </span>
          </div>

          <MeasurementTable m={results.data.measurements} />

          {efwValue != null && (
            <div className="mt-4">
              <EfwMethodCard
                value={efwValue}
                status={efwStatus}
                hcMm={hcValue}
                acMm={acValue}
                flMm={flValue}
                calibrationSource="Per-scan calibrated"
                formula={
                  efwFormula ||
                  'Hadlock HC + AC + FL'
                }
              />
            </div>
          )}

          {aggregate?.error && (
            <div className="mt-3 rounded bg-amber-50 p-2 text-sm text-amber-800">
              {aggregate.error}
            </div>
          )}
        </div>
      )}

      <div className="space-y-4">
        {displayScans.map((scan: any) => (
          <div key={scan.id} className="rounded-xl border bg-white p-4">
            <div className="flex justify-between">
              <h2 className="font-semibold">{scan.anatomy}</h2>
              <span className="text-sm text-slate-500">{scan.status}</span>
            </div>

            <p className="text-xs text-slate-500">{scan.filename}</p>

            {scan.result && (
              <>
                <div className="mt-3 grid gap-2 text-sm md:grid-cols-3">
                  <div className="rounded bg-slate-50 p-2">
                    Plane
                    <br />
                    <b>{scan.result.plane?.label || '—'}</b>
                  </div>

                  <div className="rounded bg-slate-50 p-2">
                    Quality
                    <br />
                    <b>{scan.result.quality?.label || '—'}</b>
                  </div>

                  <div className="rounded bg-slate-50 p-2">
                    Fusion
                    <br />
                    <b>{scan.result.fusion?.status || '—'}</b>
                  </div>
                </div>

                <MeasurementTable m={scan.result.measurements} />
                <VisualEvidence scan={scan} />

                {scan.result.warnings?.map((w: string) => (
                  <div
                    key={w}
                    className="mt-2 rounded bg-amber-50 p-2 text-xs text-amber-800"
                  >
                    {w}
                  </div>
                ))}
              </>
            )}
          </div>
        ))}
      </div>

      <div className="rounded-xl border bg-white p-4">
        <h2 className="font-semibold">Reports</h2>

        {reports.data?.map((r) => (
          <div
            key={r.id}
            className="mt-2 flex items-center justify-between border-t pt-2 text-sm"
          >
            <span>
              {r.type} · {r.language || '—'} · {r.status}
            </span>

            {r.download_url ? (
              <a
                className="text-indigo-600"
                href={r.download_url}
                target="_blank"
                rel="noreferrer"
              >
                Download PDF
              </a>
            ) : (
              <span>{r.error || 'Processing…'}</span>
            )}
          </div>
        ))}
      </div>

      {completedReport && (
        <PatientExplanationPanel
          explanation={explanation}
          onGenerate={async (language) => {
            const response = await generateReportNarrative(
              completedReport.id,
              language,
            )

            setExplanation(
              extractNarrativeText(response),
            )
          }}
        />
      )}
    </div>
  )
}

function isDicomFile(file: File): boolean {
  const name = file.name.toLowerCase()
  const type = (file.type || '').toLowerCase()

  return (
    name.endsWith('.dcm') ||
    name.endsWith('.dicom') ||
    type.includes('dicom')
  )
}

function canPreviewAsImage(file: File): boolean {
  const name = file.name.toLowerCase()
  const type = (file.type || '').toLowerCase()

  return (
    type.startsWith('image/') ||
    name.endsWith('.png') ||
    name.endsWith('.jpg') ||
    name.endsWith('.jpeg')
  )
}

function UploadCard({
  anatomy,
  disabled,
  onUpload,
}: {
  anatomy: Anatomy
  disabled: boolean
  onUpload: (
    a: Anatomy,
    f: File,
    sx?: string,
    sy?: string,
    roi?: any,
  ) => void
}) {
  const [file, setFile] = useState<File>()

  // Preserve the original Phase-4 spacing fields/state.
  const [sx, setSx] = useState('')
  const [sy, setSy] = useState('')

  const [calibrationSource, setCalibrationSource] =
    useState<CalibrationSource>('manual')

  const [showRuler, setShowRuler] = useState(false)
  const [previewUrl, setPreviewUrl] = useState<string>()

  const [rx, setRx] = useState('')
  const [ry, setRy] = useState('')
  const [rw, setRw] = useState('')
  const [rh, setRh] = useState('')

  useEffect(() => {
    if (!file || !canPreviewAsImage(file)) {
      setPreviewUrl(undefined)
      return
    }

    const url = URL.createObjectURL(file)
    setPreviewUrl(url)

    return () => {
      URL.revokeObjectURL(url)
    }
  }, [file])

  const roi =
    rx !== '' && ry !== '' && rw !== '' && rh !== ''
      ? {
          x: Number(rx),
          y: Number(ry),
          width: Number(rw),
          height: Number(rh),
        }
      : undefined

  function onFileSelected(nextFile?: File) {
    setFile(nextFile)
    setShowRuler(false)

    // Avoid accidentally reusing calibration/ROI from a previous file.
    setSx('')
    setSy('')
    setRx('')
    setRy('')
    setRw('')
    setRh('')

    if (nextFile && isDicomFile(nextFile)) {
      // No frontend DICOM parser is introduced here. Leaving spacing blank
      // preserves the current backend/model DICOM calibration path if available.
      setCalibrationSource('dicom')
    } else {
      setCalibrationSource('manual')
    }
  }

  function setManualSpacingX(value: string) {
    setSx(value)
    setCalibrationSource('manual')
  }

  function setManualSpacingY(value: string) {
    setSy(value)
    setCalibrationSource('manual')
  }

  const hasAppliedSpacing = sx !== '' && sy !== ''

  return (
    <div className="rounded-xl border bg-white p-4">
      <h2 className="font-semibold">{anatomy}</h2>

      <input
        className="mt-3 block w-full text-sm"
        type="file"
        accept=".dcm,.dicom,.png,.jpg,.jpeg"
        onChange={(e) => onFileSelected(e.target.files?.[0])}
      />

      <div className="mt-3 rounded-lg bg-slate-50 p-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <div className="text-xs font-medium uppercase tracking-wide text-slate-500">
              Physical calibration
            </div>
            <div className="mt-1 text-sm font-semibold text-slate-800">
              Source: {calibrationSource.toUpperCase()}
            </div>
          </div>

          {previewUrl && (
            <button
              type="button"
              onClick={() => setShowRuler((v) => !v)}
              className="rounded border border-slate-300 bg-white px-3 py-1.5 text-xs font-medium text-slate-700"
            >
              {showRuler ? 'Hide image ruler' : 'Use image ruler'}
            </button>
          )}
        </div>

        {calibrationSource === 'dicom' && !hasAppliedSpacing && (
          <p className="mt-2 text-xs text-slate-500">
            DICOM selected. Manual spacing may be left blank only when the
            existing backend/model DICOM loader provides trusted physical
            calibration. Otherwise enter spacing manually.
          </p>
        )}

        {hasAppliedSpacing && (
          <p className="mt-2 text-xs text-slate-600">
            Applied spacing: X {sx} mm/px · Y {sy} mm/px
          </p>
        )}
      </div>

      {/*
        Preserve the original Phase-4 manual spacing workflow.
        Ruler calibration writes back into these exact same fields.
      */}
      <div className="mt-3 grid grid-cols-2 gap-2">
        <input
          value={sx}
          onChange={(e) => setManualSpacingX(e.target.value)}
          className="rounded border p-2 text-sm"
          placeholder="Spacing X mm/px"
          inputMode="decimal"
        />

        <input
          value={sy}
          onChange={(e) => setManualSpacingY(e.target.value)}
          className="rounded border p-2 text-sm"
          placeholder="Spacing Y mm/px"
          inputMode="decimal"
        />
      </div>

      {showRuler && previewUrl && (
        <div className="mt-3">
          <RulerCalibration
            imageUrl={previewUrl}
            onApply={(spacing) => {
              // Existing API convention:
              // pixel_spacing_x = horizontal/column spacing
              // pixel_spacing_y = vertical/row spacing
              setSx(String(spacing.column_mm))
              setSy(String(spacing.row_mm))
              setCalibrationSource('ruler')
            }}
          />
        </div>
      )}

      {!previewUrl && file && !isDicomFile(file) && (
        <div className="mt-3 rounded bg-amber-50 p-2 text-xs text-amber-800">
          Browser preview is unavailable for this file, so image-ruler
          calibration cannot be used. Enter trusted spacing manually.
        </div>
      )}

      <div className="mt-2 grid grid-cols-4 gap-1">
        <input
          value={rx}
          onChange={(e) => setRx(e.target.value)}
          className="rounded border p-2 text-xs"
          placeholder="ROI x"
          inputMode="decimal"
        />

        <input
          value={ry}
          onChange={(e) => setRy(e.target.value)}
          className="rounded border p-2 text-xs"
          placeholder="ROI y"
          inputMode="decimal"
        />

        <input
          value={rw}
          onChange={(e) => setRw(e.target.value)}
          className="rounded border p-2 text-xs"
          placeholder="ROI w"
          inputMode="decimal"
        />

        <input
          value={rh}
          onChange={(e) => setRh(e.target.value)}
          className="rounded border p-2 text-xs"
          placeholder="ROI h"
          inputMode="decimal"
        />
      </div>

      <p className="mt-1 text-xs text-slate-400">
        For abdomen/femur, provide a trusted ROI to avoid the unvalidated
        full-image fallback. Leave blank only for deliberate REVIEW-mode testing.
      </p>

      <button
        disabled={!file || disabled}
        onClick={() => file && onUpload(anatomy, file, sx, sy, roi)}
        className="mt-3 w-full rounded bg-indigo-600 px-3 py-2 text-sm text-white disabled:opacity-50"
      >
        Upload {anatomy}
      </button>
    </div>
  )
}
