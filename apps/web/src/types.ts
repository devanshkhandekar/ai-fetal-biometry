export type Measurement = { value: number; unit?: string; confidence?: number; status?: string }
export type Scan = {
  id: string
  anatomy: 'HEAD'|'ABDOMEN'|'FEMUR'
  filename: string
  status: string
  result?: {
    plane?: any
    quality?: any
    fusion?: any
    measurements?: Record<string, Measurement | number>
    artifact_urls?: Record<string,string>
    warnings?: string[]
  }
}
export type Study = {
  id: string
  patient: { external_id: string; display_name?: string; date_of_birth?: string }
  accession_number?: string
  status: string
  scans: Scan[]
  jobs: { id:string; scan_id:string; status:string; progress_step:string; error?:string }[]
}
