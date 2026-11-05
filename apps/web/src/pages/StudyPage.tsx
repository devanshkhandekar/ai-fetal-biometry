import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useParams } from 'react-router-dom'
import { apiFetch, uploadFileToPresignedUrl } from '../api'
import type { Study } from '../types'
import { useState } from 'react'

const terminal = new Set(['COMPLETE','FAILED'])

function MeasurementTable({m}:{m:any}){
  if(!m || Object.keys(m).length===0) return null
  return <table className="mt-3 w-full text-sm"><tbody>{Object.entries(m).map(([k,v]:any)=><tr key={k} className="border-t"><td className="py-2 font-medium">{k}</td><td>{typeof v==='number'?v:v.value} {typeof v==='object'?v.unit:''}</td><td>{typeof v==='object'?v.status:''}</td></tr>)}</tbody></table>
}

function EvidenceImage({label,url}:{label:string;url?:string}){
  if(!url) return null
  return <figure><div className="mb-1 text-xs font-medium text-slate-500">{label}</div><img className="w-full rounded border bg-black object-contain" style={{maxHeight:320}} src={url} alt={label}/></figure>
}

function VisualEvidence({scan}:{scan:any}){
  const overlay=scan.result?.artifact_urls?.overlay
  const mask=scan.result?.artifact_urls?.mask
  const original=scan.original_url
  if(!original && !overlay && !mask) return null
  return <div className="mt-4">
    <div className="mb-2 text-sm font-semibold">Visual evidence</div>
    <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
      <EvidenceImage label="Original" url={original}/>
      <EvidenceImage label="AI overlay" url={overlay}/>
      <EvidenceImage label="Predicted mask" url={mask}/>
    </div>
    {!overlay && <div className="mt-2 rounded bg-slate-50 p-2 text-xs text-slate-500">No derived overlay was produced by the current {scan.anatomy.toLowerCase()} model adapter for this run.</div>}
  </div>
}

export default function StudyPage(){
 const {id}=useParams(); const qc=useQueryClient(); const [busy,setBusy]=useState(false); const [error,setError]=useState<string>()
 const study=useQuery({queryKey:['study',id],queryFn:()=>apiFetch<Study>(`/v1/studies/${id}`),refetchInterval:(q)=>{const d=q.state.data as Study|undefined;return d?.jobs?.some(j=>!terminal.has(j.status))?2000:false}})
 const results=useQuery({queryKey:['study-results',id],queryFn:()=>apiFetch<any>(`/v1/studies/${id}/results`),refetchInterval:()=>study.data?.jobs?.some(j=>!terminal.has(j.status))?2000:false})
 const reports=useQuery({queryKey:['reports',id],queryFn:()=>apiFetch<any[]>(`/v1/studies/${id}/reports`),refetchInterval:3000})
 const start=useMutation({mutationFn:()=>apiFetch(`/v1/studies/${id}/inference`,{method:'POST'}),onSuccess:()=>{qc.invalidateQueries({queryKey:['study',id]});qc.invalidateQueries({queryKey:['study-results',id]})}})
 const generateReport=useMutation({mutationFn:()=>apiFetch(`/v1/studies/${id}/reports`,{method:'POST',body:JSON.stringify({report_type:'DOCTOR',language:'en'})}),onSuccess:()=>qc.invalidateQueries({queryKey:['reports',id]})})
 async function upload(anatomy:'HEAD'|'ABDOMEN'|'FEMUR', file:File, sx?:string, sy?:string, roi?:any){setBusy(true);setError(undefined);try{const init=await apiFetch<any>(`/v1/studies/${id}/uploads/initiate`,{method:'POST',body:JSON.stringify({anatomy,filename:file.name,content_type:file.type||'application/octet-stream',pixel_spacing_x:sx?Number(sx):null,pixel_spacing_y:sy?Number(sy):null,roi:roi||null})});await uploadFileToPresignedUrl(init.upload_url,file,init.required_headers);await apiFetch(`/v1/studies/${id}/uploads/complete`,{method:'POST',body:JSON.stringify({scan_id:init.scan_id})});await qc.invalidateQueries({queryKey:['study',id]});await qc.invalidateQueries({queryKey:['study-results',id]})}catch(e:any){setError(String(e))}finally{setBusy(false)}}
 if(study.isLoading) return <p>Loading…</p>; if(!study.data) return <p>Study not found.</p>
 const s=study.data
 const aggregate=results.data?.study_aggregate
 const displayScans=results.data?.scans || s.scans
 return <div className="space-y-6"><div><h1 className="text-2xl font-bold">Study {s.id.slice(0,8)}</h1><p className="text-slate-500">Patient: {s.patient.display_name||s.patient.external_id} · Status: {s.status}</p></div>
  {error&&<div className="rounded bg-red-50 p-3 text-red-700">{error}</div>}
  <div className="grid gap-4 md:grid-cols-3">{(['HEAD','ABDOMEN','FEMUR'] as const).map(a=><UploadCard key={a} anatomy={a} disabled={busy} onUpload={upload}/>)}</div>
  <div className="flex gap-3"><button onClick={()=>start.mutate()} disabled={start.isPending||busy} className="rounded bg-emerald-600 px-4 py-2 text-white disabled:opacity-50">Start Analysis</button><button onClick={()=>generateReport.mutate()} disabled={generateReport.isPending} className="rounded bg-slate-900 px-4 py-2 text-white">Generate Doctor PDF</button></div>
  {s.jobs.length>0&&<div className="rounded-xl border bg-white p-4"><h2 className="font-semibold">Processing</h2>{s.jobs.map(j=><div key={j.id} className="mt-2 flex justify-between border-t pt-2 text-sm"><span>{s.scans.find(x=>x.id===j.scan_id)?.anatomy||j.scan_id}</span><span>{j.progress_step} · {j.status}{j.error?` · ${j.error}`:''}</span></div>)}</div>}
  {results.data&&<div className="rounded-xl border bg-white p-4"><div className="flex items-center justify-between"><h2 className="font-semibold">Overall Study Measurements</h2><span className="text-sm text-slate-500">Aggregation: {aggregate?.status||'—'}</span></div><MeasurementTable m={results.data.measurements}/>{aggregate?.error&&<div className="mt-3 rounded bg-amber-50 p-2 text-sm text-amber-800">{aggregate.error}</div>}</div>}
  <div className="space-y-4">{displayScans.map((scan:any)=><div key={scan.id} className="rounded-xl border bg-white p-4"><div className="flex justify-between"><h2 className="font-semibold">{scan.anatomy}</h2><span className="text-sm text-slate-500">{scan.status}</span></div><p className="text-xs text-slate-500">{scan.filename}</p>{scan.result&&<><div className="mt-3 grid gap-2 text-sm md:grid-cols-3"><div className="rounded bg-slate-50 p-2">Plane<br/><b>{scan.result.plane?.label||'—'}</b></div><div className="rounded bg-slate-50 p-2">Quality<br/><b>{scan.result.quality?.label||'—'}</b></div><div className="rounded bg-slate-50 p-2">Fusion<br/><b>{scan.result.fusion?.status||'—'}</b></div></div><MeasurementTable m={scan.result.measurements}/><VisualEvidence scan={scan}/>{scan.result.warnings?.map((w:string)=><div key={w} className="mt-2 rounded bg-amber-50 p-2 text-xs text-amber-800">{w}</div>)}</>}</div>)}</div>
  <div className="rounded-xl border bg-white p-4"><h2 className="font-semibold">Reports</h2>{reports.data?.map(r=><div key={r.id} className="mt-2 flex items-center justify-between border-t pt-2 text-sm"><span>{r.type} · {r.status}</span>{r.download_url?<a className="text-indigo-600" href={r.download_url} target="_blank">Download PDF</a>:<span>{r.error||'Processing…'}</span>}</div>)}</div>
 </div>
}

function UploadCard({anatomy,disabled,onUpload}:{anatomy:'HEAD'|'ABDOMEN'|'FEMUR';disabled:boolean;onUpload:(a:any,f:File,sx?:string,sy?:string,roi?:any)=>void}){
 const [file,setFile]=useState<File>(); const [sx,setSx]=useState(''); const [sy,setSy]=useState('')
 const [rx,setRx]=useState(''); const [ry,setRy]=useState(''); const [rw,setRw]=useState(''); const [rh,setRh]=useState('')
 const roi=(rx!==''&&ry!==''&&rw!==''&&rh!=='')?{x:Number(rx),y:Number(ry),width:Number(rw),height:Number(rh)}:undefined
 return <div className="rounded-xl border bg-white p-4"><h2 className="font-semibold">{anatomy}</h2><input className="mt-3 block w-full text-sm" type="file" accept=".dcm,.dicom,.png,.jpg,.jpeg" onChange={e=>setFile(e.target.files?.[0])}/><div className="mt-3 grid grid-cols-2 gap-2"><input value={sx} onChange={e=>setSx(e.target.value)} className="rounded border p-2 text-sm" placeholder="Spacing X mm/px"/><input value={sy} onChange={e=>setSy(e.target.value)} className="rounded border p-2 text-sm" placeholder="Spacing Y mm/px"/></div><div className="mt-2 grid grid-cols-4 gap-1"><input value={rx} onChange={e=>setRx(e.target.value)} className="rounded border p-2 text-xs" placeholder="ROI x"/><input value={ry} onChange={e=>setRy(e.target.value)} className="rounded border p-2 text-xs" placeholder="ROI y"/><input value={rw} onChange={e=>setRw(e.target.value)} className="rounded border p-2 text-xs" placeholder="ROI w"/><input value={rh} onChange={e=>setRh(e.target.value)} className="rounded border p-2 text-xs" placeholder="ROI h"/></div><p className="mt-1 text-xs text-slate-400">For abdomen/femur, provide a trusted ROI to avoid the unvalidated full-image fallback. Leave blank only for deliberate REVIEW-mode testing.</p><button disabled={!file||disabled} onClick={()=>file&&onUpload(anatomy,file,sx,sy,roi)} className="mt-3 w-full rounded bg-indigo-600 px-3 py-2 text-sm text-white disabled:opacity-50">Upload {anatomy}</button></div>
}
