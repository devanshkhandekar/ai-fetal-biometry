import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { apiFetch } from '../api'

type Row = { id:string; patient_external_id:string; patient_name?:string; accession_number?:string; status:string; created_at:string }

export default function Dashboard(){
  const q = useQuery({ queryKey:['studies'], queryFn:()=>apiFetch<Row[]>('/v1/studies') })
  return <div>
    <div className="mb-6 flex items-center justify-between"><div><h1 className="text-2xl font-bold">Studies</h1><p className="text-slate-500">Local production-shaped demo</p></div><Link className="rounded bg-indigo-600 px-4 py-2 text-white" to="/studies/new">New Study</Link></div>
    <div className="overflow-hidden rounded-xl border bg-white">
      <table className="w-full text-sm"><thead className="bg-slate-50 text-left"><tr><th className="p-3">Patient</th><th>Accession</th><th>Status</th><th>Created</th><th></th></tr></thead>
      <tbody>{q.data?.map(s=><tr className="border-t" key={s.id}><td className="p-3"><div className="font-medium">{s.patient_name || 'Unnamed'}</div><div className="text-slate-500">{s.patient_external_id}</div></td><td>{s.accession_number||'—'}</td><td>{s.status}</td><td>{new Date(s.created_at).toLocaleString()}</td><td><Link className="text-indigo-600" to={`/studies/${s.id}`}>Open</Link></td></tr>)}</tbody></table>
      {!q.data?.length && <div className="p-8 text-center text-slate-500">No studies yet.</div>}
    </div>
  </div>
}
