import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import { apiFetch } from '../api'
import { useNavigate } from 'react-router-dom'

const schema=z.object({patient_external_id:z.string().min(1),patient_name:z.string().optional(),date_of_birth:z.string().optional(),accession_number:z.string().optional()})
type Form=z.infer<typeof schema>
export default function NewStudy(){
 const nav=useNavigate(); const {register,handleSubmit,formState:{errors,isSubmitting}}=useForm<Form>({resolver:zodResolver(schema)})
 const submit=async(data:Form)=>{const r=await apiFetch<{id:string}>('/v1/studies',{method:'POST',body:JSON.stringify(data)});nav(`/studies/${r.id}`)}
 return <div className="max-w-xl"><h1 className="mb-6 text-2xl font-bold">New Study</h1><form onSubmit={handleSubmit(submit)} className="space-y-4 rounded-xl border bg-white p-6">
  <label className="block"><span className="text-sm">Patient external ID *</span><input className="mt-1 w-full rounded border p-2" {...register('patient_external_id')}/>{errors.patient_external_id&&<span className="text-sm text-red-600">Required</span>}</label>
  <label className="block"><span className="text-sm">Patient name</span><input className="mt-1 w-full rounded border p-2" {...register('patient_name')}/></label>
  <label className="block"><span className="text-sm">Date of birth</span><input type="date" className="mt-1 w-full rounded border p-2" {...register('date_of_birth')}/></label>
  <label className="block"><span className="text-sm">Accession number</span><input className="mt-1 w-full rounded border p-2" {...register('accession_number')}/></label>
  <button disabled={isSubmitting} className="rounded bg-indigo-600 px-4 py-2 text-white disabled:opacity-50">Create Study</button>
 </form></div>
}
