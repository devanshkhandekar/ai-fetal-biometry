import { Link, Route, Routes } from 'react-router-dom'
import { keycloak } from './auth'
import Dashboard from './pages/Dashboard'
import NewStudy from './pages/NewStudy'
import StudyPage from './pages/StudyPage'

export default function App() {
  const claims = keycloak.tokenParsed as any
  return <div>
    <header className="bg-slate-950 text-white">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
        <Link to="/" className="font-semibold">AI Fetal Biometry</Link>
        <div className="flex items-center gap-4 text-sm">
          <span>{claims?.email || claims?.preferred_username}</span>
          <button className="rounded bg-slate-800 px-3 py-1.5" onClick={() => keycloak.logout({redirectUri: window.location.origin})}>Logout</button>
        </div>
      </div>
    </header>
    <main className="mx-auto max-w-7xl p-6">
      <Routes>
        <Route path="/" element={<Dashboard/>}/>
        <Route path="/studies/new" element={<NewStudy/>}/>
        <Route path="/studies/:id" element={<StudyPage/>}/>
      </Routes>
    </main>
  </div>
}
