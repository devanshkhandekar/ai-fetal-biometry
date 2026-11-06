import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import App from './App'
import { initAuth } from './auth'
import './index.css'
import './styles/phase5.css'

const queryClient = new QueryClient()

initAuth().then(() => {
  ReactDOM.createRoot(document.getElementById('root')!).render(
    <React.StrictMode>
      <QueryClientProvider client={queryClient}>
        <BrowserRouter><App/></BrowserRouter>
      </QueryClientProvider>
    </React.StrictMode>
  )
}).catch(err => {
  document.body.innerHTML = `<pre style="padding:20px">Authentication initialization failed: ${String(err)}</pre>`
})
