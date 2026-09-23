import React from 'react';
import { Activity, ShieldCheck, Database, ExternalLink, Cpu } from 'lucide-react';

export default function Navbar({ backendOnline, totalPopulation = 15000 }) {
  return (
    <header className="h-14 bg-slate-900/90 backdrop-blur-md border-b border-slate-800 px-6 flex items-center justify-between z-30 shadow-md">
      {/* Brand & Título */}
      <div className="flex items-center space-x-3">
        <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-cyan-500 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/25">
          <Activity className="w-4 h-4 text-white" />
        </div>
        <div>
          <h1 className="font-bold text-sm text-slate-100 flex items-center space-x-2">
            <span>Gemelo Digital 3D: Cobertura Sanitaria Universal</span>
            <span className="text-[10px] bg-cyan-500/20 text-cyan-400 px-2 py-0.5 rounded-full border border-cyan-500/30">
              Three.js + React
            </span>
          </h1>
          <p className="text-[10px] text-slate-400">Microsimulación Cuasiexperimental de Gasto Catastrófico (ENAHO / SUSALUD / OMS)</p>
        </div>
      </div>

      {/* Badges y Enlaces */}
      <div className="flex items-center space-x-4 text-xs">
        {/* Badge Estado Servidor */}
        <div className="flex items-center space-x-1.5 bg-slate-950 px-2.5 py-1 rounded-full border border-slate-800">
          <span className={`w-2 h-2 rounded-full ${backendOnline ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'}`}></span>
          <span className="text-slate-300 font-mono text-[11px]">
            {backendOnline ? 'FastAPI Backend 8000: Conectado' : 'Simulador Local Activo'}
          </span>
        </div>

        {/* Badge Agentes */}
        <div className="hidden md:flex items-center space-x-1.5 bg-slate-950 px-2.5 py-1 rounded-full border border-slate-800 text-slate-300 font-mono text-[11px]">
          <Database className="w-3.5 h-3.5 text-blue-400" />
          <span>{totalPopulation.toLocaleString()} Hogares / 24 Regiones</span>
        </div>

        {/* Botón Abrir Streamlit */}
        <a
          href="http://localhost:8501"
          target="_blank"
          rel="noreferrer"
          className="flex items-center space-x-1 px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg font-medium transition-colors text-xs border border-slate-700"
        >
          <span>Abrir Panel Streamlit</span>
          <ExternalLink className="w-3.5 h-3.5 ml-1 text-slate-400" />
        </a>
      </div>
    </header>
  );
}
