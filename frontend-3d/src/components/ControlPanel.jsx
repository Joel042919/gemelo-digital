import React from 'react';
import { Sliders, Shield, Zap, Sparkles, RefreshCw, Eye, Activity, Database } from 'lucide-react';

export default function ControlPanel({
  policyParams,
  onParamChange,
  cameraMode,
  setCameraMode,
  autoRotate,
  setAutoRotate,
  onOpenLangChain,
  onResetParams
}) {
  return (
    <aside className="w-80 bg-slate-900/90 backdrop-blur-xl border-r border-slate-800 flex flex-col h-full z-20 shadow-2xl">
      {/* Header Panel */}
      <div className="p-4 border-b border-slate-800/80 flex items-center justify-between">
        <div className="flex items-center space-x-2.5">
          <div className="p-2 bg-blue-500/20 text-blue-400 rounded-lg border border-blue-500/30">
            <Sliders className="w-5 h-5" />
          </div>
          <div>
            <h2 className="font-bold text-sm text-slate-100">Palancas de Política</h2>
            <p className="text-[11px] text-slate-400">Gemelo Digital Microsimulado</p>
          </div>
        </div>
        <button
          onClick={onResetParams}
          className="p-1.5 text-slate-400 hover:text-cyan-400 hover:bg-slate-800 rounded-lg transition-colors"
          title="Restablecer Parámetros"
        >
          <RefreshCw className="w-4 h-4" />
        </button>
      </div>

      {/* Sliders Container con scroll */}
      <div className="flex-1 overflow-y-auto p-4 space-y-5 text-xs text-slate-300 custom-scrollbar">
        {/* 1. Selección de Modelo Causal */}
        <div className="space-y-1.5">
          <label className="font-semibold text-slate-200 flex items-center justify-between">
            <span>Modelo Causal CATE</span>
            <span className="text-[10px] bg-emerald-500/20 text-emerald-400 px-1.5 py-0.5 rounded border border-emerald-500/30">
              Tuned ⭐
            </span>
          </label>
          <select
            value={policyParams.modelName}
            onChange={(e) => onParamChange('modelName', e.target.value)}
            className="w-full bg-slate-800/90 border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-100 focus:outline-none focus:border-cyan-500 text-xs"
          >
            <option value="Doubly Robust (IPW + Ridge AIPW)">⭐ Doubly Robust AIPW</option>
            <option value="Double Machine Learning (DML - LightGBM)">DML (LightGBM K-Fold)</option>
            <option value="X-Learner (Gradient Boosting Meta-Learner)">X-Learner (Gradient Boosting)</option>
          </select>
        </div>

        {/* 2. Expansión de Cobertura */}
        <div className="space-y-2 bg-slate-950/60 p-3 rounded-xl border border-slate-800">
          <div className="flex justify-between items-center">
            <span className="font-semibold text-slate-200">Expansión Cobertura SIS</span>
            <span className="text-cyan-400 font-mono font-bold">{policyParams.coverageExpansion}%</span>
          </div>
          <input
            type="range"
            min="0"
            max="100"
            step="5"
            value={policyParams.coverageExpansion}
            onChange={(e) => onParamChange('coverageExpansion', Number(e.target.value))}
            className="w-full accent-cyan-400 h-1.5 bg-slate-700 rounded-lg cursor-pointer"
          />
          <span className="text-[10px] text-slate-400 block">% de la población no asegurada a afiliar</span>
        </div>

        {/* 3. Estrategia de Focalización */}
        <div className="space-y-1.5">
          <label className="font-semibold text-slate-200">Estrategia de Focalización</label>
          <select
            value={policyParams.targetingStrategy}
            onChange={(e) => onParamChange('targetingStrategy', e.target.value)}
            className="w-full bg-slate-800/90 border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-100 focus:outline-none focus:border-cyan-500 text-xs"
          >
            <option value="sisfoh_pobreza">🎯 Progresiva SISFOH (Pobreza extrema)</option>
            <option value="regional_prioritaria">🗺️ Prioridad Sierra / Selva</option>
            <option value="cronicos_vulnerables">🩺 Carga de Enfermedad (Crónicos)</option>
            <option value="universal_aleatorio">🎲 Universal Aleatoria</option>
          </select>
        </div>

        {/* 4. Subsidio a Medicamentos */}
        <div className="space-y-2 bg-slate-950/60 p-3 rounded-xl border border-slate-800">
          <div className="flex justify-between items-center">
            <span className="font-semibold text-slate-200">Subsidio Medicamentos</span>
            <span className="text-purple-400 font-mono font-bold">{policyParams.medsSubsidy}%</span>
          </div>
          <input
            type="range"
            min="0"
            max="100"
            step="10"
            value={policyParams.medsSubsidy}
            onChange={(e) => onParamChange('medsSubsidy', Number(e.target.value))}
            className="w-full accent-purple-400 h-1.5 bg-slate-700 rounded-lg cursor-pointer"
          />
          <span className="text-[10px] text-slate-400 block">Reducción de copagos en farmacias</span>
        </div>

        {/* 5. Techo de Seguridad Catastrófica */}
        <div className="space-y-2 bg-slate-950/60 p-3 rounded-xl border border-slate-800">
          <div className="flex items-center justify-between">
            <span className="font-semibold text-slate-200 flex items-center space-x-1.5">
              <Shield className="w-3.5 h-3.5 text-emerald-400" />
              <span>Techo Catastrófico</span>
            </span>
            <input
              type="checkbox"
              checked={policyParams.catastrophicCapEnabled}
              onChange={(e) => onParamChange('catastrophicCapEnabled', e.target.checked)}
              className="w-4 h-4 rounded text-emerald-500 focus:ring-emerald-400 bg-slate-800 border-slate-700 cursor-pointer"
            />
          </div>
          {policyParams.catastrophicCapEnabled && (
            <div className="pt-2 space-y-1">
              <div className="flex justify-between text-[11px]">
                <span className="text-slate-400">Tope Capacidad de Pago:</span>
                <span className="font-mono font-bold text-emerald-400">{policyParams.catastrophicCapRatio}%</span>
              </div>
              <input
                type="range"
                min="10"
                max="50"
                step="5"
                value={policyParams.catastrophicCapRatio}
                onChange={(e) => onParamChange('catastrophicCapRatio', Number(e.target.value))}
                className="w-full accent-emerald-400 h-1.5 bg-slate-700 rounded-lg cursor-pointer"
              />
            </div>
          )}
        </div>

        {/* 6. Modos de Cámara 3D */}
        <div className="space-y-1.5">
          <label className="font-semibold text-slate-200 flex items-center space-x-1.5">
            <Eye className="w-3.5 h-3.5 text-blue-400" />
            <span>Perspectiva Cámara 3D</span>
          </label>
          <div className="grid grid-cols-2 gap-2">
            <button
              onClick={() => setCameraMode('cinematic')}
              className={`py-1.5 px-2 rounded-lg font-medium text-[11px] transition-all ${
                cameraMode === 'cinematic'
                  ? 'bg-blue-600 text-white shadow-md shadow-blue-500/25'
                  : 'bg-slate-800 hover:bg-slate-700 text-slate-300'
              }`}
            >
              Cinemática 3D
            </button>
            <button
              onClick={() => setCameraMode('topdown')}
              className={`py-1.5 px-2 rounded-lg font-medium text-[11px] transition-all ${
                cameraMode === 'topdown'
                  ? 'bg-blue-600 text-white shadow-md shadow-blue-500/25'
                  : 'bg-slate-800 hover:bg-slate-700 text-slate-300'
              }`}
            >
              Mapa Cenital
            </button>
          </div>
          <button
            onClick={() => setAutoRotate(!autoRotate)}
            className={`w-full mt-1.5 py-1.5 px-2 rounded-lg font-medium text-[11px] transition-all flex items-center justify-center space-x-1.5 ${
              autoRotate ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30' : 'bg-slate-800 text-slate-400'
            }`}
          >
            <Activity className="w-3.5 h-3.5" />
            <span>{autoRotate ? 'Auto-Rotación Activada' : 'Auto-Rotación Pausada'}</span>
          </button>
        </div>
      </div>

      {/* Botón LangChain Footer */}
      <div className="p-3 border-t border-slate-800/80 bg-slate-950/70">
        <button
          onClick={onOpenLangChain}
          className="w-full py-2 px-3 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white rounded-xl font-bold flex items-center justify-center space-x-2 shadow-lg shadow-purple-600/30 transition-all text-xs"
        >
          <Sparkles className="w-4 h-4 text-amber-300" />
          <span>Limpieza de Datos LangChain</span>
        </button>
      </div>
    </aside>
  );
}
