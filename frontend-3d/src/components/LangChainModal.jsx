import React, { useState } from 'react';
import { X, Sparkles, CheckCircle, AlertTriangle, MessageSquare, Send, Database, ArrowRight } from 'lucide-react';

export default function LangChainModal({ isOpen, onClose }) {
  if (!isOpen) return null;

  const [activeTab, setActiveTab] = useState('audit'); // 'audit' | 'chat' | 'rules'
  const [cleaningStatus, setCleaningStatus] = useState('ready'); // 'ready' | 'running' | 'done'
  const [qualityBefore] = useState(58.2);
  const [qualityAfter, setQualityAfter] = useState(58.2);
  const [userPrompt, setUserPrompt] = useState('');
  const [chatMessages, setChatMessages] = useState([
    {
      sender: 'assistant',
      text: '¡Hola! Soy el Asistente de Limpieza y Harmonización de Datos con LangChain para el Gemelo Digital de Salud Pública. ¿En qué puedo ayudarte con los microdatos de la ENAHO o SUSALUD?'
    }
  ]);

  const handleRunCleaning = () => {
    setCleaningStatus('running');
    setTimeout(() => {
      setQualityAfter(96.4);
      setCleaningStatus('done');
    }, 1200);
  };

  const handleSendMessage = () => {
    if (!userPrompt.trim()) return;
    const newMsg = { sender: 'user', text: userPrompt };
    setChatMessages((prev) => [...prev, newMsg]);
    const query = userPrompt.toLowerCase();
    setUserPrompt('');

    setTimeout(() => {
      let reply = '';
      if (query.includes('outlier') || query.includes('extremo')) {
        reply = 'LangChain recomienda winsorizar el Gasto de Bolsillo (OOPE) al 95° percentil y modelar residuos ortogonales con LightGBM para evitar que casos raros de hospitalización privada distorsionen la estimación del CATE.';
      } else if (query.includes('sisfoh') || query.includes('pobreza')) {
        reply = 'El score SISFOH se estandariza imputando valores faltantes mediante la mediana condicional del quintil de gasto y área geográfica (rural/urbana), preservando el umbral cuasiexperimental de elegibilidad al SIS.';
      } else if (query.includes('capacidad') || query.includes('subsistencia')) {
        reply = 'La Capacidad de Pago de la OMS se calcula como (Gasto Total - Gasto en Alimentos de Subsistencia). Si OOPE > 40% de este saldo, el hogar incurre en Gasto Catastrófico (CHE 40%).';
      } else {
        reply = `Para la consulta sobre "${query}", la cadena de LangChain ejecuta validaciones cruzadas entre el Módulo 400 (Salud) y Sumaria de la ENAHO garantizando consistencia de variables antes del entrenamiento.`;
      }
      setChatMessages((prev) => [...prev, { sender: 'assistant', text: reply }]);
    }, 600);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-md p-4 animate-fade-in">
      <div className="bg-slate-900 border border-purple-500/40 w-full max-w-3xl rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[85vh]">
        {/* Modal Header */}
        <div className="p-4 bg-gradient-to-r from-purple-950/60 via-slate-900 to-indigo-950/60 border-b border-slate-800 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-purple-500/20 text-purple-400 rounded-xl border border-purple-500/30">
              <Sparkles className="w-5 h-5 text-amber-300" />
            </div>
            <div>
              <h2 className="font-bold text-base text-slate-100 flex items-center space-x-2">
                <span>Asistente de Limpieza de Datos LangChain</span>
                <span className="text-[10px] bg-purple-500/20 text-purple-300 px-2 py-0.5 rounded-full border border-purple-500/30">
                  LCEL Pipeline
                </span>
              </h2>
              <p className="text-xs text-slate-400">Auditoría, validación y harmonización de microdatos ENAHO / SUSALUD</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Navigation Tabs */}
        <div className="flex border-b border-slate-800 bg-slate-950/60 text-xs font-semibold">
          <button
            onClick={() => setActiveTab('audit')}
            className={`flex-1 py-3 px-4 flex items-center justify-center space-x-2 border-b-2 transition-all ${
              activeTab === 'audit' ? 'border-purple-500 text-purple-400 bg-purple-500/10' : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            <Database className="w-4 h-4" />
            <span>Auditoría & Limpieza Automática</span>
          </button>
          <button
            onClick={() => setActiveTab('chat')}
            className={`flex-1 py-3 px-4 flex items-center justify-center space-x-2 border-b-2 transition-all ${
              activeTab === 'chat' ? 'border-purple-500 text-purple-400 bg-purple-500/10' : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            <MessageSquare className="w-4 h-4" />
            <span>Consultor Metodológico NLP</span>
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-6 overflow-y-auto flex-1 space-y-5 text-xs custom-scrollbar">
          {activeTab === 'audit' && (
            <div className="space-y-5">
              {/* Barra de Progreso de Calidad */}
              <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-3">
                <div className="flex justify-between items-center">
                  <span className="font-semibold text-slate-200">Índice Global de Calidad de Datos (ENAHO):</span>
                  <span className="font-mono font-bold text-sm text-purple-400">{qualityAfter}%</span>
                </div>
                <div className="w-full h-3 bg-slate-800 rounded-full overflow-hidden flex">
                  <div
                    className="bg-gradient-to-r from-amber-500 to-purple-500 h-full transition-all duration-700"
                    style={{ width: `${qualityAfter}%` }}
                  ></div>
                </div>
                <div className="flex justify-between text-[11px] text-slate-400">
                  <span>Datos Crudos: {qualityBefore}%</span>
                  <span className="text-emerald-400 font-semibold">Post-Pipeline LangChain: +38.2% incremento</span>
                </div>
              </div>

              {/* Botón para correr limpieza */}
              <div className="flex justify-center">
                <button
                  onClick={handleRunCleaning}
                  disabled={cleaningStatus === 'running'}
                  className="py-2.5 px-6 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 disabled:opacity-50 text-white rounded-xl font-bold flex items-center space-x-2 shadow-lg shadow-purple-600/30 transition-all text-xs"
                >
                  <Sparkles className="w-4 h-4 text-amber-300" />
                  <span>
                    {cleaningStatus === 'running'
                      ? 'Ejecutando Pipeline LangChain...'
                      : cleaningStatus === 'done'
                      ? '✓ Pipeline Ejecutado con Éxito'
                      : 'Ejecutar Cadena de Limpieza LangChain'}
                  </span>
                </button>
              </div>

              {/* Reporte de Diagnóstico */}
              <div className="bg-slate-950/80 p-4 rounded-xl border border-slate-800 space-y-2 text-slate-300">
                <h4 className="font-bold text-slate-100 flex items-center space-x-1.5">
                  <CheckCircle className="w-4 h-4 text-emerald-400" />
                  <span>Diagnóstico Estructurado de la Cadena:</span>
                </h4>
                <ul className="space-y-1.5 list-disc list-inside text-slate-400">
                  <li><strong className="text-slate-200">Winsorización de Outliers:</strong> Se recortaron picos superiores al 95° percentil en gasto mensual de bolsillo.</li>
                  <li><strong className="text-slate-200">Harmonización de Aseguramiento:</strong> Consolidación de 6 variantes textuales en categorías estándar (`SIS`, `EsSalud`, `Sin Seguro`).</li>
                  <li><strong className="text-slate-200">Subsistencia de Engel:</strong> Capacidad de pago ajustada descontando el 50% de gasto alimentario básico.</li>
                  <li><strong className="text-slate-200">Imputación SISFOH:</strong> Relleno de nulos usando mediana condicional por estrato geográfico.</li>
                </ul>
              </div>
            </div>
          )}

          {activeTab === 'chat' && (
            <div className="flex flex-col h-[380px] space-y-3">
              {/* Chat Container */}
              <div className="flex-1 bg-slate-950 p-4 rounded-xl border border-slate-800 overflow-y-auto space-y-3 custom-scrollbar">
                {chatMessages.map((msg, idx) => (
                  <div
                    key={idx}
                    className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div
                      className={`max-w-[80%] p-3 rounded-xl ${
                        msg.sender === 'user'
                          ? 'bg-purple-600 text-white font-medium'
                          : 'bg-slate-800 text-slate-200 border border-slate-700'
                      }`}
                    >
                      {msg.text}
                    </div>
                  </div>
                ))}
              </div>

              {/* Input Chat */}
              <div className="flex space-x-2">
                <input
                  type="text"
                  value={userPrompt}
                  onChange={(e) => setUserPrompt(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
                  placeholder="Pregunta a LangChain sobre outliers, SISFOH, ENAHO o CHE..."
                  className="flex-1 bg-slate-950 border border-slate-800 rounded-xl px-3.5 py-2 text-slate-200 focus:outline-none focus:border-purple-500 text-xs"
                />
                <button
                  onClick={handleSendMessage}
                  className="p-2.5 bg-purple-600 hover:bg-purple-500 text-white rounded-xl transition-all shadow-md shadow-purple-600/30"
                >
                  <Send className="w-4 h-4" />
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
