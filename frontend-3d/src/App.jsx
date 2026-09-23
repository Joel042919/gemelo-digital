import React, { useState, useEffect, useCallback } from 'react';
import Navbar from './components/Navbar';
import ControlPanel from './components/ControlPanel';
import AnalyticsHUD from './components/AnalyticsHUD';
import DigitalTwin3DCanvas from './components/DigitalTwin3DCanvas';
import LangChainModal from './components/LangChainModal';
import { PERU_DEPARTMENTS } from './data/peruGeoData';

export default function App() {
  // Parámetros de Política
  const [policyParams, setPolicyParams] = useState({
    modelName: 'Doubly Robust (IPW + Ridge AIPW)',
    coverageExpansion: 85,
    targetingStrategy: 'sisfoh_pobreza',
    medsSubsidy: 50,
    catastrophicCapEnabled: true,
    catastrophicCapRatio: 30
  });

  // Estados 3D
  const [cameraMode, setCameraMode] = useState('cinematic'); // 'cinematic' | 'topdown' | 'focus'
  const [autoRotate, setAutoRotate] = useState(true);
  const [selectedDept, setSelectedDept] = useState(null);
  const [pulseTrigger, setPulseTrigger] = useState(0);

  // Estados de Datos de Simulación
  const [backendOnline, setBackendOnline] = useState(false);
  const [isLangChainOpen, setIsLangChainOpen] = useState(false);
  const [kpis, setKpis] = useState({
    coverage_pct_before: 89.2,
    coverage_pct_after: 98.7,
    coverage_increase_pts: 9.5,
    che_40_capacity_before: 13.8,
    che_40_capacity_after: 11.2,
    che_40_reduction_pts: 2.6,
    che_10_pct_before: 28.3,
    che_10_pct_after: 22.4,
    che_10_reduction_pts: 5.9,
    impoverished_pct_before: 4.1,
    impoverished_pct_after: 3.3,
    impoverished_reduction_pts: 0.8,
    total_monthly_oope_savings_soles: 341787.14,
    avg_monthly_savings_per_new_affiliate_soles: 254.3
  });

  const [departmentsData, setDepartmentsData] = useState(() =>
    PERU_DEPARTMENTS.map((d) => ({
      department: d.name,
      base_che_40_pct: d.baseChe40,
      sim_che_40_pct: d.baseChe40 * 0.78,
      che_40_reduction_pts: +(d.baseChe40 * 0.22).toFixed(1),
      new_affiliates_count: Math.round(d.population * 0.00012),
      sim_coverage_pct: 98.4,
      total_savings_soles: Math.round(d.population * 0.015)
    }))
  );

  // Ejecutar Simulación (Conexión a FastAPI o simulación local instantánea)
  const runSimulation = useCallback(async () => {
    try {
      const payload = {
        coverage_expansion_rate: policyParams.coverageExpansion / 100.0,
        targeting_strategy: policyParams.targetingStrategy,
        meds_subsidy_depth: policyParams.medsSubsidy / 100.0,
        catastrophic_cap_enabled: policyParams.catastrophicCapEnabled,
        catastrophic_cap_ratio: policyParams.catastrophicCapRatio / 100.0,
        model_name: policyParams.modelName
      };

      const res = await fetch('http://127.0.0.1:8000/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (res.ok) {
        const data = await res.json();
        setKpis(data.kpi_national_summary);
        setDepartmentsData(data.department_summary);
        setBackendOnline(true);
      } else {
        throw new Error('API response not ok');
      }
    } catch (err) {
      // Fallback a motor de simulación local
      setBackendOnline(false);
      computeLocalSimulation();
    }
    setPulseTrigger((prev) => prev + 1);
  }, [policyParams]);

  // Motor de simulación local de respaldo
  const computeLocalSimulation = () => {
    const expansion = policyParams.coverageExpansion / 100.0;
    const meds = policyParams.medsSubsidy / 100.0;
    const cap = policyParams.catastrophicCapEnabled ? 0.82 : 1.0;

    const baseCov = 89.2;
    const simCov = +(baseCov + (100 - baseCov) * expansion * 0.92).toFixed(1);
    const covDelta = +(simCov - baseCov).toFixed(1);

    const baseChe = 13.8;
    const cheFactor = (1.0 - expansion * 0.24) * (1.0 - meds * 0.08) * cap;
    const simChe = +(baseChe * cheFactor).toFixed(1);
    const cheDelta = +(baseChe - simChe).toFixed(1);

    const baseImpov = 4.1;
    const simImpov = +(baseImpov * (1.0 - expansion * 0.22)).toFixed(1);

    const totalSavings = Math.round(expansion * 380000 * (1 + meds * 0.3));

    setKpis({
      coverage_pct_before: baseCov,
      coverage_pct_after: simCov,
      coverage_increase_pts: covDelta,
      che_40_capacity_before: baseChe,
      che_40_capacity_after: simChe,
      che_40_reduction_pts: cheDelta,
      che_10_pct_before: 28.3,
      che_10_pct_after: +(28.3 - cheDelta * 1.8).toFixed(1),
      che_10_reduction_pts: +(cheDelta * 1.8).toFixed(1),
      impoverished_pct_before: baseImpov,
      impoverished_pct_after: simImpov,
      impoverished_reduction_pts: +(baseImpov - simImpov).toFixed(1),
      total_monthly_oope_savings_soles: totalSavings,
      avg_monthly_savings_per_new_affiliate_soles: Math.round(214 + meds * 60)
    });

    setDepartmentsData(
      PERU_DEPARTMENTS.map((d) => {
        const regionalFactor = d.zone === 'Sierra' ? 1.25 : d.zone === 'Selva' ? 1.15 : 0.85;
        const redPts = +(d.baseChe40 * (1 - cheFactor) * regionalFactor).toFixed(1);
        return {
          department: d.name,
          base_che_40_pct: d.baseChe40,
          sim_che_40_pct: Math.max(4.0, +(d.baseChe40 - redPts).toFixed(1)),
          che_40_reduction_pts: redPts,
          new_affiliates_count: Math.round(d.population * 0.00015 * expansion),
          sim_coverage_pct: +(d.baseCoverage + (100 - d.baseCoverage) * expansion * 0.9).toFixed(1),
          total_savings_soles: Math.round(d.population * 0.02 * expansion)
        };
      })
    );
  };

  useEffect(() => {
    runSimulation();
  }, [runSimulation]);

  const handleParamChange = (param, value) => {
    setPolicyParams((prev) => ({ ...prev, [param]: value }));
  };

  const handleResetParams = () => {
    setPolicyParams({
      modelName: 'Doubly Robust (IPW + Ridge AIPW)',
      coverageExpansion: 85,
      targetingStrategy: 'sisfoh_pobreza',
      medsSubsidy: 50,
      catastrophicCapEnabled: true,
      catastrophicCapRatio: 30
    });
    setSelectedDept(null);
    setCameraMode('cinematic');
  };

  const selectedDeptData = departmentsData.find((d) => d.department === selectedDept);

  return (
    <div className="flex flex-col h-screen w-screen bg-slate-950 text-slate-100 font-sans overflow-hidden">
      {/* Top Navbar */}
      <Navbar backendOnline={backendOnline} totalPopulation={15000} />

      {/* Main Container */}
      <div className="flex flex-1 relative overflow-hidden">
        {/* Left Control Panel */}
        <ControlPanel
          policyParams={policyParams}
          onParamChange={handleParamChange}
          cameraMode={cameraMode}
          setCameraMode={setCameraMode}
          autoRotate={autoRotate}
          setAutoRotate={setAutoRotate}
          onOpenLangChain={() => setIsLangChainOpen(true)}
          onResetParams={handleResetParams}
        />

        {/* Center 3D Scene */}
        <main className="flex-1 relative h-full">
          <DigitalTwin3DCanvas
            departmentsData={departmentsData}
            selectedDept={selectedDept}
            onSelectDept={setSelectedDept}
            cameraMode={cameraMode}
            autoRotate={autoRotate}
            pulseTrigger={pulseTrigger}
          />
        </main>

        {/* Right Analytics HUD */}
        <AnalyticsHUD
          kpis={kpis}
          selectedDeptData={selectedDeptData}
          onClearSelectedDept={() => setSelectedDept(null)}
        />
      </div>

      {/* Modal LangChain */}
      <LangChainModal
        isOpen={isLangChainOpen}
        onClose={() => setIsLangChainOpen(false)}
      />
    </div>
  );
}
