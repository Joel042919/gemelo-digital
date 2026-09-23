import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { PERU_DEPARTMENTS } from '../data/peruGeoData';

export default function DigitalTwin3DCanvas({
  departmentsData,
  selectedDept,
  onSelectDept,
  cameraMode,
  autoRotate,
  pulseTrigger
}) {
  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const rendererRef = useRef(null);
  const cameraRef = useRef(null);
  const monolithsRef = useRef([]);
  const particlesRef = useRef(null);
  const shockwaveRef = useRef(null);
  const hoverMeshRef = useRef(null);
  const [hoveredInfo, setHoveredInfo] = useState(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });

  // Referencias para control de ratón orbital
  const isDraggingRef = useRef(false);
  const prevMouseRef = useRef({ x: 0, y: 0 });
  const cameraAnglesRef = useRef({ theta: 0.85, phi: 0.95, radius: 45 });
  const targetLookAtRef = useRef(new THREE.Vector3(0, 2, 0));

  useEffect(() => {
    const currentMount = mountRef.current;
    if (!currentMount) return;

    const width = currentMount.clientWidth;
    const height = currentMount.clientHeight;

    // 1. Crear Escena y Niebla Atmosférica
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x030712); // Slate-950
    scene.fog = new THREE.FogExp2(0x030712, 0.015);
    sceneRef.current = scene;

    // 2. Cámara de Perspectiva
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    cameraRef.current = camera;
    updateCameraPosition();

    // 3. Renderer WebGL
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    currentMount.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // 4. Luces
    const ambientLight = new THREE.AmbientLight(0x38bdf8, 0.6); // Cyan suave
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
    dirLight.position.set(30, 50, 20);
    dirLight.castShadow = true;
    dirLight.shadow.mapSize.width = 2048;
    dirLight.shadow.mapSize.height = 2048;
    scene.add(dirLight);

    const pointLight = new THREE.PointLight(0x818cf8, 2.0, 60);
    pointLight.position.set(0, 15, 0);
    scene.add(pointLight);

    // 5. Grid y Plataforma 3D
    const gridHelper = new THREE.GridHelper(70, 35, 0x38bdf8, 0x1e293b);
    gridHelper.position.y = -0.05;
    scene.add(gridHelper);

    // Base reflectante
    const floorGeo = new THREE.PlaneGeometry(80, 80);
    const floorMat = new THREE.MeshStandardMaterial({
      color: 0x090d16,
      roughness: 0.8,
      metalness: 0.3
    });
    const floor = new THREE.Mesh(floorGeo, floorMat);
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = -0.1;
    floor.receiveShadow = true;
    scene.add(floor);

    // 6. Monolitos de los 24 Departamentos
    const monoliths = [];
    PERU_DEPARTMENTS.forEach((dept) => {
      // Base hexagonal o cilíndrica estilizada
      const geo = new THREE.CylinderGeometry(1.2, 1.4, 4, 6);
      geo.translate(0, 2, 0);

      const mat = new THREE.MeshStandardMaterial({
        color: 0x3b82f6,
        metalness: 0.6,
        roughness: 0.2,
        emissive: 0x1d4ed8,
        emissiveIntensity: 0.3
      });

      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.set(dept.x * 1.5, 0, dept.z * 1.5);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      mesh.userData = { deptId: dept.id, deptName: dept.name, baseData: dept, targetHeight: 4, currentHeight: 4 };

      // Anillo de base brillante
      const ringGeo = new THREE.RingGeometry(1.4, 1.6, 6);
      const ringMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8, side: THREE.DoubleSide });
      const ring = new THREE.Mesh(ringGeo, ringMat);
      ring.rotation.x = -Math.PI / 2;
      ring.position.y = 0.02;
      mesh.add(ring);

      scene.add(mesh);
      monoliths.push(mesh);
    });
    monolithsRef.current = monoliths;

    // 7. Partículas de Agentes de la Población en Órbita
    const particleCount = 1200;
    const particleGeo = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);

    for (let i = 0; i < particleCount; i++) {
      const idx = i * 3;
      const radius = 5 + Math.random() * 28;
      const angle = Math.random() * Math.PI * 2;
      positions[idx] = Math.cos(angle) * radius;
      positions[idx + 1] = 0.5 + Math.random() * 8;
      positions[idx + 2] = Math.sin(angle) * radius;

      // Color inicial (Cyan / Verde / Ámbar)
      colors[idx] = 0.2 + Math.random() * 0.2;
      colors[idx + 1] = 0.7 + Math.random() * 0.3;
      colors[idx + 2] = 0.9;
    }

    particleGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    particleGeo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const particleMat = new THREE.PointsMaterial({
      size: 0.35,
      vertexColors: true,
      transparent: true,
      opacity: 0.85,
      blending: THREE.AdditiveBlending
    });

    const particles = new THREE.Points(particleGeo, particleMat);
    scene.add(particles);
    particlesRef.current = particles;

    // 8. Anillo Shockwave para cambios de política
    const shockGeo = new THREE.RingGeometry(0.1, 0.8, 32);
    const shockMat = new THREE.MeshBasicMaterial({
      color: 0x10b981,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.0
    });
    const shockwave = new THREE.Mesh(shockGeo, shockMat);
    shockwave.rotation.x = -Math.PI / 2;
    shockwave.position.y = 0.1;
    scene.add(shockwave);
    shockwaveRef.current = { mesh: shockwave, scale: 1, active: false };

    // 9. Raycasting para interacción con el ratón
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();

    const handleMouseMove = (event) => {
      const rect = currentMount.getBoundingClientRect();
      mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

      setTooltipPos({ x: event.clientX - rect.left + 15, y: event.clientY - rect.top - 20 });

      // Dragging orbital
      if (isDraggingRef.current) {
        const deltaX = event.clientX - prevMouseRef.current.x;
        const deltaY = event.clientY - prevMouseRef.current.y;
        prevMouseRef.current = { x: event.clientX, y: event.clientY };

        cameraAnglesRef.current.theta -= deltaX * 0.006;
        cameraAnglesRef.current.phi = Math.max(0.2, Math.min(Math.PI / 2 - 0.05, cameraAnglesRef.current.phi - deltaY * 0.006));
        updateCameraPosition();
      } else {
        // Raycasting hover
        raycaster.setFromCamera(mouse, camera);
        const intersects = raycaster.intersectObjects(monolithsRef.current);

        if (intersects.length > 0) {
          const hit = intersects[0].object;
          hoverMeshRef.current = hit;
          const dept = departmentsData.find((d) => d.department === hit.userData.deptName) || hit.userData.baseData;
          setHoveredInfo({
            name: hit.userData.deptName,
            zone: hit.userData.baseData.zone,
            population: hit.userData.baseData.population,
            baseChe40: dept.base_che_40_pct || hit.userData.baseData.baseChe40,
            simChe40: dept.sim_che_40_pct !== undefined ? dept.sim_che_40_pct : hit.userData.baseData.baseChe40,
            reduction: dept.che_40_reduction_pts || 0,
            savings: dept.total_savings_soles || 0
          });
        } else {
          hoverMeshRef.current = null;
          setHoveredInfo(null);
        }
      }
    };

    const handleMouseDown = (event) => {
      isDraggingRef.current = true;
      prevMouseRef.current = { x: event.clientX, y: event.clientY };
    };

    const handleMouseUp = () => {
      isDraggingRef.current = false;
      if (hoverMeshRef.current) {
        onSelectDept(hoverMeshRef.current.userData.deptName);
      }
    };

    const handleWheel = (event) => {
      event.preventDefault();
      cameraAnglesRef.current.radius = Math.max(15, Math.min(80, cameraAnglesRef.current.radius + event.deltaY * 0.05));
      updateCameraPosition();
    };

    const handleResize = () => {
      if (!currentMount) return;
      const w = currentMount.clientWidth;
      const h = currentMount.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };

    currentMount.addEventListener('mousemove', handleMouseMove);
    currentMount.addEventListener('mousedown', handleMouseDown);
    window.addEventListener('mouseup', handleMouseUp);
    currentMount.addEventListener('wheel', handleWheel, { passive: false });
    window.addEventListener('resize', handleResize);

    // 10. Loop de Animación
    let animationFrameId;
    let clock = new THREE.Clock();

    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);
      const elapsedTime = clock.getElapsedTime();

      // Rotación automática si está activa
      if (autoRotate && !isDraggingRef.current) {
        cameraAnglesRef.current.theta += 0.002;
        updateCameraPosition();
      }

      // Animación suave de altura y color de los monolitos según datos de política
      monolithsRef.current.forEach((mesh) => {
        mesh.userData.currentHeight += (mesh.userData.targetHeight - mesh.userData.currentHeight) * 0.08;
        mesh.scale.y = Math.max(0.2, mesh.userData.currentHeight / 4.0);

        // Pulso suave en el departamento seleccionado
        if (selectedDept && mesh.userData.deptName === selectedDept) {
          const pulse = 1.0 + Math.sin(elapsedTime * 6) * 0.15;
          mesh.scale.x = 1.2 * pulse;
          mesh.scale.z = 1.2 * pulse;
          mesh.material.emissiveIntensity = 0.8;
        } else if (mesh === hoverMeshRef.current) {
          mesh.scale.x = 1.15;
          mesh.scale.z = 1.15;
          mesh.material.emissiveIntensity = 0.6;
        } else {
          mesh.scale.x = 1.0;
          mesh.scale.z = 1.0;
          mesh.material.emissiveIntensity = 0.25;
        }
      });

      // Animación de partículas flotantes
      if (particlesRef.current) {
        particlesRef.current.rotation.y = elapsedTime * 0.04;
        const pos = particlesRef.current.geometry.attributes.position.array;
        for (let i = 0; i < particleCount; i++) {
          const yIdx = i * 3 + 1;
          pos[yIdx] += Math.sin(elapsedTime * 2 + i) * 0.01;
        }
        particlesRef.current.geometry.attributes.position.needsUpdate = true;
      }

      // Animación de Shockwave de política
      if (shockwaveRef.current && shockwaveRef.current.active) {
        shockwaveRef.current.scale += 0.9;
        shockwaveRef.current.mesh.scale.set(shockwaveRef.current.scale, shockwaveRef.current.scale, 1);
        shockwaveRef.current.mesh.material.opacity = Math.max(0, 0.9 - shockwaveRef.current.scale / 35);

        if (shockwaveRef.current.scale > 35) {
          shockwaveRef.current.active = false;
          shockwaveRef.current.mesh.material.opacity = 0;
        }
      }

      renderer.render(scene, camera);
    };

    animate();

    return () => {
      cancelAnimationFrame(animationFrameId);
      currentMount.removeEventListener('mousemove', handleMouseMove);
      currentMount.removeEventListener('mousedown', handleMouseDown);
      window.removeEventListener('mouseup', handleMouseUp);
      currentMount.removeEventListener('wheel', handleWheel);
      window.removeEventListener('resize', handleResize);
      if (renderer.domElement && currentMount.contains(renderer.domElement)) {
        currentMount.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, []);

  // Función para reposicionar cámara según ángulos esféricos
  const updateCameraPosition = () => {
    if (!cameraRef.current) return;
    const { theta, phi, radius } = cameraAnglesRef.current;
    cameraRef.current.position.x = radius * Math.sin(phi) * Math.sin(theta);
    cameraRef.current.position.y = radius * Math.cos(phi);
    cameraRef.current.position.z = radius * Math.sin(phi) * Math.cos(theta);
    cameraRef.current.lookAt(targetLookAtRef.current);
  };

  // Cambio de modo de cámara
  useEffect(() => {
    if (!cameraRef.current) return;
    if (cameraMode === 'topdown') {
      cameraAnglesRef.current = { theta: 0.0, phi: 0.1, radius: 48 };
    } else if (cameraMode === 'cinematic') {
      cameraAnglesRef.current = { theta: 0.85, phi: 0.95, radius: 45 };
    } else if (cameraMode === 'focus' && selectedDept) {
      const match = PERU_DEPARTMENTS.find((d) => d.name === selectedDept);
      if (match) {
        targetLookAtRef.current.set(match.x * 1.5, 3, match.z * 1.5);
        cameraAnglesRef.current = { theta: 0.5, phi: 1.1, radius: 18 };
      }
    }
    updateCameraPosition();
  }, [cameraMode, selectedDept]);

  // Actualizar alturas y colores 3D cuando cambien los datos de la política
  useEffect(() => {
    if (!monolithsRef.current || monolithsRef.current.length === 0) return;

    monolithsRef.current.forEach((mesh) => {
      const deptName = mesh.userData.deptName;
      const liveData = departmentsData.find((d) => d.department === deptName);

      const che40 = liveData ? liveData.sim_che_40_pct : mesh.userData.baseData.baseChe40;

      // Altura proporcional al CHE 40% (4 a 14 unidades)
      mesh.userData.targetHeight = Math.max(1.5, che40 * 0.55);

      // Color dinámico según nivel de riesgo catastrófico
      let hexColor = 0x10b981; // Verde seguro (< 12%)
      let hexEmissive = 0x059669;

      if (che40 >= 18.0) {
        hexColor = 0xef4444; // Rojo crítico
        hexEmissive = 0xb91c1c;
      } else if (che40 >= 13.0) {
        hexColor = 0xf59e0b; // Ámbar moderado
        hexEmissive = 0xd97706;
      } else if (che40 >= 9.0) {
        hexColor = 0x06b6d4; // Cyan protegido
        hexEmissive = 0x0891b2;
      }

      mesh.material.color.setHex(hexColor);
      mesh.material.emissive.setHex(hexEmissive);
    });
  }, [departmentsData]);

  // Disparar shockwave cuando cambia la política
  useEffect(() => {
    if (shockwaveRef.current) {
      shockwaveRef.current.scale = 1;
      shockwaveRef.current.active = true;
    }
  }, [pulseTrigger]);

  return (
    <div className="relative w-full h-full select-none overflow-hidden" ref={mountRef}>
      {/* HUD Tooltip Flotante */}
      {hoveredInfo && (
        <div
          className="absolute z-20 pointer-events-none bg-slate-900/90 border border-cyan-500/50 backdrop-blur-md rounded-xl p-3 shadow-2xl text-xs w-64 transition-all duration-75"
          style={{ left: `${tooltipPos.x}px`, top: `${tooltipPos.y}px` }}
        >
          <div className="flex items-center justify-between border-b border-slate-700/60 pb-1.5 mb-2">
            <span className="font-bold text-sm text-cyan-400">{hoveredInfo.name}</span>
            <span className="bg-slate-800 text-slate-300 px-2 py-0.5 rounded text-[10px] uppercase tracking-wider">
              {hoveredInfo.zone}
            </span>
          </div>
          <div className="space-y-1 text-slate-300">
            <div className="flex justify-between">
              <span className="text-slate-400">Población Aprox:</span>
              <span className="font-semibold text-slate-200">{hoveredInfo.population.toLocaleString()} hab.</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">CHE 40% (Línea Base):</span>
              <span className="font-semibold text-rose-400">{hoveredInfo.baseChe40.toFixed(1)}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">CHE 40% (Simulado CSU):</span>
              <span className="font-semibold text-emerald-400">{hoveredInfo.simChe40.toFixed(1)}%</span>
            </div>
            <div className="flex justify-between border-t border-slate-800 pt-1 text-cyan-300 font-bold">
              <span>Reducción Catastrófica:</span>
              <span>-{hoveredInfo.reduction.toFixed(1)} pts</span>
            </div>
          </div>
        </div>
      )}

      {/* Leyenda 3D Inferior */}
      <div className="absolute bottom-4 left-6 z-10 bg-slate-900/80 backdrop-blur-md border border-slate-800 px-4 py-2.5 rounded-xl flex items-center space-x-5 text-xs text-slate-300 shadow-lg">
        <span className="font-bold text-slate-400">Gasto Catastrófico CHE 40%:</span>
        <div className="flex items-center space-x-1.5">
          <span className="w-3 h-3 rounded-full bg-emerald-500 inline-block shadow-sm shadow-emerald-500/50"></span>
          <span>Bajo (&lt;12%)</span>
        </div>
        <div className="flex items-center space-x-1.5">
          <span className="w-3 h-3 rounded-full bg-cyan-500 inline-block shadow-sm shadow-cyan-500/50"></span>
          <span>Protegido (12-13%)</span>
        </div>
        <div className="flex items-center space-x-1.5">
          <span className="w-3 h-3 rounded-full bg-amber-500 inline-block shadow-sm shadow-amber-500/50"></span>
          <span>Moderado (13-18%)</span>
        </div>
        <div className="flex items-center space-x-1.5">
          <span className="w-3 h-3 rounded-full bg-rose-500 inline-block shadow-sm shadow-rose-500/50"></span>
          <span>Crítico (&gt;18%)</span>
        </div>
      </div>
    </div>
  );
}
