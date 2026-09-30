* ==============================================================================
* DO-FILE STATA: PROCESAMIENTO DE MICRODATOS ENAHO (INEI PERÚ) A TABLA DE ANÁLISIS
* PROYECTO: GEMELO DIGITAL DE SALUD PÚBLICA Y GASTO CATASTRÓFICO
* ==============================================================================

clear all
set more off

* ------------------------------------------------------------------------------
* 1. RUTAS DE ENTRADA Y SALIDA
* ------------------------------------------------------------------------------
global DATA_DIR "data"
global OUT_DIR  "data"

* ------------------------------------------------------------------------------
* 2. MÓDULO 04: SALUD INDIVIDUAL (Enaho01-2025-400.dta)
* ------------------------------------------------------------------------------
use "$DATA_DIR/Enaho01-2025-400.dta", clear

* Identificador de seguro (P4191: SIS, P4192: EsSalud, P4198: Sin Seguro)
gen tiene_sis     = (p4191 == 1)
gen tiene_essalud = (p4192 == 1)
gen sin_seguro    = (p4198 == 1)

* Morbilidad y hospitalizaciones
gen es_cronico    = (p401 == 1)
gen es_agudo      = (p402 == 1)
gen fue_hosp      = (p407 == 1)

* Gasto de bolsillo mensualizado por persona (medicinas, consultas, análisis, hospital)
recode p414_1 p414_2 p414_3 p414_4 (. = 0)
gen oope_persona = p414_1 + p414_2 + p414_3 + p414_4

* Colapso a nivel de hogar (Llaves: conglome vivienda hogar)
collapse (sum) miembros_sis = tiene_sis ///
               miembros_sin_seguro = sin_seguro ///
               miembros_essalud = tiene_essalud ///
               chronic_disease_count = es_cronico ///
               oope_total = oope_persona ///
         (max) has_chronic_disease = es_cronico ///
               has_acute_illness_4w = es_agudo ///
               had_hospitalization = fue_hosp, ///
         by(conglome vivienda hogar)

tempfile salud_hogar
save `salud_hogar', replace

* ------------------------------------------------------------------------------
* 3. MÓDULO 02: MIEMBROS DEL HOGAR (Enaho01-2025-200.dta)
* ------------------------------------------------------------------------------
use "$DATA_DIR/Enaho01-2025-200.dta", clear

* Miembros residentes habituales
keep if p204 == 1

gen es_u5      = (p208a < 5)
gen es_elderly = (p208a >= 60)
gen es_jefe    = (p203 == 1)
gen es_mujer_jefe = (p203 == 1 & p207 == 2)

* Colapso demográfico a nivel de hogar
collapse (count) household_size = codperso ///
         (sum)   num_children_u5 = es_u5 ///
                 num_elderly = es_elderly ///
         (max)   has_children_u5 = es_u5 ///
                 has_elderly = es_elderly ///
                 is_female_head = es_mujer_jefe, ///
         by(conglome vivienda hogar)

* Ratio de dependencia demográfica
gen working_adults = max(1, household_size - num_children_u5 - num_elderly)
gen dependency_ratio = (num_children_u5 + num_elderly) / working_adults

tempfile demografia_hogar
save `demografia_hogar', replace

* ------------------------------------------------------------------------------
* 4. MÓDULO 34: SUMARIA (GASTOS, INGRESOS Y FACTORES DE EXPANSIÓN)
* ------------------------------------------------------------------------------
use "$DATA_DIR/sumaria-2025.dta", clear

* Mensualización de variables anuales oficiales del INEI
gen monthly_total_exp    = gashog2d / 12
gen monthly_income       = inghog2d / 12
gen subsistence_food_exp = grupo1 / 12

* Capacidad de pago neta (OMS / Xu et al., 2003)
gen capacity_to_pay = max(10, monthly_total_exp - subsistence_food_exp)

* Factor de expansión muestral
gen factor_expansion = factor07

* Departamento a partir de Ubigeo
gen dpto_code = substr(ubigeo, 1, 2)
gen is_rural  = (estrato >= 7)

* Condición de Pobreza oficial INEI
gen poverty_status = "No Pobre"
replace poverty_status = "Pobre No Extremo" if pobreza == 2
replace poverty_status = "Pobre Extremo"    if pobreza == 1

* ------------------------------------------------------------------------------
* 5. MERGE RELACIONAL ENTRE MÓDULOS
* ------------------------------------------------------------------------------
merge 1:1 conglome vivienda hogar using `salud_hogar', keep(match master) nogen
merge 1:1 conglome vivienda hogar using `demografia_hogar', keep(match master) nogen

* Reemplazar valores nulos en gasto de salud
replace oope_total = 0 if missing(oope_total)

* ------------------------------------------------------------------------------
* 6. DEFINICIÓN DE TRATAMIENTO E INDICADORES DE GASTO CATASTRÓFICO (CHE)
* ------------------------------------------------------------------------------
* Tratamiento: SIS vs Sin Seguro
gen treatment_sis = .
replace treatment_sis = 1 if miembros_sis > 0
replace treatment_sis = 0 if miembros_sin_seguro > 0 & miembros_sis == 0

* Indicadores OMS de Gasto Catastrófico en Salud
gen che_40_capacity = ( (oope_total / capacity_to_pay) >= 0.40 )
gen che_10          = ( (oope_total / monthly_total_exp) >= 0.10 )

* Empobrecimiento por motivos de salud
gen exp_post_health_pc = (monthly_total_exp - oope_total) / household_size
gen impoverished_by_health = (poverty_status == "No Pobre" & exp_post_health_pc < linea)

* ------------------------------------------------------------------------------
* 7. EXPORTACIÓN DE TABLA DE ANÁLISIS
* ------------------------------------------------------------------------------
export delimited using "$OUT_DIR/tabla_analisis_enaho.csv", replace
di "Procesamiento concluido: Base de análisis exportada exitosamente."
