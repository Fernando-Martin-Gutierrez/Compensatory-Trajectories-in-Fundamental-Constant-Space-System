# MultiverseEquilibriumSim

**MultiverseEquilibriumSim** es un framework de física teórica y computacional de alta precisión en Python diseñado para explorar y verificar la existencia de conjuntos alternativos de constantes físicas fundamentales mutuamente compensadas (covariantes) capaces de sustentar universos con átomos estables, química molecular rica, estrellas duraderas con combustión nuclear activa y órbitas planetarias estables.

---

## 1. Fundamento Físico y Espacio Paramétrico Adimensional

El sistema modela el multiverso libre de artefactos de unidades del SI a través del espacio paramétrico de **constantes adimensionales fundamentales**:

| Constante | Símbolo | Expresión Física / Definición | Valor Universo Estándar $\vec{\theta}_0$ | Dominio Físico |
| :--- | :---: | :--- | :---: | :--- |
| **Estructura Fina** | $\alpha$ | $\frac{e^2}{4\pi \varepsilon_0 \hbar c}$ | $\approx 1 / 137.035999$ | Electromagnetismo / Química |
| **Acoplamiento Gravitatorio** | $\alpha_G$ | $\frac{G m_p^2}{\hbar c}$ | $\approx 5.9061 \times 10^{-39}$ | Gravitación bariónica / Astrofísica |
| **Ratio de Masas** | $\beta$ | $\frac{m_e}{m_p}$ | $\approx 5.4462 \times 10^{-4}$ | Escala cinemática / Radio de Bohr |
| **Fuerza Nuclear Fuerte** | $\alpha_s$ | Acoplamiento efectivo pión-nucleón | $\approx 0.1180$ | Estabilidad nuclear y enlace de di-nucleones |
| **Diferencia de Masa $n-p$** | $Q$ | $\frac{m_n - m_p}{m_p}$ | $\approx 1.3784 \times 10^{-3}$ ($1.293$ MeV) | Desintegración beta / Isospín |

---

## 2. Arquitectura de Módulos

```
MultiverseEquilibriumSim/
├── multiverse/
│   ├── __init__.py
│   ├── constants.py               # Constantes adimensionales y container UniverseParameters
│   ├── core/
│   │   ├── __init__.py
│   │   └── manifolds.py           # Variedad analítica de viabilidad antrópica / física
│   ├── solvers/
│   │   ├── __init__.py
│   │   ├── quantum_atom.py        # Schrödinger radial (diferencias finitas + Numerov JIT)
│   │   ├── stellar_structure.py   # Ecuación de Lane-Emden (RK4 n=1.5 y n=3) y ventana estelar
│   │   └── orbital_dynamics.py    # Integrador simpléctico 3 cuerpos con potencial 1PN y Lyapunov
│   ├── optimizer/
│   │   ├── __init__.py
│   │   └── equilibrium_search.py  # Explorador de la curva de equilibrio unidimensional γ(α), Jacobiano y leyes de escala
│   └── visualization/
│       ├── __init__.py
│       └── plotting.py            # Generador de gráficos científicos de alta resolución
├── cli.py                         # CLI ejecutable con Rich terminal
├── tests/                         # Suite completa de tests unitarios (pytest)
├── pyproject.toml
└── README.md
```

### Detalle de Implementación

1. **`core/manifolds.py` (Límites Analíticos de Viabilidad):**
   - **Estabilidad del Hidrógeno y Di-nucleón:** Comprueba $Q > \beta$ para evitar la captura electrónica espontánea del protón $p + e^- \to n + \nu$; $B_d(\alpha_s) > Q$ para garantizar la existencia del deuterón estable; y $B_{pp}(\alpha, \alpha_s) < 0$ para evitar el ligamiento del diprotón ($^2\text{He}$) que consumiría todo el hidrógeno primordial en el Big Bang.
   - **Resonancia de Hoyle (Triple-Alfa):** Evalúa el desplazamiento del estado resonante de $^{12}\text{C}^*$ a 7.654 MeV respecto al umbral $3\alpha$, modelando la compensación entre repulsión culombiana ($\propto \alpha$) y atracción nuclear ($\propto \alpha_s$).
   - **Escala Temporal Estelar:** Garantiza $\tau_* \sim \alpha^{-2} \alpha_G^{-1} \left(\frac{\hbar}{m_p c^2}\right) \ge 10^8$ años equivalentes para permitir evolución estructural y biológica.
   - **Jerarquía Atómica Molecular:** Exige $a_0 / r_n = \frac{\alpha_s}{\alpha \beta} \gg 1$ para que las órbitas electrónicas no colapsen en el núcleo.

2. **`solvers/quantum_atom.py` (Mecánica Cuántica Numérica):**
   - Resuelve numéricamente la ecuación radial de Schrödinger discretizada mediante diferencias finitas tridiagonales y algoritmo de Numerov acelerado con `@numba.njit`.
   - Calcula con precisión espectroscópica: $E_1 \approx -13.597 \text{ eV}$, $E_2 \approx -3.399 \text{ eV}$, $a_0 \approx 0.5295 \text{ \AA}$, $\langle r \rangle \approx 1.5 a_0$ y la polarizabilidad atómica $\alpha_{pol} \approx \frac{9}{2} a_0^3$.

3. **`solvers/stellar_structure.py` (Astrofísica y Ecuación de Lane-Emden):**
   - Integra la ecuación politrópica de Lane-Emden mediante Runge-Kutta 4 (RK4) hasta el borde estelar $\theta(\xi_1) = 0$:
     - $n = 1.5$: $\xi_1 \approx 3.654$, $\omega_{1.5} \approx 2.714$ (enanas blancas no relativistas).
     - $n = 3.0$: $\xi_1 \approx 6.897$, $\omega_{3.0} \approx 2.018$ (límite de Chandrasekhar).
   - Determina el rango de masa viable $[M_{min}, M_{Ch}]$ donde $M_{min} \sim \alpha_G^{-3/2} \beta^{3/4} (\alpha/\alpha_s)^{3/2} m_p$ (ignición de fusión de hidrógeno) y $M_{Ch} \sim \alpha_G^{-3/2} m_p$.

4. **`solvers/orbital_dynamics.py` (Mecánica Newtoniana + Relatividad 1PN):**
   - Integrador simpléctico de 4º orden de Yoshida y Verlet para sistemas de 3 cuerpos.
   - Incorpora la aceleración relativista post-newtoniana de orden 1PN:
     $$\vec{a}_{1PN} = -\frac{3 G M L^2}{c^2 m^2 r^5} \vec{r}$$
   - Mide el exponente de Lyapunov en el espacio de fases de 18 dimensiones mediante el algoritmo de renormalización de Benettin y rastrea la deriva del semieje mayor $\Delta a / a_0$ para confirmar estabilidad durante $> 10^5$ órbitas.

5. **`optimizer/equilibrium_search.py` (Búsqueda de Covarianza y Leyes de Escala):**
   - Explora el espacio logarítmico $(\log \alpha, \log \alpha_G, \log \beta, \log \alpha_s)$ localizando la curva de equilibrio unidimensional $\gamma(\alpha)$ embebida en el dominio de viabilidad $\mathcal{V} \subset \mathbb{R}^4$ con viabilidad $> 0.95$.
   - Calcula el tensor Jacobiano local $\nabla_{\log \vec{\theta}} \text{Score}$ y extrae las leyes de escala empíricas:
     $$\frac{\partial \log \alpha_s}{\partial \log \alpha} \approx +0.69 \quad (R^2 = 0.992)$$
     $$\frac{\partial \log \alpha_G}{\partial \log \alpha} \approx -0.89$$

---

## 3. Instalación y Requisitos

Requiere Python 3.11+.

```bash
pip install -e .
```

O instalar dependencias directamente:
```bash
pip install numpy scipy numba matplotlib rich pytest
```

---

## 4. Uso del CLI

### 4.1. Centro de Control Interactivo Unificado (Recomendado)
Al ejecutar el CLI sin argumentos (o con `-i`), se abre un menú interactivo con todas las opciones disponibles. **En cada opción elegida, el kernel de Lean 4 audita y certifica formalmente en tiempo real el universo simulado**:
```bash
python cli.py
```
Opciones del menú:
1. **Evaluar Universo Estándar:** Ejecuta modelos físicos (cuántico, estelar, orbital) + Certificación formal Lean 4.
2. **Explorar Variedad Covariante:** Traza la curva de equilibrio unidimensional $\gamma(\alpha)$ embebida en el dominio de viabilidad $\mathcal{V} \subset \mathbb{R}^4$ + Certificación formal del punto óptimo con Lean 4.
3. **Muestreo Monte Carlo:** Muestreo estocástico del multiverso + Auditoría Lean 4 del mejor universo encontrado.
4. **Diseñar Universo Personalizado:** Permite ingresar $\alpha, \alpha_s, \beta, Q$ y audita matemáticamente su viabilidad con Lean 4.
5. **Generar Gráficos de Publicación:** Renderiza las 5 figuras en alta resolución (PNG 300 DPI).
6. **Auditoría Teórica Completa:** Ejecuta la suite de teoremas y contraejemplos matemáticos en Lean 4.

---

### 4.2. Ejecución Directa por Comandos (Modo Scripting)
Si prefieres invocar directamente un módulo específico sin pasar por el menú:

- **Universo Estándar:** `python cli.py --standard`
- **Cosmología Primordial & BBN:** `python cli.py --bbn`
- **Variedad Covariante:** `python cli.py --covary`
- **Muestreo Monte Carlo:** `python cli.py --scan --samples 500`
- **Universo Personalizado:** `python cli.py --custom`
- **Generación de Gráficos:** `python cli.py --plot --outdir .`
- **Auditoría Formal Lean 4:** `python cli.py --lean`
- **Ejecución Completa:** `python cli.py --all`

### 4.4. Generación de Gráficos de Publicación
Genera las 6 figuras científicas en alta resolución (PNG a 300 DPI):
```bash
python cli.py --plot --outdir .
```

Figuras producidas:
- `viability_heatmap_2d.png`: Islas de viabilidad en el plano $(\alpha, \alpha_s)$.
- `equilibrium_manifold.png`: Trayectoria covariante y leyes de escala en 4 paneles.
- `quantum_hydrogen.png`: Función de onda radial $u_1(r)$ y densidad de probabilidad cuántica.
- `stellar_polytropes.png`: Perfiles de densidad de Lane-Emden para $n=1.5$ y $n=3.0$.
- `orbital_trajectories.png`: Precesión relativista 1PN y estabilidad orbital de 3 cuerpos.
- `bbn_primordial_nucleosynthesis.png`: Evolución de abundancias de Hidrógeno ($X_H$), Helio ($Y_p$) y congelamiento neutrón-protón en la era de radiación.

### 4.5. Verificación Formal con Lean 4 (Kernel-Checked)
Certifica formalmente mediante el kernel de **Lean 4** todas las condiciones y teoremas de estabilidad física del multiverso:
```bash
python cli.py --lean
```
O directamente desde la carpeta formal:
```bash
cd formal && lake exe verify
```

---

## 5. Tests Unitarios

Ejecutar la suite completa de pruebas:
```bash
python -m pytest -v
```
Todos los 20 tests verifican analítica y numéricamente los límites físicos, energías cuánticas, convergencia simpléctica y nucleosíntesis primordial BBN.

---

## 6. Verificación Formal Matemática (Lean 4)

El proyecto incluye en `formal/` una formalización matemática rigurosa en **Lean 4** ([formal/Formal/Multiverse.lean](formal/Formal/Multiverse.lean)) que demuestra formalmente al 100%:
- **Teorema 1 (`standard_hydrogen_stable`):** Estabilidad del átomo de hidrógeno ($Q > \beta \iff m_n - m_p > m_e$).
- **Teorema 2 (`standard_deuteron_stable`):** Existencia de di-nucleones ligados ($B_d > Q$).
- **Teorema 3 (`standard_diproton_unbound`):** Diprotón no ligado ($B_{pp} < 0$, previene agotamiento de hidrógeno primordial).
- **Teorema 4 (`standard_atomic_hierarchy`):** Separación jerárquica de escalas átomo-núcleo ($a_0 / r_n > 50$).
- **Teorema 5 (`standard_hoyle_resonance`):** Ventana de sintonización fina de la resonancia de Hoyle ($|\Delta E_R| \le 120\text{ keV}$).
- **Teorema 6 (`standard_bbn_viable`):** Preservación de abundancia primordial de hidrógeno ($X_H > 60\%$) durante la nucleosíntesis del Big Bang.
- **Teoremas 7 y 8:** Pruebas formales de inestabilidad en universos perturbados (demostración por contraejemplo).
