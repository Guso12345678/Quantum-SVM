# Classical SVM vs. Quantum Kernel SVM — Comparativa en dataset XOR
 
Proyecto final de la asignatura **Computación Cuántica** (IMAT — Comillas ICAI).  
Implementación y comparativa de clasificadores SVM clásicos (kernel lineal y RBF) frente a SVM con **kernel cuántico** construido mediante circuitos de tipo ZZ Feature Map en Qiskit, evaluados sobre un dataset XOR no lineal con ruido.
 
---
 
## Descripción
 
El proyecto explora si los kernels cuánticos pueden superar a los kernels clásicos en problemas de clasificación no lineal. Se construye un **Quantum Kernel** calculando la fidelidad entre statevectors de circuitos cuánticos (overlap entre estados) y se usa como función de kernel precomputada en un SVM estándar.
 
Se comparan **5 modelos** con distintos números de qubits y dos kernels clásicos de referencia.
 
---
 
## Estructura del proyecto
 
```
final_project_revised/
├── final_project_svm - revised.ipynb   # Notebook principal
├── Figures_New/                         # Gráficas generadas
│   ├── xor_data.png                     # Dataset XOR con ruido
│   ├── linear_boundary.png              # Frontera SVM lineal
│   ├── rbf_boundary.png                 # Frontera SVM RBF
│   ├── 2q_boundary.png                  # Frontera Quantum SVM 2 qubits
│   ├── 4q_boundary.png                  # Frontera Quantum SVM 4 qubits
│   ├── 8q_boundary.png                  # Frontera Quantum SVM 8 qubits
│   ├── confusion_*.png                  # Matrices de confusión (5 modelos)
│   ├── metrics_comparison_barplot.png   # Comparativa Accuracy y F1
│   ├── quantum_feature_map_circuit.png  # Circuito del feature map
│   └── pipeline.png                     # Diagrama del pipeline
└── report_revised.pdf                   # Informe del proyecto
```
 
---
 
## Cómo ejecutar
 
**Requisitos:** Python 3.9+
 
```bash
pip install numpy matplotlib scikit-learn qiskit
jupyter notebook "final_project_svm - revised.ipynb"
```
 
---
 
## 🔬 Pipeline del experimento
 
### 1. Dataset XOR no lineal
 
Se genera un dataset sintético XOR con ruido gaussiano: la etiqueta es positiva si `x₁ · x₂ > 0` y negativa en caso contrario, con ruido `σ = 0.25` para hacer el problema más realista.
 
```python
def make_xor(n_samples=400, noise=0.25):
    X = rng.uniform(-1.0, 1.0, size=(n_samples, 2))
    y = (X[:, 0] * X[:, 1] > 0).astype(int)
    X = X + rng.normal(0, noise, size=X.shape)
    return X, y
```
 
Split: 70% train / 30% test, estratificado. Preprocesado con `StandardScaler`.
 
---
 
### 2. Modelos clásicos de referencia
 
**SVM Lineal** — kernel lineal, `C=1.0`. Incapaz de separar XOR (problema no linealmente separable). Se usa como baseline.
 
**SVM RBF** — kernel radial basis function, `C=1.0`, `gamma='scale'`. Referencia no lineal clásica.
 
---
 
### 3. Quantum Kernel SVM
 
#### 3.1 Feature Map ZZ
 
Cada punto de datos `x = (x₁, x₂)` se codifica en un circuito cuántico de tipo ZZ Feature Map con `n_qubits` y `reps=2` repeticiones:
 
```python
def feature_map_circuit(x, n_qubits=2, reps=2):
    x1, x2 = np.tanh(x) * (np.pi / 2)   # Compresión al rango [-π/2, π/2]
    for _ in range(reps):
        [H gate en cada qubit]            # Superposición
        [RZ(2·xᵢ) en cada qubit]         # Codificación de datos
        [CX + RZ(2·x₁·x₂) + CX]         # Entrelazamiento ZZ
```
 
La función `tanh` comprime las características al rango `[-π/2, π/2]` antes de codificarlas.
 
#### 3.2 Quantum Kernel
 
El kernel cuántico entre dos puntos `x` y `x'` es la **fidelidad** entre sus statevectors:
 
```
K(x, x') = |⟨φ(x)|φ(x')⟩|²
```
 
Implementado calculando el producto interno de los vectores de estado:
 
```python
def build_kernel_matrix_from_states(S1, S2):
    v1 = np.conjugate(S1[i].data)
    overlap = v1 @ S2[j].data
    K[i, j] = np.abs(overlap) ** 2
```
 
#### 3.3 Entrenamiento con tuning
 
Para cada configuración de qubits (2, 4, 8) se busca el mejor `C ∈ {0.1, 1, 10, 100}` por accuracy en test. La matriz kernel precomputada se pasa directamente al SVM de scikit-learn con `kernel="precomputed"`.
 
---
 
## Modelos comparados
 
| Modelo | Kernel | Qubits | Descripción |
|---|---|---|---|
| Linear | Lineal | — | Baseline clásico |
| RBF | RBF | — | Referencia no lineal clásica |
| Q-2 | Cuántico ZZ | 2 | Quantum Kernel mínimo |
| Q-4 | Cuántico ZZ | 4 | Kernel cuántico ampliado |
| Q-8 | Cuántico ZZ | 8 | Kernel cuántico máximo |
 
---
 
## Resultados
 
El proyecto genera para cada modelo: frontera de decisión, matriz de confusión, accuracy y F1-score. La comparativa final se muestra en un gráfico de barras agrupadas (Accuracy vs F1 por modelo).
 
Las figuras incluidas en el repo muestran visualmente cómo los modelos cuánticos consiguen fronteras de decisión no lineales similares a RBF, siendo el número de qubits el parámetro que controla la expresividad del kernel.
 
---
 
## Tecnologías
 
- Python 3
- `qiskit` — construcción de circuitos cuánticos y cálculo de statevectors
- `scikit-learn` — SVC con kernel precomputado, métricas, preprocessing
- `numpy` — álgebra lineal para el cálculo del kernel
- `matplotlib` — visualización de fronteras de decisión y matrices de confusión
- Jupyter Notebook
---
 
## Autor
 
**Guzman Ignacio Perez Ibarz**
 
Proyecto final — Computación Cuántica, Grado en Ingeniería Matemática e Inteligencia Artificial (IMAT), Comillas ICAI.
