# Purpose:

- Optimize for stiffness

# Phases:

0. Problem definition and setup
   - Define project goals and phases
   - Set up project timelines
   - General research and background knowledge
1. B-spline geometry
   - Configure parametric definition of 3D geometry for shank strut
   - Input control points and output 3D geometry mathematically
   - **Optional**: Add preview of geometry
   - Timeline: Jun 5 - Jun 19
2. Meshing
   - Automatic meshing for arbitrary parametric definition of shank strut
   - Definition of constraints and loads for arbitrary shank strut
   - Timeline: Jun 22 - Jun 30
3. Baseline FEA
   - Solve FEA problem, max stress, strain, and deflection
   - Composite failure criteria
   - Auto check for convergence
   - **Optional**: Auto mesh refinement to satisfy convergence criteria
   - Timeline: Jul 1 - Jul 17
4. Direct optimization
   - Define and solve for gradient of each parameter
   - Redefine geometry, remesh, resolve FEA
   - Define optimization criteria
   - Repeat until optimization criteria met
   - Timeline: Jul 20 - Jul 31
5. Dataset generation
   - Create dataset of 500-2000 analyzed designs to train model
   - Aug 3 - Aug 7
6. Surrogate modelling
   - Use surrogate modelling exclusively to optimize design
   - **Optional**: Compare model efficacy between different datasets
   - Timeline: Aug 10 - Aug 19
7. Surrogate-assisted optimization
   - Combine surrogate modelling with traditional gradient optimization
   - Timeline: Aug 20 - Aug 28
8. Report
   - Compare approaches and their ability to be generalized
   - Timeline: Aug 31 - Sep 11

# Phase 1: B-Spline Geometry

The purpose of this phase of the project is to be able to input a starting design for the shank strut by inputing control points. Though it will be used to give the optimization process a starting point, by the end of this phase, a given design should be able to be translated to it's mathematical definition and visualized. Additionally, forces and any fixed items will be defined in the context of the parametric definition.

## Deliverables

- Control point CSV
- Forces CSV
- Fixed point CSV
- Translation to 3D geometry
- Visualization of 3D geometry

## Useful Resources

## Notes

- TODO: Have no implemented any checks for axis mode on hole definitions

# Recommended structure:

```text
shank-strut-bspline-mdo/
│
├── README.md
├── pyproject.toml
├── environment.yml
├── .gitignore
│
├── docs/
│ ├── problem_definition.md
│ ├── modeling_assumptions.md
│ ├── optimization_formulation.md
│ └── final_report.md
│
├── data/
│ ├── baseline_geometry.json
│ ├── material_properties.json
│ └── strut_design_dataset.csv
│
├── examples/
│ ├── generate_baseline.py
│ ├── mesh_baseline.py
│ ├── analyze_baseline.py
│ ├── run_direct_optimization.py
│ └── run_surrogate_optimization.py
│
├── src/
│ └── exo_strut_mdo/
│ ├── geometry/
│ │ ├── bspline_strut.py
│ │ ├── constraints.py
│ │ └── export_geometry.py
│ │
│ ├── mesh/
│ │ ├── generate_mesh.py
│ │ └── mesh_checks.py
│ │
│ ├── analysis/
│ │ ├── structural_solver.py
│ │ ├── load_cases.py
│ │ └── postprocess.py
│ │
│ ├── composites/
│ │ ├── laminate_properties.py
│ │ └── failure_criteria.py
│ │
│ ├── optimization/
│ │ ├── objectives.py
│ │ ├── constraints.py
│ │ ├── direct_opt.py
│ │ └── surrogate_opt.py
│ │
│ ├── sampling/
│ │ └── design_sampler.py
│ │
│ ├── surrogates/
│ │ ├── train.py
│ │ ├── evaluate.py
│ │ └── models.py
│ │
│ └── utils/
│ ├── io.py
│ └── plotting.py
│
├── results/
│ ├── baseline/
│ ├── direct_optimization/
│ ├── surrogate_optimization/
│ └── final_comparison/
│
├── figures/
│
└── tests/
├── test_geometry.py
├── test_materials.py
├── test_failure_criteria.py
└── test_optimization_constraints.py
```
