# `dist_prep` — Probability Distribution Encoding

Circuits that encode probability distributions into quantum amplitudes on the PDF register.

`build_pdf_circuit` is the shared typed builder; `pdf_initialize` retains the
original dictionary interface. The optimizer converts the shared circuit to a
gate. Nonuniform PDFs use unitary `StatePreparation`, rather than reset-based
`Initialize`, so the circuits can be converted to gates, inverted, and controlled
for QAE. Applied to an all-zero register, they prepare amplitudes
`sqrt(pdf[scenario])`, mapping tuple entry `j` to qubit `j`. Missing scenarios
have zero amplitude; uniform preparation still uses Hadamards.

::: qiskit_impl.dist_prep.build_pdf_circuit

---

::: qiskit_impl.dist_prep.pdf_initialize

---

::: qiskit_impl.dist_prep.make_pdf_uniform

---

::: qiskit_impl.dist_prep.make_pdf_skewnormal

---

::: qiskit_impl.dist_prep.make_normal_distribution_circuit

---

::: qiskit_impl.dist_prep.make_variational_distribution_circuit
