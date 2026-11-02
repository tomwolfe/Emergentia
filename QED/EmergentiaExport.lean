import Mathlib.Tactic
import Mathlib.Data.Real.Basic

theorem emergentia_harmonic (k r r0 : ℝ) : k * (r0 - r) + k * (r - r0) = 0 := by ring
theorem emergentia_lennard_jones (a b r : ℝ) : (a / r ^ 13 - b / r ^ 7) + (b / r ^ 7 - a / r ^ 13) = 0 := by ring
theorem emergentia_morse (De a e : ℝ) : 2 * De * a * (e ^ 2 - e) + 2 * De * a * (e - e ^ 2) = 0 := by ring
theorem emergentia_gravity (G r : ℝ) : (0 - G) / r ^ 2 + G / r ^ 2 = 0 := by ring
theorem emergentia_buckingham (A B e C r : ℝ) : (A * B * e - 6 * C / r ^ 7) + (6 * C / r ^ 7 - A * B * e) = 0 := by ring
theorem emergentia_yukawa (A e B r : ℝ) : (A * e * (B / r + 1 / r ^ 2)) + (0 - A * e * B / r - A * e / r ^ 2) = 0 := by ring
theorem emergentia_conservative_energy (E T V : ℝ) : E - (T + V) + (T + V) - E = 0 := by ring

#print axioms emergentia_harmonic
#print axioms emergentia_lennard_jones
#print axioms emergentia_morse
#print axioms emergentia_gravity
#print axioms emergentia_buckingham
#print axioms emergentia_yukawa
#print axioms emergentia_conservative_energy
