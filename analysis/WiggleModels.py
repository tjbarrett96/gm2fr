from gm2fr.analysis.Model import Model
import numpy as np
import gm2fr.constants as const

# ==============================================================================

class TwoParameter(Model):

  def __init__(self):
    super().__init__()
    self.name = "two-parameter wiggle"
    self.names = ["N", "tau"]
    self.units = ["",  "us" ]
    self.seeds = [1,   64.4 ]
    self.bounds = (
      [0,      0 ],
      [np.inf, 70]
    )

  # ============================================================================

  def two(self, t, N, tau):
    return N * np.exp(-t/tau)

  def function(self, t, N, tau):
    return self.two(t, N, tau)

  # ============================================================================

  def gradient(self, t):

    result = np.zeros(shape = (2, len(t)))

    # Optimal parameters.
    N, tau = self.p_opt[:2]

    # df/dN
    result[0] = self.two(t, N, tau) / N

    # df/d(tau)
    result[1] = self.two(t, N, tau) * (t/tau**2)

    return result

# ==============================================================================

class FiveParameter(TwoParameter):

  def __init__(self):
    super().__init__()
    self.name = "five-parameter wiggle"
    self.names += ["A", "f_a", "phi_a"]
    self.units += ["",  "MHz", "rad"  ]
    self.seeds += [0.3, 0.229, np.pi  ]
    self.bounds[0].extend([-1.0,      0.2288,    0      ])
    self.bounds[1].extend([ 1.0,      0.2292,    2*np.pi])

  # ============================================================================

  # Extra part beyond the two-parameter model.
  def five(self, t, A, fa, phi):
    return 1 + A * np.cos(2 * np.pi * fa * t + phi)

  # Five-parameter model is the two-parameter model times the extra part.
  def function(self, t, N, tau, A, fa, phi):
    return super().function(t, N, tau) * self.five(t, A, fa, phi)

  # ============================================================================

  def gradient(self, t):

    result = np.zeros(shape = (5, len(t)))

    # Optimal parameters, and helper variable for the two-parameter result.
    N, tau, A, fa, phi = self.p_opt[:5]
    two = super().function(t, N, tau)

    # df/dN and df/d(tau) same as before, with new part tacked on.
    result[:2] = super().gradient(t) * self.five(t, A, fa, phi)

    # df/dA
    result[2] = two * np.cos(2*np.pi*fa*t + phi)

    # df/d(fa)
    result[3] = two * (-A)*np.sin(2*np.pi*fa*t + phi)*(2*np.pi*t)

    # df/d(phi)
    result[4] = two * (-A)*np.sin(2*np.pi*fa*t + phi)

    return result

  # ============================================================================

  # Discard the omega_a fit value.
  def results(self, prefix = "fit", parameters = True):
    results = super().results(prefix, parameters)
    results.table.drop(f"{prefix}_f_a", axis = "columns", inplace = True)
    return results

# ==============================================================================

class NineParameter(FiveParameter):

  def __init__(self, n = 0.108):
    super().__init__()
    self.name = "nine-parameter wiggle"
    fcbo = (1 - np.sqrt(1 - n)) * const.info["f"].magic * 1E-3
    self.names += ["tau_cbo", "A_cbo", "f_cbo", "phi_cbo"]
    self.units += ["us",      "",      "MHz",   "rad"    ]
    self.seeds += [150,       0.005,   fcbo,    np.pi    ]
    self.bounds[0].extend([0,   0,    0.300, 0      ])
    self.bounds[1].extend([500, 0.01, 0.450, 2*np.pi])

  # ============================================================================

  # Extra part beyond the five-parameter model.
  def nine(self, t, tcbo, Acbo, fcbo, pcbo):
    return 1 + np.exp(-t/tcbo) * Acbo * np.cos(2 * np.pi * fcbo * t + pcbo)

  # Nine-parameter model is the five-parameter model times the extra part.
  def function(self, t, N, tau, A, fa, phi, tcbo, Acbo, fcbo, pcbo):
    return super().function(t, N, tau, A, fa, phi) * self.nine(t, tcbo, Acbo, fcbo, pcbo)

  # ============================================================================

  def gradient(self, t):

    result = np.zeros(shape = (9, len(t)))

    # Optimal parameters, and helper variable for five-parameter result.
    N, tau, A, fa, phi, tcbo, Acbo, fcbo, pcbo = self.p_opt[:9]
    five = super().function(t, N, tau, A, fa, phi)

    # First five derivatives same as before, with new part tacked on.
    result[:5] = super().gradient(t) * self.nine(t, tcbo, Acbo, fcbo, pcbo)

    # df/d(tcbo)
    result[5] = five * np.exp(-t/tcbo) * (t/tcbo**2) * Acbo * np.cos(2 * np.pi * fcbo * t + pcbo)

    # df/d(Acbo)
    result[6] = five * np.exp(-t/tcbo) * np.cos(2 * np.pi * fcbo * t + pcbo)

    # df/d(fcbo)
    result[7] = five * np.exp(-t/tcbo) * (-Acbo)*np.sin(2*np.pi*fcbo*t + pcbo)*(2*np.pi*t)

    # df/d(pcbo)
    result[8] = five * np.exp(-t/tcbo) * (-Acbo)*np.sin(2*np.pi*fcbo*t + pcbo)

    return result

# ==============================================================================
