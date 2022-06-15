from gm2fr.analysis.Model import Model
import gm2fr.constants as const
import gm2fr.calculations as calc
import numpy as np
import scipy.special as sp

# ==============================================================================

class Polynomial(Model):

  def __init__(self, degree):
    super().__init__()
    self.name = "polynomial background"
    self.degree = degree
    self.seeds = np.ones(degree + 1)

  # ============================================================================

  def function(self, f, *p):
    return np.polyval(p, f)

  # ============================================================================

  def gradient(self, f):

    result = np.zeros(shape = (self.degree + 1, len(f)))

    # df/d(p_i) = x^(N-i)
    for i in np.arange(self.degree + 1):
      result[i] = f**(self.degree - i)

    return result

# ==============================================================================

class Sinc(Model):

  def __init__(self, scale = 1, gap = 1):
    super().__init__()
    self.name = "sinc background"
    self.gap = gap*const.kHz_us
    # self.seeds = [scale, const.info["f"].magic, 1/(2*np.pi*gap*const.kHz_us)]
    self.seeds = [0.5, 0, 1]

  # ============================================================================

  def function(self, f, a, df, s):
    # return -a * np.sinc((f-fc)/s * (1/np.pi))
    return -a * calc.sinc(2*np.pi*(f-const.info["f"].magic-df), s*self.gap)

  # ============================================================================

  def gradient(self, f):

    result = np.zeros(shape = (len(self.seeds), len(f)))

    # The optimal parameters, and helper variable for brevity below.
    # a, fc, s = self.p_opt
    # x = (f-fc)/s
    #
    # # df/da
    # result[0] = self.function(f, a, fc, s) / a
    #
    # # df/d(fc)
    # result[1] = -a * (np.cos(x)/x - np.sin(x)/x**2) * (-1/s)
    #
    # # df/ds
    # result[2] = -a * (np.cos(x)/x - np.sin(x)/x**2) * (-x/s)

    a, df, s = self.p_opt
    f_term = (f - const.info["f"].magic - df)
    arg = 2*np.pi*f_term * s * self.gap

    # df/da
    result[0] = self.function(f, a, df, s) / a

    # df/d(df)
    result[1] = -a * (-s*self.gap*np.cos(arg)/f_term + np.sin(arg)/(2*np.pi*f_term**2))

    # df/ds
    result[2] = -a * self.gap * np.cos(arg)

    return result

# ==============================================================================

class Error(Model):

  def __init__(self, gap = 1):
    super().__init__()
    self.name = "error background"
    self.b = np.pi * gap * const.kHz_us
    # self.seeds = [0.5, const.info["f"].magic, 14]
    self.seeds = [0.5, 0, 14]

  # ============================================================================

  def function(self, f, a, df, s):

    result = -a / (np.pi*s) * np.exp(-(s*self.b)**2) * np.imag(
      np.exp(-2j*(f-const.info["f"].magic-df)*self.b) * sp.dawsn(-(f-const.info["f"].magic-df)/s + 1j*s*self.b)
    )

    # This model has a problem with blowing up while exploring parameter space.
    if (np.isinf(result)).any() or (np.isnan(result)).any():
      return 0
    else:
      return result

  # ============================================================================

  def gradient(self, f):

    result = np.zeros(shape = (len(self.seeds), len(f)))

    # # The optimal parameters, and helper variables for brevity below.
    # a, fc, s = self.p_opt
    # z = -(f-fc)/s + 1j*s*self.b
    # Dz = sp.dawsn(z)
    #
    # # df/da
    # result[0] = self.function(f, a, fc, s) / a
    #
    # # df/d(fc)
    # result[1] = -a / (np.pi*s) * np.exp(-(s*self.b)**2) * np.imag(
    #   np.exp(-2j*(f-fc)*self.b) * (2j*self.b*Dz + (1-2*z*Dz)/s)
    # )
    #
    # # df/ds
    # result[2] = -self.function(f, a, fc, s)/s - self.function(f, a, fc, s) * 2*s*self.b**2 \
    #   - a / (np.pi*s) * np.exp(-(s*self.b)**2) * np.imag(
    #     np.exp(-2j*(f-fc)*self.b) * (1-2*z*Dz)*((f-fc)/s**2 + 1j*self.b)
    #   )

    # The optimal parameters, and helper variables for brevity below.
    a, df, s = self.p_opt
    z = -(f-const.info["f"].magic-df)/s + 1j*s*self.b
    Dz = sp.dawsn(z)

    # df/da
    result[0] = self.function(f, a, df, s) / a

    # df/d(fc)
    result[1] = -a / (np.pi*s) * np.exp(-(s*self.b)**2) * np.imag(
      np.exp(-2j*(f-const.info["f"].magic-df)*self.b) * (2j*self.b*Dz + (1-2*z*Dz)/s)
    )

    # df/ds
    result[2] = -self.function(f, a, df, s)/s - self.function(f, a, df, s) * 2*s*self.b**2 \
      - a / (np.pi*s) * np.exp(-(s*self.b)**2) * np.imag(
        np.exp(-2j*(f-const.info["f"].magic-df)*self.b) * (1-2*z*Dz)*((f-const.info["f"].magic-df)/s**2 + 1j*self.b)
      )

    return result

# ==============================================================================

class Template(Model):

  def __init__(self, template):
    super().__init__()
    self.name = "template"
    self.template = template
    self.seeds = [1, 1, 0]

  def function(self, f, a, b, c):
    return a * self.template(b * (f - c))

  def gradient(self, f):
    return np.zeros(shape = (len(self.seeds), len(f)))
