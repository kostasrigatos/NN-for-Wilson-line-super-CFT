# Derivations

Reference material for the exact evaluation of the Curvature function
$\mathbb{C}(g)$, used by the integral constraints in
`src/constraints.py` and `src/coupling.py`.

Nothing here is reconstructible from the code alone. If you are picking this
project up, read this first — the implementation is much easier to follow
once the substitution and its validation are in hand.

**Source.** Cavaglià, Gromov, Julius, Preti, [arXiv:2203.09556](https://arxiv.org/abs/2203.09556).
Equation (2.12) defines $\mathbb{C}(g)$; Appendix A supplies the ingredients. Eq.(A.6) is the Zhukovsky map, eq.(A.10) 
are the infinite sums $S_0(x)$ and $S_1(x)$ that appear in the definition of $F[x,y]$ given in equation (A.9).

---

## 0. The problem

The Curvature function is given exactly — at any coupling, not as an
asymptotic expansion — by a double contour integral:

$$
\mathbb{C}(g) = -4\mathbb{B}^2(g) - \frac{1}{2}\oint \frac{du_x}{2\pi i}\oint \frac{du_y}{2\pi i}\; K_0(u_x - u_y)\, F[x,y]
$$

Both contours run **clockwise** around the cut $u \in [-2g, 2g]$, and the
integrand is expressed through the Zhukovsky variable

$$
u = g\left(x + \frac{1}{x}\right), \qquad |x| \ge 1 .
$$

The goal of this document is to convert this into an ordinary real double
integral that can be evaluated numerically, and to establish enough
independent checks that a bug in the implementation cannot pass silently.

**Why this matters for the project.** `coupling.py` currently computes
$\mathbb{C}(g)$ from its strong-coupling *asymptotic* series, which is only
trustworthy for $g \gtrsim 3$ — the origin of the `G_MIN = 3` restriction.
That restriction is the binding limitation on the integral constraints:
they can only be imposed where $\mathbb{C}(g)$ is known, which is the
opposite end of the coupling range from where the $C_2^2/C_3^2$ degeneracy
is worst. Equation (2.12) is exact, so implementing it removes the
restriction entirely rather than softening it.

---

## 1. The contour becomes the unit circle

### 1.1 Setup

Probe the map just outside the unit disk. Let

$$
x = (1+\varepsilon)e^{i\phi}, \qquad \varepsilon \ge 0 \ \text{small}, \quad \phi \in [0, 2\pi).
$$

Then

$$
\begin{aligned}
x &= (1+\varepsilon)(\cos\phi + i\sin\phi) \\
\frac{1}{x} &= \frac{1}{1+\varepsilon}e^{-i\phi} = \frac{1}{1+\varepsilon}(\cos\phi - i\sin\phi)
\end{aligned}
$$

Adding them inside the transformation:

$$
u = g\left[\left(1+\varepsilon + \frac{1}{1+\varepsilon}\right)\cos\phi + i\left(1+\varepsilon - \frac{1}{1+\varepsilon}\right)\sin\phi\right]
$$

### 1.2 Exact real and imaginary parts

Putting each bracket over a common denominator:

$$
(1+\varepsilon) + \frac{1}{1+\varepsilon} = \frac{(1+\varepsilon)^2 + 1}{1+\varepsilon} = \frac{\varepsilon^2 + 2\varepsilon + 2}{1+\varepsilon}
$$

$$
(1+\varepsilon) - \frac{1}{1+\varepsilon} = \frac{(1+\varepsilon)^2 - 1}{1+\varepsilon} = \frac{\varepsilon(\varepsilon+2)}{1+\varepsilon}
$$

so that, with no approximation yet,

$$
\begin{aligned}
\operatorname{Re}(u) &= g\left(\frac{\varepsilon^2 + 2\varepsilon + 2}{1+\varepsilon}\right)\cos\phi \\
\operatorname{Im}(u) &= g\left(\frac{\varepsilon(\varepsilon+2)}{1+\varepsilon}\right)\sin\phi
\end{aligned}
$$

### 1.3 Small-$\varepsilon$ expansion

Both coefficients split exactly, with no series expansion required:

$$
\frac{\varepsilon^2 + 2\varepsilon + 2}{1+\varepsilon} = \frac{2(1+\varepsilon) + \varepsilon^2}{1+\varepsilon} = 2 + \frac{\varepsilon^2}{1+\varepsilon} = 2 + \mathcal{O}(\varepsilon^2)
$$

$$
\frac{\varepsilon(\varepsilon+2)}{1+\varepsilon} = \frac{2\varepsilon(1+\varepsilon) - \varepsilon^2}{1+\varepsilon} = 2\varepsilon - \frac{\varepsilon^2}{1+\varepsilon} = 2\varepsilon + \mathcal{O}(\varepsilon^2)
$$

(Both are algebraic identities — verify by cross-multiplying. The
$\mathcal{O}(\varepsilon^2)$ statements then follow immediately.)

Hence

$$
\operatorname{Re}(u) \approx 2g\cos\phi, \qquad \operatorname{Im}(u) \approx 2g\varepsilon\sin\phi .
$$

### 1.4 Geometric picture: the contour is an ellipse

Squaring and adding, using $\cos^2\phi + \sin^2\phi = 1$:

$$
\frac{\operatorname{Re}(u)^2}{(2g)^2} + \frac{\operatorname{Im}(u)^2}{(2g\varepsilon)^2} = 1
$$

At finite $\varepsilon$ the image of the circle $|x| = 1+\varepsilon$ is a genuine
closed **ellipse** in the $u$-plane, with semi-axes $2g$ and $2g\varepsilon$.
As $\varepsilon \to 0^+$ it collapses onto the slit:

$$
u \in [-2g, 2g], \qquad \operatorname{Im}(u) = 0 .
$$

This is what justifies replacing "a contour encircling the cut" by "the unit
circle in $x$": the two are related by a continuous deformation through a
region where the integrand is analytic.

> **Do not discard $\varepsilon$ too early.** The statement
> $\operatorname{Im}(u) = 0$ is correct in the limit, but the *sign* of
> $\operatorname{Im}(u) \approx 2g\varepsilon\sin\phi$ is what identifies which
> side of the cut you are on. That information is needed in §2 and is lost
> if you set $\varepsilon = 0$ at this stage.

---

## 2. Orientation

From §1.3, with $\varepsilon > 0$ small:

| $\phi$ | $x$ | $u$ | position |
|---|---|---|---|
| $0$ | $\approx 1$ | $\approx 2g$ | right end of cut |
| $\pi/2$ | $\approx i$ | $\approx +2ig\varepsilon$ | just **above** the cut, near $u=0$ |
| $\pi$ | $\approx -1$ | $\approx -2g$ | left end of cut |
| $3\pi/2$ | $\approx -i$ | $\approx -2ig\varepsilon$ | just **below** the cut, near $u=0$ |

So increasing $\phi$ traverses: right end $\to$ along the **top**, moving
left $\to$ left end $\to$ along the **bottom**, moving right $\to$ back to
the right end.

Compare with a standard counterclockwise ellipse $(a\cos t,\, b\sin t)$ with
$t$ increasing: from the rightmost point it goes to the top, then to the
leftmost — i.e. along the top from right to left. Identical. Therefore:

$$
\phi:\ 0 \to 2\pi \ \text{(counterclockwise in } x) \quad \Longleftrightarrow \quad \text{counterclockwise around the cut in } u .
$$

The problem specifies **clockwise**, so we integrate in the direction of
decreasing $\phi$, which introduces one overall minus sign when written over
the standard domain:

$$
\oint_{\mathrm{cw}} \frac{du}{2\pi i}\, f = -\int_0^{2\pi} \frac{du}{2\pi i}\, f .
$$

> **Not a double count.** The segment $[-2g, 2g]$ is traversed twice *as a
> set of real numbers*, but this is a single circuit of the closed contour:
> out along the upper side, back along the lower. There is no compensating
> factor of $\tfrac12$. The reason is that $\phi$ and $2\pi - \phi$ give the
> same $u = 2g\cos\phi$ but *different* $x$ (complex conjugates on the unit
> circle), so $F[x,y]$ takes genuinely different values on the two passes.
> The upper and lower sides of the cut are the images of the upper and lower
> halves of the unit circle.

---

## 3. The measure

On the circle, $u = 2g\cos\phi$ exactly, so

$$
du = -2g\sin\phi\, d\phi .
$$

Substituting into the clockwise measure from §2:

$$
\oint_{\mathrm{cw}} \frac{du}{2\pi i}\, f = -\int_0^{2\pi}\left[\frac{-2g\sin\phi}{2\pi i}\right] f\, d\phi = \int_0^{2\pi}\frac{2g\sin\phi}{2\pi i}\, f\, d\phi = \frac{g}{\pi i}\int_0^{2\pi} \sin\phi \, f \, d\phi
$$

Using $1/i = -i$:

$$
\boxed{\ \oint_{\mathrm{cw}} \frac{du}{2\pi i}\, f = -i\left(\frac{g}{\pi}\right)\int_0^{2\pi} d\phi\; \sin\phi \cdot f\ }
$$

---

## 4. The assembled result

Applying §3 to both variables, the prefactors multiply:

$$
\left[-i\frac{g}{\pi}\right]\cdot\left[-i\frac{g}{\pi}\right] = i^2\frac{g^2}{\pi^2} = -\frac{g^2}{\pi^2}
$$

and the kernel argument becomes $u_x - u_y = 2g(\cos\phi_x - \cos\phi_y)$. So

$$
I \equiv \oint \frac{du_x}{2\pi i}\oint \frac{du_y}{2\pi i}\, K_0(u_x-u_y) F[x,y]
= -\frac{g^2}{\pi^2}\int_0^{2\pi}\!\!\int_0^{2\pi} d\phi_x\, d\phi_y\; \sin\phi_x \sin\phi_y\, K_0\big(2g(\cos\phi_x - \cos\phi_y)\big)\, F\big[e^{i\phi_x}, e^{i\phi_y}\big]
$$

Finally, combining with the $-\tfrac12$ prefactor in §0 — note the sign
flips, so the integral enters **positively**:

$$
\boxed{\ \mathbb{C}(g) = -4\mathbb{B}^2(g) + \frac{g^2}{2\pi^2}\int_0^{2\pi}\!\!\int_0^{2\pi} d\phi_x\, d\phi_y\; \sin\phi_x \sin\phi_y\, K_0\big(2g(\cos\phi_x - \cos\phi_y)\big)\, F\big[e^{i\phi_x}, e^{i\phi_y}\big]\ }
$$

with $u_x = 2g\cos\phi_x$ and $u_y = 2g\cos\phi_y$ wherever they appear
inside $F$.

---

## 5. Validation 1 — the measure, against known residues

A wrong substitution produces a plausible-looking but incorrect function of
$g$, and nothing about running the code reveals it. These checks are
therefore not optional.

### 5.1 What the answer should be

Invert the map on the outer sheet:

$$
x = \frac{u + \sqrt{u^2 - 4g^2}}{2g}
$$

and expand $1/x$ at large $u$:

$$
\frac{1}{x} = \frac{2g}{u\left(1 + \sqrt{1 - \frac{4g^2}{u^2}}\right)} = \frac{2g}{u\left(2 - \frac{2g^2}{u^2} + \dots\right)} = \frac{g}{u} + \mathcal{O}\!\left(\frac{1}{u^3}\right)
$$

The function $1/x$ is analytic outside the cut, so a counterclockwise
contour around the cut may be deformed out to a large circle, where the
integral picks up the coefficient of $1/u$, namely $g$. Clockwise therefore
gives $-g$.

Higher powers $x^{-n}$, $n \ge 2$, start at $\mathcal{O}(u^{-2})$ or beyond
and have no $1/u$ term, so their integrals vanish.

### 5.2 Checking the parametrisation reproduces these

**Case $f = 1/x = e^{-i\phi}$.** Using $\sin\phi = (e^{i\phi}-e^{-i\phi})/2i$:

$$
\begin{aligned}
-i\frac{g}{\pi}\int_0^{2\pi} \sin\phi\, e^{-i\phi}\, d\phi
&= -i\frac{g}{\pi}\int_0^{2\pi}\left(\frac{e^{i\phi}-e^{-i\phi}}{2i}\right)e^{-i\phi} d\phi \\
&= -\frac{g}{2\pi}\int_0^{2\pi}\left(1 - e^{-2i\phi}\right) d\phi \\
&= -\frac{g}{2\pi}\Big[2\pi - 0\Big] \;=\; -g \quad \checkmark
\end{aligned}
$$

**Case $f = 1/x^n = e^{-in\phi}$, $n \ge 2$.**

$$
-i\frac{g}{\pi}\int_0^{2\pi}\left(\frac{e^{i\phi}-e^{-i\phi}}{2i}\right)e^{-in\phi} d\phi
= -\frac{g}{2\pi}\int_0^{2\pi}\left(e^{-i(n-1)\phi} - e^{-i(n+1)\phi}\right) d\phi
$$

For $n \ge 2$ both exponents $n-1$ and $n+1$ are non-zero integers, and a
pure complex exponential integrated over a full period gives zero. Hence the
result is $0$. $\checkmark$

> **Scope.** The vanishing argument requires *both* $n-1 \neq 0$ and
> $n+1 \neq 0$, so it covers $n \ge 2$ and $n = 0$, but **not** $n = -1$.
> At $n = -1$ (i.e. $f = x$) the second exponent vanishes and the integral
> gives $+g$ — which is correct: from $x + 1/x = u/g$ and
> $1/x = g/u + g^3/u^3 + \dots$ one finds $x = u/g - g/u - \dots$, whose
> $1/u$ coefficient is $-g$, so clockwise gives $+g$. The formula handles
> this case correctly; only the blanket statement "$n \neq 1$" would be
> wrong.

Numerically (uniform grid, $N = 2\times10^5$, $g = 0.7$): $1/x \to -0.7000000000$,
$1/x^2 \to 0$, $1/x^3 \to 0$, all to ten decimal places.

---

## 6. Validation 2 — pole cancellation on the contour

$F[x,y]$ contains denominators $(x^2-1)$ and $(y^2-1)$, which vanish at
$x = \pm 1$ and $y = \pm 1$ — that is, at $\phi = 0, \pi$, **on the
integration contour**. Naively this would make the integral singular.

On the unit circle,

$$
x^2 - 1 = e^{2i\phi} - 1 = e^{i\phi}\left(e^{i\phi} - e^{-i\phi}\right) = 2i\, e^{i\phi} \sin\phi
$$

using $e^{i\phi} - e^{-i\phi} = 2i\sin\phi$. Therefore

$$
\boxed{\ \frac{\sin\phi}{x^2-1} = \frac{\sin\phi}{2i\,e^{i\phi}\sin\phi} = \frac{1}{2i\,e^{i\phi}} = -\frac{i}{2x}\ }
$$

**The $\sin\phi$ supplied by the measure exactly cancels the pole.** The
combination is $-\tfrac{i}{2}e^{-i\phi}$, which is smooth and bounded on the
whole interval. Each variable's pole is cancelled by its own measure factor,
so the same works for $(y^2-1)$ via $\sin\phi_y$.

This is also strong evidence that the substitution in §1–§3 is right: a
wrong parametrisation would not produce this cancellation.

> **Two conditions to confirm against the PDF.** The cancellation is
> first-order and per-factor. It works only if (i) no term in (A.9) carries
> $(x^2-1)$ or $(y^2-1)$ to a power greater than one, and (ii) every
> $1/(y^2-1)$ appears alongside its own $\sin\phi_y$ from the $y$ measure
> (which it does automatically, since the measure factorises). Verify (i)
> term by term — a squared denominator would survive and need separate
> treatment.

---

## 7. The kernel $K_0$

### 7.1 Derivation

$$
K_0(u) = \frac{\partial}{\partial u}\ln\left[\frac{\Gamma(iu+1)}{\Gamma(-iu+1)}\right]
$$

Split the logarithm:

$$
K_0(u) = \frac{\partial}{\partial u}\ln\Gamma(iu+1) - \frac{\partial}{\partial u}\ln\Gamma(-iu+1)
$$

With $\psi(z) = \frac{d}{dz}\ln\Gamma(z)$ and the chain rule
$\frac{d}{du}\ln\Gamma(au+b) = a\,\psi(au+b)$:

$$
\begin{aligned}
\frac{\partial}{\partial u}\ln\Gamma(iu+1) &= i\,\psi(iu+1) \\
\frac{\partial}{\partial u}\ln\Gamma(-iu+1) &= -i\,\psi(-iu+1)
\end{aligned}
$$

Substituting — note the two minus signs on the second term combine into an
addition:

$$
K_0(u) = \big[i\,\psi(iu+1)\big] - \big[-i\,\psi(-iu+1)\big] = i\,\psi(iu+1) + i\,\psi(-iu+1)
$$

$$
\boxed{\ K_0(u) = i\big[\psi(1+iu) + \psi(1-iu)\big]\ }
$$

`scipy.special.digamma` accepts complex arguments, so this is a direct
one-line implementation.

### 7.2 Property: purely imaginary on the real axis

For real $u$, the arguments $1+iu$ and $1-iu$ are complex conjugates. The
digamma function is real-analytic, so $\psi(\bar z) = \overline{\psi(z)}$,
and therefore

$$
\psi(1+iu) + \psi(1-iu) = 2\operatorname{Re}\psi(1+iu) \in \mathbb{R}
$$

Hence $K_0(u)$ is **purely imaginary** for real $u$. Any non-zero real part
in an implementation indicates a dropped factor of $i$ or a sign error.

### 7.3 Property: even

$$
K_0(-u) = i\big[\psi(1-iu) + \psi(1+iu)\big] = K_0(u)
$$

So $K_0$ is **even**, and the kernel $K_0(u_x - u_y)$ is symmetric under
$x \leftrightarrow y$. Useful both as a structural check on the assembled
integrand and, potentially, to halve the quadrature work.

### 7.4 Value at the origin

Since $\psi(1) = -\gamma_E$:

$$
K_0(0) = i\big[\psi(1) + \psi(1)\big] = -2i\gamma_E \approx -1.154431329803\,i
$$

Finite — so the coincident point $u_x = u_y$ (which occurs on the diagonal
$\phi_x = \pm\phi_y$ of the integration domain) is not singular.

Numerically confirmed: real part exactly zero at a range of $u$;
$K_0(0)$ matching $-2i\gamma_E$ to twelve digits; $K_0(u) - K_0(-u) = 0$
identically.

---

## 8. Numerical evaluation

The integrand is **smooth and $2\pi$-periodic**, integrated over exactly one
full period in each variable. In that setting the uniform **trapezoidal rule
converges exponentially** and outperforms Gauss–Legendre.

Benchmark on $\int_0^{2\pi} e^{\cos\phi}\,d\phi = 2\pi I_0(1)$:

| $N$ | trapezoid (periodic) | Gauss–Legendre |
|---|---|---|
| 8 | 1.3e-06 | 1.7e-04 |
| 16 | 0.0e+00 | 1.7e-11 |
| 24 | 8.9e-16 | 7.1e-15 |
| 32 | 0.0e+00 | 8.9e-15 |

Machine precision at $N = 16$ versus $N = 24$. For a 2D tensor product that
is $16^2 = 256$ evaluations of $F[x,y]$ instead of $24^2 = 576$ — and
$F$ is the expensive part.

**Recommended scheme.** Uniform grid
`phi = np.linspace(0, 2*np.pi, N, endpoint=False)`, equal weights
$2\pi/N$, tensor product over $(\phi_x, \phi_y)$. Establish convergence by
doubling $N$ until the result stops moving; exponential convergence makes
this fast and unambiguous.

---

## 9. Implementation checklist

In dependency order, each step with its own check:

1. **$K_0(u)$** — §7. Check: purely imaginary for real $u$; $K_0(0) = -2i\gamma_E$; even.
2. **$S_0(x)$, $S_1(x)$** — infinite Bessel sums, eq. (A.10). These sum
   $I_n(4\pi g)$ over many orders $n$ at fixed large argument. Work entirely
   in `scipy.special.ive`-scaled quantities and let the shared $e^{4\pi g}$
   factors cancel algebraically, exactly as `B_function` and `F_function` in
   `coupling.py` already do — do **not** reconstruct the true $I_n$ before
   summing, which overflows immediately near $g = 4$. Check: partial sums
   stabilise; no overflow or `NaN` anywhere in the intended range.
3. **$F[x,y]$** — eq. (A.9). Long but mechanical once (1) and (2) exist.
   Too large to unit-test in isolation; its test is step (5).
4. **The double integral** — §4 and §8.
5. **Validation against the trusted asymptotic series.** Compare
   `curvature_C_exact(g)` against the existing asymptotic
   `curvature_C(g)` at $g = 3, 3.5, 4$, where the series is already trusted.
   **This is the load-bearing test.** Agreement confirms §1–§7 and the
   implementation together, before the result is ever used at a new
   coupling. Disagreement means stop and debug — do not proceed to (6).
6. **Extend downward.** Recompute `RHS1`/`RHS2` with the exact
   $\mathbb{C}(g)$, drop the `g >= G_MIN` restriction, and re-run the
   constraint sweeps across the full coupling range — including where the
   $C_2^2/C_3^2$ degeneracy is worst.

> **Caveat on (A.9).** The machine-readable version of this equation
> available online contains at least one visible transcription error (a
> `sinh(2*pi*u)` with its subscript dropped). Work from the PDF for this
> equation specifically.

---

## Summary of results

| Quantity | Result |
|---|---|
| Contour | Unit circle $x = e^{i\phi}$; $u = 2g\cos\phi$ |
| Measure | $\oint_{\mathrm{cw}}\frac{du}{2\pi i} f = -i\frac{g}{\pi}\int_0^{2\pi} d\phi \sin\phi\, f$ |
| Double-integral prefactor | $-g^2/\pi^2$, becoming $+g^2/2\pi^2$ in $\mathbb{C}(g)$ |
| Pole cancellation | $\dfrac{\sin\phi}{x^2-1} = -\dfrac{i}{2x}$ |
| Kernel | $K_0(u) = i[\psi(1+iu)+\psi(1-iu)]$; purely imaginary, even, $K_0(0) = -2i\gamma_E$ |
| Quadrature | Periodic trapezoidal rule, $N \approx 16$–32 per dimension |
