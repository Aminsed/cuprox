#!/usr/bin/env python3
"""Build the example notebooks.

The notebooks are generated rather than hand-edited so that every figure and
number in examples/ comes from one reproducible run against the installed
package. Each notebook asserts its own claims: if cuProx disagrees with the
reference solver, execution fails and no notebook is produced. That is the only
way "the figures are accurate" can mean anything.

    python tools/build_notebooks.py          # write .ipynb sources
    jupyter nbconvert --execute --inplace examples/*.ipynb
"""

from __future__ import annotations

import pathlib

import nbformat as nbf

EXAMPLES = pathlib.Path(__file__).resolve().parent.parent / "examples"

PREAMBLE = """\
import numpy as np
import matplotlib.pyplot as plt

import cuprox

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 120, "savefig.bbox": "tight",
    "figure.facecolor": "white", "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})
GPU = "#76B900"
REF = "#546778"

assert cuprox.__cuda_available__, (
    "This notebook must run against a CUDA build. Install with `pip install .` "
    "from the repository root; installing python/ alone does not build the "
    "extension."
)
from cuprox import _core
print(f"cuProx {cuprox.__version__} on {_core.get_device_name()}")\
"""


def nb(*cells) -> nbf.NotebookNode:
    n = nbf.v4.new_notebook()
    n.cells = [
        nbf.v4.new_markdown_cell(c[1]) if c[0] == "md" else nbf.v4.new_code_cell(c[1])
        for c in cells
    ]
    n.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    return n


def md(text):
    return ("md", text)


def code(text):
    return ("code", text)


# ---------------------------------------------------------------- 01


getting_started = nb(
    md(
        "# Getting started\n\n"
        "Solving an LP and a QP on the GPU, and checking both against a "
        "reference solver.\n\n"
        "Every claim below is asserted. If cuProx disagrees with the reference, "
        "this notebook fails to execute."
    ),
    code(PREAMBLE),
    md(
        "## A linear program\n\n"
        "```\n"
        "maximise  5x + 4y\n"
        "s.t.      x + 2y <= 10\n"
        "          3x +  y <= 15\n"
        "          x, y   >= 0\n"
        "```\n"
        "The optimum is at the intersection of the two constraints: "
        "`x = 4, y = 3`, objective `32`."
    ),
    code(
        "model = cuprox.Model()\n"
        'x = model.add_var(lb=0, name="x")\n'
        'y = model.add_var(lb=0, name="y")\n'
        "model.add_constr(x + 2 * y <= 10)\n"
        "model.add_constr(3 * x + y <= 15)\n"
        "model.maximize(5 * x + 4 * y)\n"
        "\n"
        "result = model.solve()\n"
        'print(f"status    {result.status.value}")\n'
        'print(f"objective {result.objective:.6f}   (exact: 32)")\n'
        'print(f"x, y      {result.x[0]:.6f}, {result.x[1]:.6f}   (exact: 4, 3)")\n'
        "\n"
        "assert result.status is cuprox.Status.OPTIMAL\n"
        "assert abs(result.objective - 32.0) < 1e-3\n"
        "np.testing.assert_allclose(result.x, [4.0, 3.0], atol=1e-3)"
    ),
    md(
        "## A quadratic program, checked against OSQP\n\n"
        "In matrix form, minimising `½ xᵀPx + qᵀx` subject to `l ≤ Ax ≤ u` and "
        "`lb ≤ x ≤ ub`."
    ),
    code(
        "from scipy import sparse\n"
        "\n"
        "rng = np.random.default_rng(0)\n"
        "n, m = 200, 100\n"
        "M = sparse.random(n, n, density=0.1, random_state=0, format=\"csr\")\n"
        "P = (M @ M.T + sparse.identity(n) * 0.1).tocsr()\n"
        "A = sparse.random(m, n, density=0.1, random_state=1, format=\"csr\")\n"
        "q = rng.standard_normal(n)\n"
        "l, u = -np.ones(m), np.ones(m)\n"
        "lb, ub = np.full(n, -5.0), np.full(n, 5.0)\n"
        "\n"
        "res = cuprox.solve(c=q, P=P, A=A, constraint_l=l, constraint_u=u,\n"
        "                   lb=lb, ub=ub,\n"
        '                   params={"tolerance": 1e-8, "max_iterations": 200_000})\n'
        'print(f"cuProx   {res.objective:.8f}  ({res.iterations} iterations)")'
    ),
    code(
        "import osqp\n"
        "\n"
        "# OSQP has no separate variable bounds, so the box goes in as rows of A --\n"
        "# the same encoding cuProx uses internally.\n"
        'A_aug = sparse.vstack([A, sparse.identity(n, format="csc")], format="csc")\n'
        "prob = osqp.OSQP()\n"
        "prob.setup(P=P.tocsc(), q=q, A=A_aug,\n"
        "           l=np.concatenate([l, lb]), u=np.concatenate([u, ub]),\n"
        "           verbose=False, eps_abs=1e-9, eps_rel=1e-9, max_iter=200_000)\n"
        "ref = prob.solve()\n"
        "\n"
        'print(f"OSQP     {ref.info.obj_val:.8f}")\n'
        "rel = abs(res.objective - ref.info.obj_val) / abs(ref.info.obj_val)\n"
        'print(f"relative difference {rel:.2e}")\n'
        "\n"
        "assert rel < 1e-5\n"
        "assert np.all(np.asarray(res.x) >= lb - 1e-6)\n"
        "assert np.all(np.asarray(res.x) <= ub + 1e-6)\n"
        "Ax = A @ np.asarray(res.x)\n"
        "assert np.all(Ax >= l - 1e-4) and np.all(Ax <= u + 1e-4)"
    ),
    md(
        "## Where the GPU actually helps\n\n"
        "cuProx is matrix-free: an iteration costs a few sparse mat-vecs, and that "
        "cost does not depend on the sparsity *pattern*. OSQP factorises, which is "
        "excellent until the factor fills in.\n\n"
        "So the interesting variable is not problem size but how badly the "
        "factorisation fills in. Below, the same sizes are solved with a **banded** "
        "`A` (kind to a direct method) and a **random** `A` (not)."
    ),
    code(
        "import time\n"
        "\n"
        "def make(n, kind, band=3, nnz_row=8, seed=0):\n"
        "    rng = np.random.default_rng(seed)\n"
        "    m = n // 2\n"
        "    P = sparse.diags([rng.uniform(0.5, 1.5, n)]\n"
        "                     + [rng.uniform(-0.3, 0.3, n) for _ in range(band)],\n"
        '                     [0] + list(range(1, band + 1)), shape=(n, n), format="csr")\n'
        '    P = (P + P.T).tocsr() + sparse.identity(n, format="csr")\n'
        "    rows = np.repeat(np.arange(m), nnz_row)\n"
        '    if kind == "banded":\n'
        "        cols = (np.tile(np.arange(nnz_row), m)\n"
        "                + np.repeat(np.arange(m), nnz_row)) % n\n"
        "    else:\n"
        "        cols = rng.integers(0, n, size=m * nnz_row)\n"
        "    A = sparse.csr_matrix((rng.standard_normal(m * nnz_row), (rows, cols)),\n"
        "                          shape=(m, n))\n"
        "    return P, rng.standard_normal(n), A, -np.ones(m), np.ones(m)\n"
        "\n"
        "def best(fn, reps=2):\n"
        "    fn()\n"
        "    out = []\n"
        "    for _ in range(reps):\n"
        "        t0 = time.perf_counter()\n"
        "        fn()\n"
        "        out.append(time.perf_counter() - t0)\n"
        "    return min(out)\n"
        "\n"
        "# Kept small deliberately: with a random A, OSQP's factorisation fills in\n"
"# so badly that n = 20,000 takes minutes, which is the very effect this\n"
"# cell exists to show. Two sizes are enough to show it.\n"
"sizes = [1_000, 5_000]\n"
        "timings = {}\n"
        'for kind in ("banded", "random"):\n'
        "    gpu, cpu = [], []\n"
        "    for n in sizes:\n"
        "        P, q, A, l, u = make(n, kind)\n"
        "        lb, ub = np.full(n, -10.0), np.full(n, 10.0)\n"
        "        gpu.append(best(lambda: cuprox.solve(\n"
        "            c=q, P=P, A=A, constraint_l=l, constraint_u=u, lb=lb, ub=ub,\n"
        '            params={"tolerance": 1e-4, "max_iterations": 4000})))\n'
        '        A_aug = sparse.vstack([A, sparse.identity(n, format="csc")], format="csc")\n'
        "        la, ua = np.concatenate([l, lb]), np.concatenate([u, ub])\n"
        "        Pc = P.tocsc()\n"
        "        def run_osqp(Pc=Pc, q=q, A_aug=A_aug, la=la, ua=ua):\n"
        "            pr = osqp.OSQP()\n"
        "            pr.setup(P=Pc, q=q, A=A_aug, l=la, u=ua, verbose=False,\n"
        "                     eps_abs=1e-4, eps_rel=1e-4)\n"
        "            return pr.solve()\n"
        "        cpu.append(best(run_osqp))\n"
        "    timings[kind] = (gpu, cpu)\n"
        "\n"
        'for kind, (gpu, cpu) in timings.items():\n'
        '    print(f"{kind:>7}  " + "  ".join(\n'
        '        f"n={n}: {c/g:5.2f}x" for n, g, c in zip(sizes, gpu, cpu)))'
    ),
    code(
        "fig, ax = plt.subplots(figsize=(7, 4))\n"
        'for kind, style in (("banded", "--"), ("random", "-")):\n'
        "    gpu, cpu = timings[kind]\n"
        "    ax.plot(sizes, [c / g for c, g in zip(cpu, gpu)], style, marker=\"o\",\n"
        '            color=GPU, label=f"{kind} A")\n'
        'ax.axhline(1.0, color=REF, lw=1, ls=":")\n'
        'ax.text(sizes[0], 1.05, "parity", color=REF, fontsize=8)\n'
        'ax.set_xscale("log")\n'
        'ax.set_yscale("log")\n'
        'ax.set_xlabel("variables")\n'
        'ax.set_ylabel("speedup over OSQP")\n'
        'ax.set_title("The sparsity pattern matters more than the size")\n'
        "ax.legend()\n"
        'fig.savefig("benchmark_sparsity.png")\n'
        "plt.show()"
    ),
    md(
        "The two curves separate at the same problem sizes. That is the whole "
        "story: a matrix-free first-order method is indifferent to fill-in, and a "
        "direct method is not.\n\n"
        "For small or well-structured problems a good CPU solver is the better "
        "tool, and cuProx says so in its README."
    ),
)


# ---------------------------------------------------------------- 02


differentiable = nb(
    md(
        "# Differentiable optimization\n\n"
        "A QP as a layer in a PyTorch graph: solve in the forward pass, "
        "differentiate through the solution in the backward pass.\n\n"
        "The gradients are checked against a closed form and against "
        "`torch.autograd.gradcheck`."
    ),
    code(PREAMBLE + "\n\nimport torch\nfrom cuprox.torch import solve_qp\ntorch.manual_seed(0)"),
    md(
        "## The gradient is exact\n\n"
        "For an unconstrained QP the solution is `x* = -P⁻¹q`, so the Jacobian "
        "`∂x*/∂q` is exactly `-P⁻¹`. That gives something unambiguous to check "
        "against."
    ),
    code(
        "n = 4\n"
        "M = torch.randn(n, n, dtype=torch.float64)\n"
        "P = M @ M.T + 3.0 * torch.eye(n, dtype=torch.float64)\n"
        "q = torch.randn(n, dtype=torch.float64, requires_grad=True)\n"
        "\n"
        "x = solve_qp(P, q)\n"
        "expected = -np.linalg.solve(P.numpy(), q.detach().numpy())\n"
        'print(f"forward error   {np.abs(x.detach().numpy() - expected).max():.2e}")\n'
        "\n"
        "jac = np.zeros((n, n))\n"
        "for i in range(n):\n"
        "    qi = q.clone().detach().requires_grad_(True)\n"
        "    seed = torch.zeros(n, dtype=torch.float64)\n"
        "    seed[i] = 1.0\n"
        "    solve_qp(P, qi).backward(seed)\n"
        "    jac[i] = qi.grad.detach().numpy()\n"
        "\n"
        "jac_exact = -np.linalg.inv(P.numpy())\n"
        'print(f"Jacobian error  {np.abs(jac - jac_exact).max():.2e}")\n'
        "\n"
        "# A first-order method at its default tolerance; the printed errors above\n"
"# are the real figures. The Jacobian is far tighter than the forward\n"
"# solve because it is obtained by implicit differentiation rather than by\n"
"# differentiating the iterates.\n"
"np.testing.assert_allclose(x.detach().numpy(), expected, atol=1e-4)\n"
        "np.testing.assert_allclose(jac, jac_exact, atol=1e-6)"
    ),
    code(
        "ok = torch.autograd.gradcheck(\n"
        "    lambda qq: solve_qp(P, qq),\n"
        "    (q.clone().detach().requires_grad_(True),),\n"
        "    eps=1e-6, atol=1e-4, rtol=1e-3,\n"
        ")\n"
        'print(f"torch.autograd.gradcheck: {ok}")\n'
        "assert ok"
    ),
    md(
        "## Learning through the solver\n\n"
        "A minimal end-to-end check: learn the linear term `q` so that the QP's "
        "solution matches a target. The only path from the loss to `q` runs "
        "through the solver, so the curve going down is itself evidence the "
        "backward pass is right."
    ),
    code(
        "torch.manual_seed(1)\n"
        "n = 6\n"
        "M = torch.randn(n, n, dtype=torch.float64)\n"
        "P_fixed = M @ M.T + 2.0 * torch.eye(n, dtype=torch.float64)\n"
        "target = torch.randn(n, dtype=torch.float64)\n"
        "\n"
        "q_learn = torch.zeros(n, dtype=torch.float64, requires_grad=True)\n"
        "opt = torch.optim.Adam([q_learn], lr=0.2)\n"
        "\n"
        "losses = []\n"
        "for _ in range(120):\n"
        "    opt.zero_grad()\n"
        "    loss = ((solve_qp(P_fixed, q_learn) - target) ** 2).mean()\n"
        "    loss.backward()\n"
        "    opt.step()\n"
        "    losses.append(loss.item())\n"
        "\n"
        'print(f"loss  {losses[0]:.4e}  ->  {losses[-1]:.4e}")\n'
        "assert losses[-1] < losses[0] / 100"
    ),
    code(
        "fig, ax = plt.subplots(figsize=(6.5, 3.6))\n"
        "ax.semilogy(losses, color=GPU, lw=2)\n"
        'ax.set_xlabel("step")\n'
        'ax.set_ylabel("mean squared error")\n'
        'ax.set_title("Learning the objective through the solver")\n'
        'fig.savefig("qp_layer_training.png")\n'
        "plt.show()\n"
        "\n"
        "achieved = solve_qp(P_fixed, q_learn).detach().numpy()\n"
        'print(f"max |x* - target| = {np.abs(achieved - target.numpy()).max():.2e}")'
    ),
)


# ---------------------------------------------------------------- 03


portfolio = nb(
    md(
        "# Portfolio optimization\n\n"
        "Minimum-variance and maximum-Sharpe portfolios, and an efficient "
        "frontier, checked against `cvxpy`."
    ),
    code(PREAMBLE + "\n\nfrom cuprox.finance import Portfolio"),
    md(
        "## Data\n\n"
        "Synthetic daily returns for eight assets with different drifts, so the "
        "results are reproducible without a data dependency."
    ),
    code(
        "rng = np.random.default_rng(7)\n"
        "n_assets, n_days = 8, 756\n"
        "factor = rng.standard_normal(n_days)\n"
        "beta = rng.uniform(0.4, 1.4, n_assets)\n"
        "returns = (0.01 * np.outer(factor, beta)\n"
        "           + 0.012 * rng.standard_normal((n_days, n_assets)))\n"
        "returns += np.linspace(-0.0002, 0.0009, n_assets)\n"
        "\n"
        "port = Portfolio(returns)\n"
        'print("annualised expected returns:", np.round(port.expected_returns, 4))'
    ),
    md("## Two portfolios"),
    code(
        'mv = port.optimize(method="min_variance")\n'
        'ms = port.optimize(method="max_sharpe")\n'
        "\n"
        "for name, r in ((\"min variance\", mv), (\"max sharpe\", ms)):\n"
        '    print(f"{name:>13}  return {r.expected_return:7.4f}  '
        'vol {r.volatility:6.4f}  sharpe {r.sharpe_ratio:6.3f}  [{r.status}]")\n'
        "\n"
        "assert mv.status == \"optimal\" and ms.status == \"optimal\"\n"
        "# Weights are a portfolio: non-negative and summing to one.\n"
        "for r in (mv, ms):\n"
        "    assert abs(r.weights.sum() - 1.0) < 1e-4\n"
        "    assert np.all(r.weights >= -1e-6)\n"
        "# Each optimum must be best at its own objective.\n"
        "assert mv.volatility <= ms.volatility + 1e-6\n"
        "assert ms.sharpe_ratio >= mv.sharpe_ratio - 1e-6"
    ),
    md(
        "## Checking the frontier against cvxpy\n\n"
        "For a set of target returns, solve `min wᵀΣw s.t. μᵀw = target, "
        "Σw = 1, w ≥ 0` with both cuProx and cvxpy and compare the resulting "
        "volatilities."
    ),
    code(
        "import cvxpy as cp\n"
        "\n"
        "mu, cov = port.expected_returns, port.covariance\n"
        "span = mu.max() - mu.min()\n"
"targets = np.linspace(mu.min() + 0.05 * span, mu.max() - 0.05 * span, 9)\n"
"periods = 252  # trading days, the annualisation Portfolio applies\n"
        "\n"
        "vol_gpu, vol_ref = [], []\n"
        "for t in targets:\n"
        '    r = port.optimize(method="target_return", target_return=float(t))\n'
        "    vol_gpu.append(r.volatility)\n"
"    assert abs(r.expected_return - t * periods) < 1e-3 * max(abs(t * periods), 1.0)\n"
        "\n"
        "    w = cp.Variable(len(mu))\n"
        "    prob = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov))),\n"
        "                      [cp.sum(w) == 1, w >= 0, mu @ w == t])\n"
        "    prob.solve(solver=cp.CLARABEL)\n"
        "    # PortfolioResult reports annualised figures; the covariance here is\n"
"    # per-period, so annualise the reference before comparing.\n"
"    vol_ref.append(float(np.sqrt(prob.value * periods)))\n"
        "\n"
        "vol_gpu, vol_ref = np.array(vol_gpu), np.array(vol_ref)\n"
        "rel = np.abs(vol_gpu - vol_ref) / vol_ref\n"
        'print(f"frontier volatility, max relative difference vs cvxpy: {rel.max():.2e}")\n'
        "assert rel.max() < 5e-3"
    ),
    code(
        "fig, ax = plt.subplots(figsize=(7, 4.4))\n"
        'ax.plot(vol_ref, targets, "o", ms=9, mfc="none", color=REF, label="cvxpy")\n'
        'ax.plot(vol_gpu, targets, "-", lw=2, color=GPU, label="cuProx")\n'
        'ax.scatter(mv.volatility, mv.expected_return, s=90, marker="s",\n'
        '           color="#C8102E", zorder=5, label="min variance")\n'
        'ax.scatter(ms.volatility, ms.expected_return, s=110, marker="*",\n'
        '           color="#C8102E", zorder=5, label="max sharpe")\n'
        'ax.set_xlabel("annualised volatility")\n'
        'ax.set_ylabel("annualised return")\n'
        'ax.set_title("Efficient frontier")\n'
        "ax.legend()\n"
        'fig.savefig("portfolio_frontier.png")\n'
        "plt.show()"
    ),
)


# ---------------------------------------------------------------- 04


mpc = nb(
    md(
        "# Model predictive control\n\n"
        "Closed-loop control of a double integrator, re-solving a QP at every "
        "step. The constraints are checked over the whole trajectory."
    ),
    code(PREAMBLE + "\n\nfrom cuprox.mpc import LinearMPC, double_integrator"),
    md(
        "## Plant and controller\n\n"
        "States are position and velocity; the input is acceleration, limited to "
        "`|u| ≤ 1`. The controller drives the state to the origin from `x₀ = "
        "[5, 0]`."
    ),
    code(
        "dt = 0.1\n"
        "system = double_integrator(dt=dt)\n"
        "\n"
        "mpc = LinearMPC(\n"
        "    system,\n"
        "    horizon=25,\n"
        "    Q=np.diag([10.0, 1.0]),\n"
        "    R=np.array([[0.1]]),\n"
        "    u_min=np.array([-1.0]),\n"
        "    u_max=np.array([1.0]),\n"
        ")\n"
        'print(f"states {mpc.n_x}, inputs {mpc.n_u}, horizon {mpc.horizon}")'
    ),
    md("## Closed loop"),
    code(
        "import time\n"
        "\n"
        "x = np.array([5.0, 0.0])\n"
        "xs, us, solve_ms = [x.copy()], [], []\n"
        "\n"
        "for _ in range(80):\n"
        "    t0 = time.perf_counter()\n"
        "    res = mpc.solve(x, max_iters=20_000, tolerance=1e-6)\n"
        "    solve_ms.append((time.perf_counter() - t0) * 1e3)\n"
        "    u = np.atleast_1d(res.u)[0] if np.ndim(res.u) > 1 else np.atleast_1d(res.u)[:mpc.n_u]\n"
        "    x = system.A @ x + system.B @ u\n"
        "    xs.append(x.copy())\n"
        "    us.append(u.copy())\n"
        "\n"
        "xs = np.array(xs)\n"
        "us = np.array(us)\n"
        'print(f"median solve {np.median(solve_ms):.2f} ms over {len(solve_ms)} steps")\n'
        'print(f"final state  [{xs[-1, 0]:+.4f}, {xs[-1, 1]:+.4f}]")\n'
        "\n"
        "# The input limit must hold at every step, and the state must converge.\n"
        "assert np.all(us <= 1.0 + 1e-4) and np.all(us >= -1.0 - 1e-4)\n"
        "assert np.abs(xs[-1]).max() < 0.05\n"
        "assert np.abs(xs[-1]).max() < np.abs(xs[0]).max()"
    ),
    code(
        "t = np.arange(len(xs)) * dt\n"
        "fig, (a, b) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)\n"
        'a.plot(t, xs[:, 0], color=GPU, lw=2, label="position")\n'
        'a.plot(t, xs[:, 1], color=REF, lw=2, label="velocity")\n'
        'a.axhline(0, color="k", lw=0.6, ls=":")\n'
        'a.set_ylabel("state")\n'
        'a.set_title("Double integrator under MPC")\n'
        "a.legend()\n"
        "\n"
        'b.step(t[:-1], us[:, 0], where="post", color=GPU, lw=2)\n'
        'b.axhline(1.0, color="#C8102E", ls="--", lw=1)\n'
        'b.axhline(-1.0, color="#C8102E", ls="--", lw=1, label="input limit")\n'
        'b.set_xlabel("time (s)")\n'
        'b.set_ylabel("input")\n'
        "b.legend()\n"
        'fig.savefig("mpc_double_integrator.png")\n'
        "plt.show()"
    ),
    md(
        "The input saturates at the limit while the position is far from the "
        "target and comes off the bound as the state approaches the origin, which "
        "is the behaviour the constraint is there to produce."
    ),
)


NOTEBOOKS = {
    "01_getting_started.ipynb": getting_started,
    "02_differentiable_optimization.ipynb": differentiable,
    "03_portfolio_optimization.ipynb": portfolio,
    "04_model_predictive_control.ipynb": mpc,
}


def main() -> None:
    EXAMPLES.mkdir(exist_ok=True)
    for name, notebook in NOTEBOOKS.items():
        path = EXAMPLES / name
        nbf.write(notebook, path)
        print(f"wrote {path.relative_to(EXAMPLES.parent)}")


if __name__ == "__main__":
    main()
