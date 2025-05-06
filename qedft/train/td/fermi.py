"""This file contains the Fermi-Dirac distribution functions and the
loss function for the chemical potential.

Different implementations of the Fermi-Dirac distribution are available.
The most stable implementation is get_fractional_occupations_jax.
"""

import jax
import jax.numpy as jnp


@jax.jit
def fermi_function4entropy(
    energy: float | jnp.ndarray,
    mu: float,
    theta: float,
) -> float | jnp.ndarray:
    """Fermi-Dirac distribution function for entropy calculations.

    Args:
        energy: Orbital energy or array of energies.
        mu: Chemical potential.
        theta: Temperature parameter.

    Returns:
        Fermi-Dirac occupation probability (0 to 1).
    """
    return 1 / (1 + jnp.exp((energy - mu) / theta))


@jax.jit
def fermi_function(
    energy: float | jnp.ndarray,
    mu: float,
    theta: float,
) -> float | jnp.ndarray:
    """Fermi-Dirac distribution for restricted orbital occupation.

    Accounts for two electrons per orbital in restricted calculations.

    Args:
        energy: Orbital energy or array of energies.
        mu: Chemical potential.
        theta: Temperature parameter.

    Returns:
        Orbital occupation (0 to 2).
    """
    return 2 / (1 + jnp.exp((energy - mu) / theta))


@jax.jit
def loss_fermi_function(
    mu: float,
    mo_energy: jnp.ndarray,
    N: int,
    theta: float,
) -> float:
    """Loss function for optimizing chemical potential.

    Computes squared difference between target electron count and
    sum of Fermi-Dirac occupations.

    Args:
        mu: Chemical potential.
        mo_energy: Array of orbital energies.
        N: Target number of electrons.
        theta: Temperature parameter.

    Returns:
        Squared error between target and actual electron count.
    """
    fermi_val = 0.0
    for energy in mo_energy:
        fermi_val += fermi_function(energy, mu, theta)
    loss = (N - fermi_val) ** 2
    return loss


@jax.jit
def make_occ_fermi(
    mu: float,
    mo_energy: jnp.ndarray,
    theta: float,
) -> jnp.ndarray:
    """Generate orbital occupations using Fermi-Dirac distribution.

    Args:
        mu: Chemical potential.
        mo_energy: Array of orbital energies.
        theta: Temperature parameter.

    Returns:
        Array of orbital occupations.
    """
    list_occ = []
    for energy in mo_energy:
        list_occ.append(fermi_function(energy, mu, theta))
    return jnp.array(list_occ)


@jax.jit
def entropy(fermi_distr: float | jnp.ndarray) -> float | jnp.ndarray:
    """Calculate electronic entropy from Fermi-Dirac distribution.

    Args:
        fermi_distr: Fermi-Dirac occupation probability (0 to 1).

    Returns:
        Entropy contribution.
    """
    return fermi_distr * jnp.log(fermi_distr) + (1 - fermi_distr) * jnp.log(
        1 - fermi_distr,
    )


@jax.jit
def etropic_energy_contribution(
    mu: float,
    mo_energy: jnp.ndarray,
    theta: float,
) -> float:
    """Calculate entropic energy contribution.

    Args:
        mu: Chemical potential.
        mo_energy: Array of orbital energies.
        theta: Temperature parameter.

    Returns:
        Total entropic energy contribution.
    """
    list_entr = []
    for energy in mo_energy:
        # factor 2 because we need to add the entropy per electron
        # and each MO can have 2 with the same probability.
        list_entr.append(2 * entropy(fermi_function4entropy(energy, mu, theta)))
    return theta * sum(list_entr)


grad_loss_fermi_function = jax.jit(jax.grad(loss_fermi_function))
gradgrad_loss_fermi_function = jax.jit(jax.grad(grad_loss_fermi_function))


def optimize_fermi_occupations(
    mo_energy: jnp.ndarray,
    n_electrons: float,
    theta: float = 0.04,
    mu_initial: float | None = None,
    mu_shift: float = 0.001,
    step_grad: float = 0.6,
    max_steps: int = 100,
    tol: float = 1e-9,
) -> tuple[jnp.ndarray, float, float, float]:
    """Optimize Fermi-Dirac fractional occupations for molecular orbitals.

    This function implements fractional occupation for molecular orbitals using
    the Fermi-Dirac distribution. It optimizes the chemical potential (mu) to
    achieve the target number of electrons.

    Args:
        mo_energy: Array of molecular orbital energies
        n_electrons: Target number of electrons
        theta: Temperature parameter (default: 0.04)
        mu_initial: Initial guess for chemical potential (default: None)
        mu_shift: Shift applied to lowest orbital energy if mu_initial is None (default: 0.001)
        step_grad: Step size for gradient descent (default: 0.6)
        max_steps: Maximum number of optimization steps (default: 100)
        tol: Convergence tolerance for the loss (default: 1e-9)

    Returns:
        Tuple containing:
            - mo_occ: Array of optimized occupation numbers
            - energy_entr: Entropic energy contribution
            - mu_opt: Optimized chemical potential
            - loss_final: Final loss value

    Raises:
        ValueError: If optimization fails to converge.
    """
    # Set initial chemical potential
    if mu_initial is None:
        mu_min = mo_energy[0] + mu_shift
    else:
        mu_min = mu_initial

    # Optimize chemical potential
    loss_fermi = 100.0  # Initial loss value
    for i in range(max_steps):
        grad_fermi = grad_loss_fermi_function(mu_min, mo_energy, n_electrons, theta=theta)
        gradgrad_fermi = gradgrad_loss_fermi_function(mu_min, mo_energy, n_electrons, theta)

        # Avoid division by zero
        safe_gradgrad = jnp.where(
            jnp.abs(gradgrad_fermi) < 1e-10,
            1e-10 * jnp.sign(gradgrad_fermi),
            gradgrad_fermi,
        )

        # Update mu using Newton's method
        mu_min = mu_min - step_grad * grad_fermi / safe_gradgrad
        loss_fermi = loss_fermi_function(mu_min, mo_energy, n_electrons, theta)

        # Check convergence
        if loss_fermi < tol:
            break

    # Check if optimization was successful
    if loss_fermi > 1e-2:
        raise ValueError(
            f"Fermi occupation optimization failed: loss={loss_fermi}. "
            "Try increasing max_steps or adjusting parameters.",
        )

    # Calculate occupation numbers and entropic contribution
    mo_occ = make_occ_fermi(mu_min, mo_energy, theta=theta)
    energy_entr = etropic_energy_contribution(mu_min, mo_energy, theta)

    return mo_occ, energy_entr, mu_min, loss_fermi


def get_fractional_occupations(
    mo_energy: jnp.ndarray,
    n_electrons: float,
    frac_mo_occ: int | jnp.ndarray | None = 1,
    frac_theta: float = 0.04,
    frac_mu: float | None = None,
    frac_mu_shift: float = 0.001,
    frac_step_grad: float = 0.6,
    frac_max_steps: int = 100,
) -> tuple[jnp.ndarray, float]:
    """Get fractional occupation numbers for molecular orbitals.

    This is a wrapper function that handles different ways to specify occupation:
    1. Fermi-Dirac optimization (frac_mo_occ = 1)
    2. Direct specification of occupation numbers (frac_mo_occ = numpy array)
    3. Integer occupation (frac_mo_occ = None)

    Args:
        mo_energy: Array of molecular orbital energies
        n_electrons: Number of electrons
        frac_mo_occ: Specifies how to determine occupations:
                    - 1: Use Fermi-Dirac distribution (default)
                    - numpy array: Direct specification of occupation numbers
                    - None: Integer occupation (no fractional occupation)
        frac_theta: Temperature parameter for Fermi-Dirac (default: 0.04)
        frac_mu: Initial chemical potential (default: None)
        frac_mu_shift: Shift for chemical potential initialization (default: 0.001)
        frac_step_grad: Step size for gradient descent (default: 0.6)
        frac_max_steps: Maximum iterations for optimization (default: 100)

    Returns:
        Tuple containing:
            - mo_occ: Array of occupation numbers
            - energy_entr: Entropic energy contribution (0.0 if not using Fermi-Dirac)
    """
    if frac_mo_occ == 1:
        # Use Fermi-Dirac distribution
        mo_occ, energy_entr, mu_opt, loss_final = optimize_fermi_occupations(
            mo_energy=mo_energy,
            n_electrons=n_electrons,
            theta=frac_theta,
            mu_initial=frac_mu,
            mu_shift=frac_mu_shift,
            step_grad=frac_step_grad,
            max_steps=frac_max_steps,
        )

    elif isinstance(frac_mo_occ, jnp.ndarray):
        # Use pre-specified occupation numbers
        mo_occ = frac_mo_occ
        energy_entr = 0.0

    elif frac_mo_occ is None:
        # Integer occupation (this assumes a function mf.get_occ exists)
        # In this standalone function, we'll just use a simple filling scheme
        # Fill orbitals from lowest energy up with 2 electrons each
        mo_occ = jnp.zeros_like(mo_energy)
        remaining = n_electrons
        for i in range(len(mo_energy)):
            if remaining >= 2.0:
                mo_occ = mo_occ.at[i].set(2.0)
                remaining -= 2.0
            elif remaining > 0.0:
                mo_occ = mo_occ.at[i].set(remaining)
                remaining = 0.0
            else:
                break
        energy_entr = 0.0

    else:
        raise ValueError(
            f"frac_mo_occ={frac_mo_occ} not implemented. "
            "Use None, 1 (for Fermi-Dirac), or a jnp.ndarray.",
        )

    return mo_occ, energy_entr, mu_opt, loss_final


@jax.jit
def get_fractional_occupations_jax(
    mo_energy,
    n_electrons,
    theta,
    frac_mu,
    frac_mu_shift,
    frac_step_grad,
    frac_max_steps,
):
    """Differentiable implementation of Fermi-Dirac fractional occupation using jax.lax.scan.

    This implementation is designed to be fully differentiable by avoiding hard conditionals
    and using fixed iteration counts.

    Args:
        mo_energy: Array of molecular orbital energies
        n_electrons: Target number of electrons
        theta: Temperature parameter for Fermi-Dirac distribution
        frac_mu: Initial chemical potential (if None, derived from mo_energy)
        frac_mu_shift: Shift applied to lowest orbital energy if frac_mu is None
        frac_step_grad: Step size for Newton-like updates
        frac_max_steps: Number of optimization iterations

    Returns:
        Tuple containing:
            - mo_occ: Array of occupation numbers
            - energy_entr: Entropic energy contribution
            - mu_final: Final chemical potential
            - loss_final: Final loss value
    """
    # Set initial chemical potential in a differentiable way
    mu_from_energy = mo_energy[0] + frac_mu_shift

    # Simply use the provided value or calculate from energy
    # We don't need isnan check since we handle None outside JIT
    mu_init = mu_from_energy if frac_mu is None else frac_mu

    # Define the optimization step function without early stopping
    def fermi_step(carry, _):
        mu, _ = carry
        grad_fermi = grad_loss_fermi_function(
            mu,
            mo_energy,
            n_electrons,
            theta=theta,
        )
        gradgrad_fermi = gradgrad_loss_fermi_function(
            mu,
            mo_energy,
            n_electrons,
            theta,
        )

        # Smooth handling of small second derivatives
        safe_gradgrad = jnp.sign(gradgrad_fermi) * jnp.maximum(jnp.abs(gradgrad_fermi), 1e-10)

        # Update using a Newton-like step
        mu_new = mu - frac_step_grad * grad_fermi / safe_gradgrad
        loss_new = loss_fermi_function(mu_new, mo_energy, n_electrons, theta)

        return (mu_new, loss_new), None

    # Use lax.fori_loop instead of scan with arange
    # This avoids the need for a concrete frac_max_steps
    def body_fun(i, carry):
        return fermi_step(carry, None)[0]

    init_carry = (mu_init, jnp.array(1.0))
    final_carry = jax.lax.fori_loop(
        0,  # lower
        100,  # Upper bound - use a fixed maximum value
        lambda i, carry: jax.lax.cond(
            i < frac_max_steps,  # Only perform iterations up to frac_max_steps
            lambda _: body_fun(i, carry),
            lambda _: carry,
            None,
        ),
        init_carry,
    )

    mu_final, loss_final = final_carry

    # Calculate occupation with the optimized mu
    mo_occ = make_occ_fermi(mu_final, mo_energy, theta=theta)

    # Calculate entropy contribution
    energy_entr = etropic_energy_contribution(mu_final, mo_energy, theta)

    return mo_occ, energy_entr, mu_final, loss_final


@jax.jit
def get_fractional_occupations_jax_stable(
    mo_energy,
    n_electrons,
    theta,
    frac_mu,
    frac_mu_shift,
    frac_step_grad,
    frac_max_steps,
):
    """Jit-compatible implementation of Fermi-Dirac fractional occupation."""

    # Initial integer occupations - vectorized implementation
    n_orbs = mo_energy.shape[0]
    orbital_indices = jnp.arange(n_orbs)

    # Calculate how many doubly occupied orbitals we have
    n_double = jnp.floor(n_electrons / 2.0).astype(jnp.int32)
    remainder = n_electrons - 2.0 * n_double

    # Create mask for doubly occupied, singly occupied, and empty orbitals
    double_mask = orbital_indices < n_double
    single_mask = (orbital_indices == n_double) & (remainder > 0)

    # Create initial occupations
    initial_occ = 2.0 * double_mask.astype(jnp.float32) + remainder * single_mask.astype(
        jnp.float32,
    )

    # Set initial mu between HOMO and LUMO if possible
    homo_idx = jnp.maximum(n_double - 1, 0)
    remainder_factor = jnp.where(remainder > 0, 1, 0)
    lumo_idx = jnp.minimum(n_double + remainder_factor, n_orbs - 1)

    homo_energy = mo_energy[homo_idx]
    lumo_energy = mo_energy[lumo_idx]

    # Default mu between HOMO-LUMO or use provided value
    default_mu = 0.5 * (homo_energy + lumo_energy)

    # Handle None case - no need for isnan check
    mu_init = default_mu if frac_mu is None else frac_mu

    # Define the optimization step function
    def fermi_step(mu, _):
        grad_fermi = grad_loss_fermi_function(mu, mo_energy, n_electrons, theta=theta)
        gradgrad_fermi = gradgrad_loss_fermi_function(mu, mo_energy, n_electrons, theta)
        safe_gradgrad = jnp.where(jnp.abs(gradgrad_fermi) < 1e-10, 1e-10, gradgrad_fermi)
        mu_new = mu - frac_step_grad * grad_fermi / safe_gradgrad
        return mu_new, None

    # Use fori_loop instead of scan with a fixed maximum number of iterations
    def body_fun(i, mu):
        return fermi_step(mu, None)[0]

    # Run a fixed max number of iterations, but use conditional to only perform up to frac_max_steps
    mu_final = jax.lax.fori_loop(
        0,
        100,  # Fixed upper bound
        lambda i, mu: jax.lax.cond(
            i < frac_max_steps,
            lambda _: body_fun(i, mu),
            lambda _: mu,
            None,
        ),
        mu_init,
    )

    # Calculate occupation with the optimized mu
    fermi_occ = make_occ_fermi(mu_final, mo_energy, theta=theta)

    # Compute total electrons to check validity
    total_electrons = jnp.sum(fermi_occ)
    electron_error = jnp.abs(total_electrons - n_electrons)

    # For fallback, create slightly smoothed occupations with small theta
    small_theta = jnp.array(0.01)

    # Simple fixed-point iteration to find mu that preserves electron count with small theta
    def mu_finder_step(mu, _):
        occ = make_occ_fermi(mu, mo_energy, small_theta)
        curr_n_elec = jnp.sum(occ)
        new_error = curr_n_elec - n_electrons
        # Update direction based on error
        mu_new = mu - 0.1 * new_error  # Simple step size
        return mu_new, None

    # Run a few iterations to find a better mu for fallback using fori_loop
    fallback_mu = jax.lax.fori_loop(
        0,
        10,  # Just 10 iterations
        lambda i, mu: mu_finder_step(mu, None)[0],
        default_mu,
    )

    # Calculate fallback occupations
    fallback_occ = make_occ_fermi(fallback_mu, mo_energy, small_theta)

    # Use optimized occupations if error is small, fallback otherwise
    is_valid = electron_error < 1e-2
    final_occ = jnp.where(is_valid, fermi_occ, fallback_occ)
    final_mu = jnp.where(is_valid, mu_final, fallback_mu)

    # Calculate entropy contribution
    entropy_theta = jnp.where(is_valid, theta, small_theta)

    # Use vmap instead of list comprehension for vectorization
    fd_occ_for_entropy = jax.vmap(lambda e: fermi_function4entropy(e, final_mu, entropy_theta))(
        mo_energy,
    )
    energy_entr = 2 * entropy_theta * jnp.sum(entropy(fd_occ_for_entropy))

    # Use a dummy loss value since we don't compute it in this version
    loss_final = jnp.array(0.0)

    return final_occ, energy_entr, final_mu, loss_final


if __name__ == "__main__":
    # Test values
    test_energy = jnp.array([-1.1, -1.2, -1.3])  # Some MO energies
    test_mu = 1.0  # Chemical potential
    test_theta = 0.5  # Temperature parameter
    test_N = 4  # Number of electrons
    # Test get_fractional_occupations
    test_mu_shift = 0.5
    test_step_grad = 0.6
    test_max_steps = 100

    # Test each function
    print("Testing fermi_function4entropy:")
    print(fermi_function4entropy(test_energy, test_mu, test_theta))

    print("\nTesting fermi_function:")
    print(fermi_function(test_energy, test_mu, test_theta))

    print("\nTesting loss_fermi_function:")
    print(loss_fermi_function(test_mu, test_energy, test_N, test_theta))

    print("\nTesting gradient of loss function:")
    print(grad_loss_fermi_function(test_mu, test_energy, test_N, test_theta))

    print("\nTesting second gradient of loss function:")
    print(gradgrad_loss_fermi_function(test_mu, test_energy, test_N, test_theta))

    print("\nTesting make_occ_fermi:")
    print(make_occ_fermi(test_mu, test_energy, test_theta))

    # Test entropy with some occupation numbers
    test_fermi = fermi_function4entropy(test_energy, test_mu, test_theta)
    print("\nTesting entropy:")
    print(entropy(test_fermi))

    print("\nTesting entropic_energy_contribution:")
    print(etropic_energy_contribution(test_mu, test_energy, test_theta))

    print("\nTesting get_fractional_occupations:", "---" * 8)

    mo_occ, energy_entr, mu_opt, loss_final = get_fractional_occupations(
        mo_energy=test_energy,
        n_electrons=test_N,
        frac_mo_occ=1,
        frac_theta=test_theta,
        frac_mu=None,
        frac_mu_shift=test_mu_shift,
        frac_step_grad=test_step_grad,
        frac_max_steps=test_max_steps,
    )

    mo_occ_jax, energy_entr_jax, mu_final_jax, loss_final_jax = get_fractional_occupations_jax(
        mo_energy=test_energy,
        n_electrons=test_N,
        theta=test_theta,
        frac_mu=None,
        frac_mu_shift=test_mu_shift,
        frac_step_grad=test_step_grad,
        frac_max_steps=test_max_steps,
    )

    print("\nTesting get_fractional_occupations (original | jax):")
    print("mo_occ: ", mo_occ, mo_occ_jax)
    print("energy_entr: ", energy_entr, energy_entr_jax)
    print("mu_opt: ", mu_opt, mu_final_jax)
    print("loss_final: ", loss_final, loss_final_jax)
    print("sum mo_occ (nelectrons): ", jnp.sum(mo_occ), jnp.sum(mo_occ_jax))
    print("---" * 20)

    print("\nTesting jax theta:")

    # Get mo_occ with PySCF for H2
    from pyscf import gto
    from pyscf.scf import RHF

    mol = gto.M(atom="H 0 0 0; H 0 0 3.74", basis="6-31G", unit="Ang")
    mf = RHF(mol)
    mf.kernel()
    mo_occ = mf.mo_occ
    mo_energy = mf.mo_energy
    print(f"mo_energy: {mo_energy}")

    # mo_energy =jnp.array([-1.3, -1.2, -1.1, -1.0])
    mu_opt = None  # -0.17

    mo_occ, energy_entr, mu_opt, loss_final = get_fractional_occupations_jax(
        mo_energy=jnp.array(mo_energy),
        n_electrons=2,
        theta=0.02,
        frac_mu=mu_opt,
        frac_mu_shift=0.0,
        frac_step_grad=0.6,
        frac_max_steps=100,
    )

    print(
        f"Large theta: mo_occ: {mo_occ} | energy_entr: {energy_entr} | mu_opt: {mu_opt} | loss_final: {loss_final}",
    )
    print(f"sum mo_occ (nelectrons): {jnp.sum(mo_occ)}")
