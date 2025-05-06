"""Generate and process molecular electronic structure data.

This module provides functionality to:
1. Generate single data points for training, including electronic energies and densities
2. Parse and convert density data between different unit systems
3. Support both CCSD and FCI calculations

Key features:
- Configurable grid density and derivative order
- Unit handling between Angstroms and Bohr
- Support for custom grids
- Consistent file organization

Example usage:
    >>> config = MoleculeConfig(
    ...     name="H2",
    ...     atom_coords="H 0 0 0; H 0 0 0.74",
    ...     units="Ang",
    ...     basis="631g",
    ...     method="CCSD",
    ...     grid_density=3
    ... )
    >>> generator = DataGenerator("./data")
    >>> energy, density, coords = generator.generate_data(config)

Notes:
    - Units are standardized to Angstroms throughout
    - Density derivatives not yet implemented but framework exists
    - PySCF unit handling requires careful attention
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pyscf
from loguru import logger
from pyscf import cc, fci, scf
from pyscf.dft import numint
from pyscf.lib import param
from pyscfad import gto
from pyscfad.dft import numint as numintad

from qedft.train.td import rks_legacy as dft

FloatArray = npt.NDArray[np.float64]


@dataclass
class MoleculeConfig:
    """Configuration for molecular electronic structure calculations.

    Args:
        name: Identifier for the molecule
        atom_coords: Atomic coordinates in PySCF format (e.g. "H 0 0 0; H 0 0 0.74")
        units: Length units, either "Ang" (Angstroms) or "B" (Bohr)
        basis: Basis set name (e.g. "631g", "cc-pvdz")
        method: Electronic structure method ("CCSD", "FCI", or "rks")
        grid_density: DFT grid level from 0 (sparse) to 9 (dense)
        deriv: Order of density derivatives (0=none, 1=gradients, 2=hessians)
        label: Optional identifier for the calculation
        custom_grid: Optional custom grid points
        xc: Exchange-correlation functional (for RKS method)
        max_cycle: Maximum number of SCF cycles
        max_memory: Maximum memory usage in bytes
        use_ccsd_triples_correction: Whether to include (T) correction for CCSD
    """

    name: str
    atom_coords: str
    units: Literal["Ang", "B"] = "Ang"
    basis: str = "631g"
    method: Literal["CCSD", "FCI", "rks"] = "CCSD"
    grid_density: int = 3
    deriv: int = 0
    label: int | str = 0
    custom_grid: FloatArray | None = None
    xc: str = "lda"
    max_cycle: int = 20
    max_memory: float = 1e6
    use_ccsd_triples_correction: bool = False
    symmetry: bool = True


class DataGenerator:
    """Handles generation and storage of molecular electronic structure data.

    Args:
        output_dir: Directory to store generated data files
    """

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)

    def _get_filepath_prefix(self, config: MoleculeConfig) -> Path:
        """Generates consistent file paths for all outputs.

        Args:
            config: Molecule configuration

        Returns:
            Path prefix for output files
        """
        suffix = (
            f"{config.label}-{config.units}-{config.method}-"
            f"{config.basis}-d{config.deriv}-g{config.grid_density}"
        )
        mol_dir = self.output_dir / config.name
        mol_dir.mkdir(parents=True, exist_ok=True)
        return mol_dir / suffix

    def generate_data(
        self,
        config: MoleculeConfig,
        save_data: bool = True,
    ) -> tuple[gto.Mole, Any, FloatArray, float, FloatArray, FloatArray]:
        """Generates and saves molecular data based on configuration.

        Args:
            config: Complete molecule specification
            save_data: Whether to save the data to files
        Returns:
            Tuple containing:
                - Molecule object
                - SCF object
                - Density matrix in AO basis
                - Total electronic energy
                - Electron density on grid
                - Grid coordinates
        """
        # Build molecule
        mol = gto.Mole(symmetry=config.symmetry)
        mol.atom = config.atom_coords
        mol.unit = config.units
        mol.basis = config.basis
        mol.build()

        # Generate grid
        grids = pyscf.dft.gen_grid.Grids(mol)
        grids.level = config.grid_density
        grids.becke_scheme = pyscf.dft.gen_grid.stratmann
        grids.build()

        logger.info(f"Grid points: {grids.coords.shape}")

        # Calculate energy and density
        mol, mf, dm_ao, energy, density, coords = calculate_energy_and_density(
            mol=mol,
            grid=grids,
            method=config.method.lower(),
            xc=config.xc,
            max_cycle=config.max_cycle,
            use_ccsd_triples_correction=config.use_ccsd_triples_correction,
            max_memory=config.max_memory,
        )

        # Setup file paths
        prefix = self._get_filepath_prefix(config)
        filepath_energy = f"{prefix}-energy.npy"
        filepath_coords = f"{prefix}-coords.npy"
        filepath_density = f"{prefix}-density.chdens"
        filepath_density_npy = f"{prefix}-density.npy"
        filepath_dm_ao = f"{prefix}-dm_ao.npy"

        # Save data
        if save_data:
            np.save(filepath_energy, energy)
            np.save(filepath_coords, coords)
            np.save(filepath_density_npy, density)
            np.save(filepath_dm_ao, dm_ao)

            # Save density in chdens format
            with open(filepath_density, "w") as f:
                for i in range(coords.shape[0]):
                    for j in range(coords.shape[1]):
                        f.write(str(f"{round(coords[i][j], 6):5f} "))
                    f.write(" " + str(f"{density[i]:e}"))
                    f.write("\n")

            _ = self.parse_density_data(filepath_density, config.units)

        return mol, mf, dm_ao, energy, density, coords

    @staticmethod
    def parse_density_data(
        filepath: str | Path,
        units: str = "A",
    ) -> tuple[FloatArray, FloatArray]:
        """Parses chdens file to a numpy array.

        Args:
            filepath: Path to density file
            units: Length units, either "A" (Angstroms) or "B" (Bohr)

        Returns:
            Tuple containing:
                - Density array
                - Coordinates array

        Note:
            PySCF unit handling requires careful attention:
            - Default output is in Angstroms despite internal Bohr units
            - Unit conversion is handled explicitly for consistency
        """

        logger.info(f"\nPARSER: Starting and converting if needed to {units}...")
        logger.info(
            "PARSER: PySCF typically does everything in NOT in Angstroms (A) but Bohr (B) - atomic units.",
        )

        data: list[list[float]] = []
        if "B" in units.upper():
            logger.info("PARSER: Converting to Bohr the coordinates.")
        elif "A" in units.upper():
            logger.info("PARSER: Already in Angstroms. No conversion needed.")
        else:
            raise ValueError("PARSER: Choose correct units! ANG or B.")

        # I am still not convinced here. I think ti is all in atomic units.
        # So no conversion needed.
        with open(filepath) as file:
            for line in file:

                # Old version here. But it is correct.
                # by default Angstroms are assumed (not atomic units)!
                # It is a bit weird, but it is what it is.

                # Removing empty spaces.
                line = " ".join(line.split())
                # Make sure it is Bohr.
                if "B" in units.upper():
                    converted_line = map(np.float64, line.split(" "))
                    parsed = list(converted_line)
                    # Conversion to Bohr.
                    # x, y, z, rho for each line.
                    # Only modify coordinates.
                    for i in range(3):
                        parsed[i] *= 1 / param.BOHR  # param.BOHR = 0.529 [A / Bohr]
                elif "A" in units.upper():
                    # This was float32 in the original code.
                    parsed = list(map(np.float64, line.split(" ")))

                # FIXME: Bad version here. PySCFAD got a but inside.
                # This below is with the reverse conversion.
                # Just to keep in case.

                # # Removing empty spaces.
                # line = " ".join(line.split())
                # # Make sure it is Bohr.
                # if "A" in units.upper():
                #     converted_line = map(np.float64, line.split(" "))
                #     parsed = list(converted_line)
                #     # Conversion to Bohr.
                #     # x, y, z, rho for each line.
                #     # Only modify coordinates.
                #     for i in range(3):
                #         parsed[i] *= param.BOHR  # param.BOHR = 0.529 [A / Bohr]
                #     parsed[3] *= param.BOHR**3  # rho in e / A^3
                # elif "B" in units.upper():
                #     # This was float32 in the original code.
                #     # No modification is needed if in Bohr since it is for sure
                #     # in Bohr for the distances and e / Bohr**3 for the density.
                #     parsed = list(map(np.float64, line.split(" ")))
                #     # No conversion is needed.
                # else:
                #     raise ValueError("PARSER: Choose correct units! B or AU.")

                data.append(parsed)

        # Converting to numpy.
        data_array = np.array(data)

        # Final tensor shape.
        logger.info(f"PARSER: Final data shape {data_array.shape}")
        filepath = os.path.splitext(filepath)[0] + ".npy"

        # Save the density.
        np.save(
            filepath,
            data_array,
        )
        return data_array


def calculate_energy_and_density(
    mol: gto.Mole,
    grid: pyscf.dft.gen_grid.Grids,
    method: Literal["rks", "ccsd", "fci"],
    xc: str,
    max_cycle: int = 20,
    use_ccsd_triples_correction: bool = False,
    max_memory: float = 1e6,
    **kwargs: Any,
) -> tuple[gto.Mole, Any, FloatArray, float, FloatArray, FloatArray]:
    """Calculate molecular energy and electron density using various electronic structure methods.

    Performs electronic structure calculations using the specified method (RKS, CCSD, or FCI)
    to obtain the total energy and electron density on a grid.

    Args:
        mol: PySCF Mole object containing molecular geometry and basis set
        grid: DFT integration grid specification
        method: Electronic structure method to use ("rks", "ccsd", or "fci")
        xc: Exchange-correlation functional for DFT calculations
        max_cycle: Maximum number of SCF cycles
        use_ccsd_triples_correction: Whether to include perturbative triples correction for CCSD
        max_memory: Maximum memory usage in bytes
        **kwargs: Additional keyword arguments passed to the electronic structure methods

    Returns:
        Tuple containing:
            - mol: Input molecule object
            - mf: Mean field object (RKS or RHF)
            - dm_ao: One-particle density matrix in AO basis
            - exact_energy: Total electronic energy
            - exact_density: Electron density evaluated on the grid points
            - coords: Grid point coordinates

    Raises:
        ValueError: If an unsupported method is specified
    """

    logger.info(f"    Generating data using {method} method")

    if method.lower() == "rks":
        mf = dft.RKS(
            mol,
            max_cycle=max_cycle,
            max_memory=max_memory,
            **kwargs,
        )
        mf.grids = grid
        mf.xc = xc
        # DFT calculation
        mf.kernel()
        # Grid units are Angstroms because that is default.
        coords = mf.grids.coords
        dm_ao = mf.make_rdm1()

        # Energy
        energy = mf.energy_tot(dm=dm_ao)
        # energy_elec = mf.energy_elec(dm=dm)

        # Density
        ao_value = numint.eval_ao(mol, coords, deriv=0)
        # rho = numint.eval_rho(mol, ao_value, dm_ao, xctype="LDA")
        # Below I tested has the same result
        rho = numintad.eval_rho(mol, ao_value, dm_ao, xctype="LDA")

        # TODO: GGA to evaluate the gradients for the additional loss terms
        # ao_value = numint.eval_ao(
        #     mol, coords, deriv=1
        # )  # AO value and its gradients
        # rho_all = numintad.eval_rho(
        #     mol, ao_value, dm, xctype="GGA"
        # )  # density & density gradients
        # rho, _, _, _ = rho_all

        exact_density = rho.flatten()
        exact_energy = energy

    elif method.lower() == "ccsd":
        # Run SCF and use CCSD to get RDMs.
        mf = scf.RHF(mol)
        mf.kernel()

        mycc = cc.CCSD(mf).run()
        mycc.kernel()
        energy = mycc.e_tot
        if use_ccsd_triples_correction:
            et = mycc.ccsd_t()  # works only when not reading FCIDUMP file.
            logger.info(f"    CCSD(T) correction: {et}")
            energy += et

        dm_ao = mycc.make_rdm1(ao_repr=True)
        # Or you can use RHF RDM1 (it is not CCSD one)
        # But the results will be of RHF
        # dm_ao = mycc._scf.make_rdm1(mycc.mo_coeff, mycc.mo_occ)
        dm = mycc.make_rdm1()

        # and then use numint.eval_rho, use Lebdev grid from RKS object, or use any
        # coords you want to evaluate:
        mf = dft.RKS(
            mol,
            max_cycle=max_cycle,
            max_memory=max_memory,
            **kwargs,
        )
        # mf = scf.addons.frac_occ(mf)
        mf.grids = grid
        # LDA should be sufficient as we just want the coordinates.
        # The orbitals are not from DFT but from CCSD/FCI.
        # This is just to call the correct method that is used by LDA.
        mf.xc = "lda"  # LDA just to eval the rho
        # Run the DFT calculation to get the density.
        mf.kernel()
        # Grid units are Bohr because that is default.
        coords = mf.grids.coords
        # # Here we also have an option to train on the gradients of the density.
        ao_value = numint.eval_ao(mol, coords, deriv=0)  # no derivatives
        # rho = numintad.eval_rho(mol, ao_value, dm_ao, xctype="LDA")
        rho = numint.eval_rho(mol, ao_value, dm_ao, xctype="LDA")
        # rho = numint.eval_rho2(mol, ao_value, mo_coeff, mo_occ, xctype="LDA")

        # TODO: GGA to evaluate the gradients for the additional loss terms
        # ao_value = numint.eval_ao(
        #     mol, coords, deriv=1
        # )  # AO value and its gradients
        # rho_all = numintad.eval_rho(
        #     mol, ao_value, dm, xctype="GGA"
        # )  # density & density gradients
        # rho, _, _, _ = rho_all

        exact_density = rho.flatten()
        exact_energy = energy

    elif method.lower() == "fci":
        # Run SCF and use CCSD to get RDMs.

        # RHF is also possible to run here
        # mf = scf.RHF(mol)
        # mf.grids = grid
        # mf.kernel()

        mf = dft.RKS(
            mol,
            max_cycle=max_cycle,
            max_memory=max_memory,
            **kwargs,
        )
        mf.grids = grid
        mf.xc = xc
        # DFT calculation
        mf.kernel()

        logger.info(" FCI calculation and generating density.")
        norb = mf.mo_energy.size
        # Make for alphas and betas orbitals the same ones.
        # mo_coeffs = mf.mo_coeff  # misses beta orbitals.
        mo_coeffs = np.stack([mf.mo_coeff, mf.mo_coeff], axis=0)
        fs = fci.FCI(mol, mo_coeffs)
        energy, c = fs.kernel()
        # FCI with PySCF: Computes the RDM of the molecule.
        dm = fs.make_rdm1s(c, norb, mol.nelectron)

        # Factor 2 is needed to get the correct density by comparing
        # with CCSD density.
        # Below we perform the basis transformation to get the density
        # in the AO basis.
        dm_ao = 2.0 * mo_coeffs[0] @ dm[0] @ mo_coeffs[0].T
        # Correction for the true dm same as in CCSD value.
        dm = 2.0 * dm[0]

        fs = fci.FCI(mol, mf.mo_coeff)
        dm_fs = fs.make_rdm1(fcivec=c, norb=norb, nelec=mol.nelectron)
        # assert
        if np.allclose(dm, dm_fs) is False:
            # better prefer FCI dm_fs
            logger.info("    FCI and DFT density are not the same.")
            logger.info("    Using FCI density.")
            dm = dm_fs / 2.0
            dm_ao = 2.0 * mo_coeffs[0] @ dm @ mo_coeffs[0].T

        # and then use numint.eval_rho, use Lebdev grid from RKS object, or use any
        # coords you want to evaluate:
        mf = dft.RKS(
            mol,
            max_cycle=max_cycle,
            **kwargs,
        )
        mf.grids = grid
        # LDA should be sufficient as we just want the coordinates.
        # The orbitals are not from DFT but from CCSD/FCI.
        # This is just to call the correct method that is used by LDA.
        mf.xc = "lda"  # LDA just to eval the rho
        # Run the DFT calculation to get the density.
        mf.kernel()
        # Grid units are Bohr because that is default.
        coords = mf.grids.coords
        # Here we also have an option to train on the gradients of the density.
        ao_value = numint.eval_ao(mol, coords, deriv=0)  # no derivatives
        rho = numint.eval_rho(mol, ao_value, dm_ao, xctype="LDA")

        # TODO: GGA fashion for additional loss terms
        # ao_value = numint.eval_ao(
        #     mol, coords, deriv=1
        # )  # AO value and its gradients
        # rho_all = numintad.eval_rho(
        #     mol, ao_value, dm, xctype="GGA"
        # )  # density & density gradients
        # rho, _, _, _ = rho_all

        exact_density = rho.flatten()
        exact_energy = energy

    else:
        raise ValueError(
            f"  Method {method} for data generation is not implemented (e.g.," " ccsd, rks, fci).",
        )

    logger.info(f"    Number of grid points: {coords.shape[0]}")
    logger.info(f"    Density matrix AO shape: {dm_ao.shape}")
    logger.info(f"    Shape of rho: {rho.shape}")
    logger.info(f"    Energy total: {energy}")

    return (
        mol,
        mf,
        dm_ao,
        exact_energy,
        exact_density,
        mf.grids.coords,
    )


if __name__ == "__main__":
    import qedft

    # Example usage
    config = MoleculeConfig(
        name="H2_test",
        atom_coords="H 0 0 0; H 0 0 0.74",
        units="Ang",
        basis="631g",
        method="CCSD",
        grid_density=0,
    )

    project_path = Path(os.path.dirname(os.path.dirname(qedft.__file__)))
    output_dir = project_path / "data" / "td"
    generator = DataGenerator(output_dir=output_dir)
    mol, mf, dm_ao, energy, density, coords = generator.generate_data(config, save_data=True)
    logger.info(f"Generated data for {config.name}")
    logger.info(f"Energy: {energy}")
    logger.info(f"Grid points: {coords.shape[0]}")
    logger.info(f"Density: {density}")
    logger.info(f"Density shape: {density.shape}")
    logger.info(f"DM AO shape: {dm_ao.shape}")
    logger.info(f"DM AO: {dm_ao}")
    logger.info(f"Molecule: {mol}")
    logger.info(f"Mean field: {mf}")
