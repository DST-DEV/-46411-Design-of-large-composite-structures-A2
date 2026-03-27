import numpy as np

from classical_laminate_theory.materials.base import Material2D
from classical_laminate_theory.ply.ply import Ply
from classical_laminate_theory.laminate.laminate import Laminate

class LaminateBuilder:
    """Laminate stackup builder"""
    @staticmethod
    def from_stack(stack: str, materials: dict | Material2D,
                   ply_thicknesses: float | list[float]):
        """
        Construct a Laminate instance from a compact stack definition string.

        This factory method parses a laminate stackup definition and creates
        the corresponding sequence of Ply objects with uniform thickness.

        Parameters
        ----------
        stack : str
            Laminate stack definition using bracket notation.

            Format
            ------
            The stack must be defined inside square brackets:

                "[<ply_1>/<ply_2>/.../<ply_n>]"            (unsymmetric)
                "[<ply_1>/<ply_2>/.../<ply_n>]_S"          (symmetric)

            Each ply entry has the form:

                <angle>[_<repeat>][@<material_key>]

            where:

            - <angle> : float
                Fiber orientation angle in degrees (local 1-axis relative to
                laminate x-axis). Positive angles follow the right-hand rule.
            - _<repeat> : optional integer
                Repeats the ply orientation consecutively.
                Example: "-45_2" expands to [-45, -45].
            - @<material_key> : optional string
                Key referencing a material in the `materials` mapping. This key
                can be omitted if only a single material is specified.

            Examples
            --------
            "[0/45/-45/90]"
                Four-ply unsymmetric laminate.

            "[0/-45_2/45_2/90]_S"
                Symmetric laminate. The sequence inside brackets is mirrored
                about the mid-plane.

            "[0@C/-45_2@G/45_2@G/90@C]_S"
                Symmetric laminate with material assignment per ply.

            Symmetry
            --------
            If the suffix "_S" is present (case-insensitive), the laminate is
            assumed symmetric and the parsed core sequence is mirrored.
        materials : dict[str, Material2D] or Material2D
            Either:
            - A mapping from material keys (used after '@' in the stack
              string) to Material2D instances, or
            - A single material applied to all plies. In this case the
              material_key can be omitted in the stackup string.
        ply_thickness : float or list[float]
            Thickness assigned to every ply.

        Returns
        -------
        Laminate
            Laminate instance containing the expanded ply sequence.

        Notes
        -----
        - All angles are interpreted in degrees.
        - Ply ordering is from bottom to top.
        - The mid-plane is located automatically by the Laminate class.
        """
        angles, material_tags = StackParser.parse(stack)

        if isinstance(ply_thicknesses, (int, float, np.number)):
            ply_thicknesses = [ply_thicknesses]*len(angles)
        elif isinstance(ply_thicknesses, (tuple, list, np.ndarray)):
            if not len(angles) == len(ply_thicknesses):
                raise ValueError("Length mismatch of stackup sequence and ply "
                                 "thicknesses.")
        else:
            raise TypeError("Ply thicknesses must be a scalar numeric value or"
                            " a list-like object.")

        if isinstance(materials, dict):
            if not len(material_tags) == len(angles):
                raise ValueError("Insufficient number of materials specified"
                                 " for stackup sequence.")
            plies = [Ply(material=materials[mtag], theta=angle,
                         thickness=ply_thicknesses[i])
                     for i, (angle, mtag) in enumerate(zip(angles,
                                                           material_tags))]
        elif isinstance(materials, Material2D):
            plies = [Ply(material=materials, theta=angle,
                          thickness=ply_thicknesses[i])
                     for i, angle in enumerate(angles)]
        else:
            raise TypeError("Material must be a Material2D instance.")

        return Laminate(plies)

class StackParser:
    @staticmethod
    def parse(stack: str) -> tuple[list[float], list[str]]:
        """
        Parse a stackup sequence to a list of angles and material tags.

        Examples
        --------
        "[0@C/-45@G/45@G/90@C]"  -> ([0,45,-45,90], ["C","G","G","C"])
        "[0@C/90@C]_S"           -> ([0,90,90,0], ["C","C","C","C"])
        "[0@C/90_2@G]"           -> ([0,90,90], ["C","G","G"])

        Parameters
        ----------
        stack : str
            Laminate stack definition string.

        Returns
        -------
        angles : list[float]
            List of angles of the stackup in degrees.
        material_tags : list[str]
            List of the material tag for each ply.
        """
        core, symmetric = StackParser._extract_symmetry(stack)
        tokens = core.split("/")
        angles = []
        material_tags = []

        for token in tokens:
            tag = None
            # Split up optional material tag (e.g. 0@C)
            if "@" in token:
                token, tag = token.split("@")

            token = StackParser._expand_token(token)
            angles.extend(token)

            if tag is not None:
                material_tags.extend([tag]*len(token))

        if symmetric:
            angles = angles + angles[::-1]
            material_tags = material_tags + material_tags[::-1]

        return angles, material_tags

    @staticmethod
    def _extract_symmetry(stack: str) -> tuple[str, bool]:
        """
        Extract laminate core definition and symmetry flag.

        Examples
        --------
        "[0/45/-45/90]"     -> ("0/45/-45/90", False)
        "[0/45/-45/90]_S"   -> ("0/45/-45/90", True)
        "[0]_s"             -> ("0", True)

        Parameters
        ----------
        stack : str
            Laminate stack definition string.

        Returns
        -------
        core : str
            Stack definition inside brackets without symmetry suffix.
        symmetric : bool
            True if laminate is symmetric (_S), otherwise False.
        """

        if not stack.startswith("["):
            raise ValueError("Stack definition must start with '['")

        if "]" not in stack:
            raise ValueError("Stack definition must contain ']'")

        close_idx = stack.index("]")
        core = stack[1:close_idx]

        suffix = stack[close_idx + 1 :].strip()

        symmetric = False
        if suffix:
            if suffix.lower() == "_s":
                symmetric = True
            else:
                raise ValueError(
                    f"Invalid stack suffix '{suffix}'. "
                    "Only '_S' is supported."
                )

        return core, symmetric

    @staticmethod
    def _expand_token(token: str) -> list[float]:
        """Expand repeated token."""
        if "_" in token:
            angle, repeat = token.split("_")
            return [float(angle)] * int(repeat)
        return [float(token)]

if __name__ == "__main__":
    from classical_laminate_theory.materials.orthotropic import OrthotropicLamina
    glass = OrthotropicLamina(E1=40e9, E2=9.8e9, G12=2.8e9, nu12=.3)
    carbon = OrthotropicLamina(E1=136e9, E2=10e9, G12=5.2e9, nu12=.3)

    laminate = LaminateBuilder.from_stack(
        stack="[0@C/-45_2@G/45_2@G/90@C]_S",
        materials={"C": carbon, "G": glass},
        ply_thicknesses=0.125e-3
)
