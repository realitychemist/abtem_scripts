import rle
from sys import maxsize
from ase import Atom, Atoms
from random import randint, seed, shuffle
from copy import deepcopy
from itertools import combinations_with_replacement, permutations
from collections.abc import Generator


def randomize_chem(atoms: Atoms,
                   replacements: dict[str, dict[str, float]],
                   prseed: int = randint(0, maxsize))\
        -> Atoms:
    """Randomize the chemistry of an ASE ``Atoms`` object via to user-defined replacement rules.

    Parameters
    ----------
    atoms : Atoms
        Initial ASE ``Atoms`` object.  A changed copy of this object will be returned.
    replacements : Dict[str, Dict[str, float]]
        Replacement dictionary.  The keys should be the symbols of the initial elements to replace,
        and the values should themselves be dictionaries.  The value dicts should have keys which
        are the elements that will replace the corresponding initial element, and the values
        should be floats representing the fraction of the initial element to replace with the given
        element.  The sum of the floats must be <= 1 for each initial element.  For example:
            >>> {"Ba": {"Sr": 1},
            >>>  "Ti": {"Zr": 0.4,
            >>>         "Nb": 0.05}}
        would replace all Ba atoms in ``atoms`` with Sr, would randomly replace 40% of Ti atoms in
        ``atoms`` with Zr, and randomly replace 5% (of the initial amount of Ti) with Nb.
    prseed : int, optional
        Pseudo-random seed.  The default is to randomly choose a seed between 0 and sys.maxsize.

    Returns
    -------
    Atoms
        ASE ``Atoms`` object based on ``atoms``, but with the specified elemental replacements.
    """
    seed(prseed)
    new_atoms = deepcopy(atoms)

    # Add in NOP weights and then sanity check
    for elem, reps in replacements.items():
        if sum(reps.values()) < 1:  # Add in the "NOP weights" (chance to not replace) if needed
            reps[elem] = 1 - sum(reps.values())
        if sum(reps.values()) != 1:  # If this is ever False, we're likely to get garbage results
            raise RuntimeError(f"Sum of rep values ({sum(reps.values())}) not equal to 1!")

    symbols = new_atoms.get_chemical_symbols()
    uniques = list(set(symbols))  # Since set iteration order is not guaranteed, cast back to a list
    counts = dict(zip(uniques, [symbols.count(e) for e in uniques]))

    for elem, reps in replacements.items():
        elem_idxs = [idx for idx, sym in enumerate(symbols) if sym == elem]
        k = counts[elem]
        rep_counts = [round(k*w) for w in reps.values()]
        n = sum(rep_counts)

        while n != k:  # Counts may be off due to cumulative rounding error
            diffs = [c/k - w for w, c in zip(reps.values(), rep_counts)]
            if n < k:  # Since we have too few reps, we want to add one to the most negative diff
                mindex = diffs.index(min(diffs))
                rep_counts[mindex] = rep_counts[mindex] + 1
            elif n > k:  # Since we have too many reps, we want to remove one from the most positive diff
                maxdex = diffs.index(max(diffs))
                rep_counts[maxdex] = rep_counts[maxdex] - 1
            n = sum(rep_counts)

        rep_with = [x for sublist in  # Outer list comprehension to make sure we get a flat list
                    [[e]*c for e, c in zip(reps.keys(), rep_counts)]
                    for x in sublist]
        shuffle(rep_with)  # Now the only random step in this whole process is the shuffle

        for i in elem_idxs:
            symbols[i] = rep_with.pop()

    new_atoms.set_chemical_symbols(symbols)
    return new_atoms


def gen_variations(model: Atoms,
                   subs: tuple[str, str])\
        -> Generator[tuple[str, Atoms]]:
    """Generate ASE ``Atoms`` models by all possible combinations and permutations of the given site.

    Parameters
    ----------
    model : Atoms
        The model to change the chemistry of.  Should be an ASE ``Atoms`` object, with all sites
        which are to be changed having the same element.  Any sites with a different element
        on them will not participate in the variation generation.
    subs : tuple[str, str]
        A tuple of elements allowed on the given sites, represented using their atomic symbols.
        For example:
            >>> ["Ti", "Nb"]
        The first element of the list *must* be the element which sits on the model sites where variations
        are going to be generated, this is how this function determines what counts as the same kind of site!

    Returns
    ------
    Generator[tuple[str, Atoms]]
        Yields tuples.  Each tuple consists of an RLE-encoded label string representing the changed sites in
        the model (in the order in which they occur) followed by an ASE ``Atoms`` object with the sites
        substituted as the label would suggest.  The order in which label-model pairs are yielded is arbitrary.
    """
    # Get all atoms in model which have the element in subs[0]
    symbols = model.get_chemical_symbols()
    num_sites = symbols.count(subs[0])
    idxs = [idx for idx, sym in enumerate(symbols) if sym == subs[0]]

    # Generate all possible arrangements of the elements for this model
    # For larger models these structures will become *very* large, hence: generators
    b_combos = combinations_with_replacement(subs, num_sites)
    b_perms = (set(permutations(combo)) for combo in b_combos)
    b_arrs = (arrangement for subset in b_perms for arrangement in subset)

    # Yield a model for each arrangement
    for arrangement in b_arrs:
        for counter, b_idx in enumerate(idxs):
            symbols[b_idx] = arrangement[counter]
        new_model = deepcopy(model)
        new_model.set_chemical_symbols(symbols)
        encoding = list(zip(*reversed(rle.encode([elem for elem in arrangement]))))
        label = "".join(str(cnt)+str(sym) for cnt, sym in encoding)
        yield label, new_model


def split_by_projected_columns(model: Atoms,
                               kinds: str | list[str],
                               tol: int = 3)\
        -> list[list[Atom]]:
    symbols = model.get_chemical_symbols()
    positions = model.get_positions()

    selected_positions = [pos for i, pos in enumerate(positions) if symbols[i] in kinds]
    unique_xys = {str([round(pos[0], tol), round(pos[1], tol)]) for pos in selected_positions}

    submodels = {}
    for uxy in unique_xys:
        submodels[uxy] = []
    for atom in model:
        if atom.symbol not in kinds:
            continue  # Skip atoms if they're the wrong kind of element
        xy = str([round(atom.position[0], tol), round(atom.position[1], tol)])
        submodels[xy].append(atom)
    return list(submodels.values())
