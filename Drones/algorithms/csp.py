from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from algorithms.problems_csp import DroneAssignmentCSP


def backtracking_search(csp: DroneAssignmentCSP, assignment: dict[str, str] | None = None) -> dict[str, str] | None:
    """
    Basic backtracking search without optimizations.

    Tips:
    - An assignment is a dictionary mapping variables to values (e.g. {X1: Cell(1,2), X2: Cell(3,4)}).
    - Use csp.assign(var, value, assignment) to assign a value to a variable.
    - Use csp.unassign(var, assignment) to unassign a variable.
    - Use csp.is_consistent(var, value, assignment) to check if an assignment is consistent with the constraints.
    - Use csp.is_complete(assignment) to check if the assignment is complete (all variables assigned).
    - Use csp.get_unassigned_variables(assignment) to get a list of unassigned variables.
    - Use csp.domains[var] to get the list of possible values for a variable.
    - Use csp.get_neighbors(var) to get the list of variables that share a constraint with var.
    - Add logs to measure how good your implementation is (e.g. number of assignments, backtracks).

    You can find inspiration in the textbook's pseudocode:
    Artificial Intelligence: A Modern Approach (4th Edition) by Russell and Norvig, Chapter 5: Constraint Satisfaction Problems
    """
    if assignment is None:
        assignment = {}
    # Base case: all variables assigned
    if csp.is_complete(assignment):
        return assignment
    # Select next unassigned variable
    unassigned = csp.get_unassigned_variables(assignment)
    var = unassigned[0]
    # Try each value in the domain
    for value in csp.domains[var]:
        if csp.is_consistent(var, value, assignment):
            csp.assign(var, value, assignment)
            result = backtracking_search(csp, assignment)
            print("Esto es un result: ", result)
            if result is not None:
                return result
            # Backtrack
            print("Esto es un assignment: ", assignment)
            csp.unassign(var, assignment)

    return None

# def backtracking_fc(csp: DroneAssignmentCSP) -> dict[str, str] | None:
#     """
#     Backtracking search with Forward Checking.

#     Tips:
#     - Forward checking: After assigning a value to a variable, eliminate inconsistent values from
#       the domains of unassigned neighbors. If any neighbor's domain becomes empty, backtrack immediately.
#     - Save domains before forward checking so you can restore them on backtrack.
#     - Use csp.get_neighbors(var) to get variables that share constraints with var.
#     - Use csp.is_consistent(neighbor, val, assignment) to check if a value is still consistent.
#     - Forward checking reduces the search space by detecting failures earlier than basic backtracking.
#     """
#     assignment: dict[str, str] = {}

#     while True:
#         if csp.is_complete(assignment):
#             return assignment

#         var = csp.get_unassigned_variables(assignment)[0]
#         assigned = False

#         for value in csp.domains[var]:
#             if not csp.is_consistent(var, value, assignment):
#                 continue

#             saved_domains = {v: list(csp.domains[v]) for v in csp.variables}
#             csp.assign(var, value, assignment)

#             pruned_ok = True
#             for neighbor in csp.get_neighbors(var):
#                 if neighbor in assignment:
#                     continue
#                 csp.domains[neighbor] = [
#                     v for v in csp.domains[neighbor]
#                     if csp.is_consistent(neighbor, v, assignment)
#                 ]
#                 if not csp.domains[neighbor]:
#                     pruned_ok = False
#                     break

#             if pruned_ok:
#                 assigned = True
#                 break

#             csp.domains = saved_domains
#             csp.unassign(var, assignment)

#         if not assigned:
#             return None
  
def backtracking_fc(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    assignment: dict[str, str] = {}

    def backtrack() -> dict[str, str] | None:

        if csp.is_complete(assignment):
            return assignment

        var = csp.get_unassigned_variables(assignment)[0]

        for value in csp.domains[var]:

            if not csp.is_consistent(var, value, assignment):
                continue

            # Guardar dominios
            saved_domains = {v: list(csp.domains[v]) for v in csp.variables}

            csp.assign(var, value, assignment)

            if forward_check(var):

                result = backtrack()

                if result is not None:
                    return result

            # restaurar dominios y deshacer asignación
            csp.domains = saved_domains
            csp.unassign(var, assignment)

        return None

    def forward_check(var: str) -> bool:

        for neighbor in csp.get_neighbors(var):

            if neighbor in assignment:
                continue

            new_domain = [
                val for val in csp.domains[neighbor]
                if csp.is_consistent(neighbor, val, assignment)
            ]

            if not new_domain:
                return False

            csp.domains[neighbor] = new_domain

        return True

    return backtrack()

def backtracking_ac3(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Backtracking search with AC-3 arc consistency.

    Tips:
    - AC-3 enforces arc consistency: for every pair of constrained variables (Xi, Xj), every value
      in Xi's domain must have at least one supporting value in Xj's domain.
    - Run AC-3 before starting backtracking to reduce domains globally.
    - After each assignment, run AC-3 on arcs involving the assigned variable's neighbors.
    - If AC-3 empties any domain, the current assignment is inconsistent - backtrack.
    - You can create helper functions such as:
      - a values_compatible function to check if two variable-value pairs are consistent with the constraints.
      - a revise function that removes unsupported values from one variable's domain.
      - an ac3 function that manages the queue of arcs to check and calls revise.
      - a backtrack function that integrates AC-3 into the search process.
    """
    saved = {v: list(csp.domains[v]) for v in csp.variables}
    if not _ac3(csp):
        csp.domains = saved  # restore if AC-3 already finds inconsistency
        return None
    return _backtrack_ac3({}, csp)


def _values_compatible(csp: DroneAssignmentCSP, xi: str, vi: str, xj: str, vj: str) -> bool:
    """Check if xi=vi and xj=vj can coexist (no constraint violation)."""
    temp = {xj: vj}
    return csp.is_consistent(xi, vi, temp)


def _revise(csp: DroneAssignmentCSP, xi: str, xj: str) -> bool:
    """
    Remove values from domain of xi that have no support in xj's domain.
    Returns True if the domain of xi was changed.
    """
    revised = False
    for vi in list(csp.domains[xi]):
        # vi needs at least one compatible value in xj's domain
        if not any(_values_compatible(csp, xi, vi, xj, vj) for vj in csp.domains[xj]):
            csp.domains[xi].remove(vi)
            revised = True
    return revised

def _ac3(csp: DroneAssignmentCSP, arcs: list[tuple[str, str]] | None = None) -> bool:
    """
    Enforce arc consistency. Returns False if any domain is emptied.
    If arcs is None, initialises with all constraint arcs.
    """
    if arcs is None:
        queue = [
            (xi, xj)
            for xi in csp.variables
            for xj in csp.get_neighbors(xi)
        ]
    else:
        queue = list(arcs)

    while queue:
        xi, xj = queue.pop(0)
        if _revise(csp, xi, xj):
            if not csp.domains[xi]:
                return False  # Domain wiped → inconsistency
            # Re-check all neighbors of xi (except xj)
            for xk in csp.get_neighbors(xi):
                if xk != xj:
                    queue.append((xk, xi))
    return True


def _backtrack_ac3(assignment: dict[str, str], csp: DroneAssignmentCSP) -> dict[str, str] | None:
    if csp.is_complete(assignment):
        return assignment

    var = csp.get_unassigned_variables(assignment)[0]

    for value in list(csp.domains[var]):
        if not csp.is_consistent(var, value, assignment):
            continue

        saved_domains = {v: list(csp.domains[v]) for v in csp.variables}
        csp.assign(var, value, assignment)

        # Restrict domain of var to just the chosen value, then run AC-3
        csp.domains[var] = [value]
        neighbor_arcs = [(nb, var) for nb in csp.get_neighbors(var) if nb not in assignment]

        if _ac3(csp, neighbor_arcs):
            result = _backtrack_ac3(assignment, csp)
            if result is not None:
                return result

        csp.domains = saved_domains
        csp.unassign(var, assignment)

    return None


def backtracking_mrv_lcv(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Backtracking with Forward Checking + MRV + LCV.

    Tips:
    - Combine the techniques from backtracking_fc, mrv_heuristic, and lcv_heuristic.
    - MRV (Minimum Remaining Values): Select the unassigned variable with the fewest legal values.
      Tie-break by degree: prefer the variable with the most unassigned neighbors.
    - LCV (Least Constraining Value): When ordering values for a variable, prefer
      values that rule out the fewest choices for neighboring variables.
    - Use csp.get_num_conflicts(var, value, assignment) to count how many values would be ruled out for neighbors if var=value is assigned.
    """
    return _backtrack_mrv_lcv({}, csp)


def _select_mrv(csp: DroneAssignmentCSP, assignment: dict[str, str]) -> str:
    """
    MRV: pick unassigned variable with fewest legal values.
    Tie-break: most unassigned neighbors (degree heuristic).
    """
    unassigned = csp.get_unassigned_variables(assignment)

    def mrv_key(var: str) -> tuple[int, int]:
        remaining = len(csp.domains[var])
        degree = sum(1 for nb in csp.get_neighbors(var) if nb not in assignment)
        return (remaining, -degree)  # fewer values first, more neighbors first

    return min(unassigned, key=mrv_key)


def _order_lcv(csp: DroneAssignmentCSP, var: str, assignment: dict[str, str]) -> list[str]:
    """
    LCV: order values by least number of conflicts introduced in neighbors.
    """
    return sorted(
        csp.domains[var],
        key=lambda val: csp.get_num_conflicts(var, val, assignment)
    )


def _backtrack_mrv_lcv(assignment: dict[str, str], csp: DroneAssignmentCSP) -> dict[str, str] | None:
    if csp.is_complete(assignment):
        return assignment

    var = _select_mrv(csp, assignment)

    for value in _order_lcv(csp, var, assignment):
        if not csp.is_consistent(var, value, assignment):
            continue

        saved_domains = {v: list(csp.domains[v]) for v in csp.variables}
        csp.assign(var, value, assignment)

        # Forward checking
        pruned_ok = True
        for neighbor in csp.get_neighbors(var):
            if neighbor in assignment:
                continue
            csp.domains[neighbor] = [
                v for v in csp.domains[neighbor]
                if csp.is_consistent(neighbor, v, assignment)
            ]
            if not csp.domains[neighbor]:
                pruned_ok = False
                break

        if pruned_ok:
            result = _backtrack_mrv_lcv(assignment, csp)
            if result is not None:
                return result

        csp.domains = saved_domains
        csp.unassign(var, assignment)

    return None
