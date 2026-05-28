import numpy as np
import numpy.typing as npt
from typing import Dict, List, Set
from src_2206596_2122457.src.util.graph import Graph


def prunable_from_inferiority_graph(graph: Graph) -> npt.NDArray[np.bool_]:
    n_cells: int = graph.num_vertices()
    trans_graph: Graph = graph.transpose()
    visited: npt.NDArray[np.bool_] = np.zeros(n_cells, dtype=bool)
    stack: List[int] = []
    
    for v in trans_graph.vertices:
        if not visited[v]:
            dfs_collect(trans_graph, v, visited, stack)
    
    prunable_mask: npt.NDArray[np.bool_] = np.zeros(n_cells, dtype=bool)
    visited[:] = False
    killing: npt.NDArray[np.bool_] = np.zeros(n_cells, dtype=bool)
    
    for v in reversed(stack):
        if visited[v]:
            prunable_mask[v] = True
            continue
        
        reached: npt.NDArray[np.bool_] = np.zeros(n_cells, dtype=bool)
        found: bool = dfs_with_accumulate(graph, v, reached, killing)
        
        if found:
            prunable_mask[v] = True
        
        killing |= reached
        visited[:] = False
    
    return prunable_mask


def dfs_collect(
    graph: Graph,
    start: int,
    visited: npt.NDArray[np.bool_],
    stack: List[int],
) -> None:
    local_stack: List[int] = [start]
    
    while local_stack:
        v: int = local_stack[-1]
        
        if visited[v]:
            local_stack.pop()
            continue
        
        visited[v] = True
        
        out_neighbors: npt.NDArray[np.int_] = graph.out_set(v)
        for u in out_neighbors:
            if not visited[u]:
                local_stack.append(int(u))
        
        if local_stack and local_stack[-1] == v:
            stack.append(v)
            local_stack.pop()


def dfs_with_accumulate(
    graph: Graph,
    start: int,
    reached: npt.NDArray[np.bool_],
    killing: npt.NDArray[np.bool_],
) -> bool:
    local_stack: List[int] = [start]
    local_visited: npt.NDArray[np.bool_] = np.zeros(graph.num_vertices(), dtype=bool)
    found_killing: bool = False
    
    while local_stack:
        v: int = local_stack[-1]
        
        if local_visited[v]:
            local_stack.pop()
            continue
        
        local_visited[v] = True
        reached[v] = True
        
        if killing[v]:
            found_killing = True
        
        out_neighbors: npt.NDArray[np.int_] = graph.out_set(v)
        for u in out_neighbors:
            u_int: int = int(u)
            if not local_visited[u_int]:
                local_stack.append(u_int)
        
        if local_stack and local_stack[-1] == v:
            local_stack.pop()
    
    return found_killing


class InferiorCells:
    def __init__(self, n_cells: int) -> None:
        self.n_cells: int = n_cells
        self.m_fillin: List[npt.NDArray[np.bool_]] = [
            np.zeros(n_cells, dtype=bool),
            np.zeros(n_cells, dtype=bool),
        ]
        self.m_vulnerable: npt.NDArray[np.bool_] = np.zeros(n_cells, dtype=bool)
        self.m_killers: List[Set[int]] = [set() for _ in range(n_cells)]
        self.m_s_reversible: npt.NDArray[np.bool_] = np.zeros(n_cells, dtype=bool)
        self.m_s_reversers: List[Set[int]] = [set() for _ in range(n_cells)]
        self.m_blockers: npt.NDArray[np.bool_] = np.zeros(n_cells, dtype=bool)
        self.m_s_reversible_carriers: npt.NDArray[np.bool_] = np.zeros(n_cells, dtype=bool)
        self.m_inf_graph: Graph = Graph()
        self.m_inferior_computed: bool = False
        self.m_inferior: npt.NDArray[np.bool_] = np.zeros(n_cells, dtype=bool)

    def fillin(self, color: int) -> npt.NDArray[np.bool_]:
        ci: int = 0 if color == 1 else 1
        return self.m_fillin[ci].copy()

    def vulnerable(self) -> npt.NDArray[np.bool_]:
        return self.m_vulnerable.copy()

    def s_reversible(self) -> npt.NDArray[np.bool_]:
        return self.m_s_reversible.copy()

    def killers(self, p: int) -> Set[int]:
        return set(self.m_killers[p])

    def s_reversers(self, p: int) -> Set[int]:
        return set(self.m_s_reversers[p])

    def blockers(self) -> npt.NDArray[np.bool_]:
        return self.m_blockers.copy()

    def s_reversible_carriers(self) -> npt.NDArray[np.bool_]:
        return self.m_s_reversible_carriers.copy()

    def all(self) -> npt.NDArray[np.bool_]:
        result: npt.NDArray[np.bool_] = (
            self.m_fillin[0]
            | self.m_fillin[1]
            | self.m_vulnerable
            | self.m_s_reversible
            | self.inferior()
        )
        return result

    def inferior(self) -> npt.NDArray[np.bool_]:
        if self.m_inferior_computed:
            return self.m_inferior.copy()
        
        g: Graph = self._copy_graph(self.m_inf_graph)
        
        to_remove: npt.NDArray[np.bool_] = self.m_vulnerable | self.m_s_reversible
        for v in np.nonzero(to_remove)[0]:
            g.remove_vertex(int(v))
        
        pruned_result = prunable_from_inferiority_graph(g)
        
        if pruned_result.size == 0 or pruned_result.shape[0] != self.n_cells:
            self.m_inferior = np.zeros(self.n_cells, dtype=bool)
        else:
            self.m_inferior = pruned_result
        
        self.m_inferior_computed = True
        
        return self.m_inferior.copy()

    def _copy_graph(self, graph: Graph) -> Graph:
        new_graph: Graph = Graph()
        for v in graph.vertices:
            new_graph.add_vertex(int(v))
        
        for v in graph.vertices:
            out_neighbors: npt.NDArray[np.int_] = graph.out_set(int(v))
            for u in out_neighbors:
                new_graph.add_edge(int(v), int(u))
        
        return new_graph

    def add_fillin(self, color: int, fillin: npt.NDArray[np.bool_]) -> None:
        ci: int = 0 if color == 1 else 1
        self.m_fillin[ci] |= fillin
        for v in np.nonzero(fillin)[0]:
            v_int: int = int(v)
            self.remove_vulnerable(v_int)
            self.remove_s_reversible(v_int)
            self.remove_inferior(v_int)

    def add_vulnerable(self, vulnerable: int, killer: Set[int]) -> None:
        self.m_vulnerable[vulnerable] = True
        self.m_killers[vulnerable].update(killer)
        self.remove_s_reversible(vulnerable)
        self.m_inferior_computed = False

    def add_vulnerable_from_mask(
        self, vulnerable_mask: npt.NDArray[np.bool_], killers_by_index: Dict[int, Set[int]]
    ) -> None:
        for v in np.nonzero(vulnerable_mask)[0]:
            v_int: int = int(v)
            self.add_vulnerable(v_int, killers_by_index[v_int])

    def add_s_reversible(
        self,
        reversible: int,
        carrier_mask: npt.NDArray[np.bool_],
        reverser: int,
        is_threat: bool,
    ) -> None:
        if self.m_vulnerable[reversible] or self.m_s_reversible[reversible]:
            return
        
        cond1: bool = (
            not self.m_s_reversible_carriers[reverser]
            and (not is_threat or not self.m_s_reversible_carriers[reversible])
        )
        cond2: bool = not (carrier_mask & self.m_blockers).any()
        
        if cond1 or cond2:
            self.m_blockers[reverser] = True
            if is_threat:
                self.m_blockers[reversible] = True
            self.m_s_reversible_carriers |= carrier_mask
            self.m_s_reversible[reversible] = True
            self.m_s_reversers[reversible].add(reverser)
            self.m_inferior_computed = False

    def add_inferior(self, inferior: int, superior: Set[int]) -> None:
        if not self.m_inf_graph.vertex_exists(inferior):
            self.m_inf_graph.add_vertex(inferior)
        
        for sup in superior:
            if not self.m_inf_graph.vertex_exists(sup):
                self.m_inf_graph.add_vertex(sup)
            self.m_inf_graph.add_edge(inferior, sup)
        
        self.m_inferior_computed = False

    def add_vulnerable_from(self, other: 'InferiorCells') -> None:
        for i in np.nonzero(other.m_vulnerable)[0]:
            i_int: int = int(i)
            self.add_vulnerable(i_int, other.m_killers[i_int])

    def add_s_reversible_from(self, other: 'InferiorCells') -> None:
        self.m_s_reversible |= other.m_s_reversible
        for i in np.nonzero(other.m_s_reversible)[0]:
            idx: int = int(i)
            self.m_s_reversers[idx].update(other.m_s_reversers[idx])
        self.m_blockers |= other.m_blockers
        self.m_s_reversible_carriers |= other.m_s_reversible_carriers
        self.m_inferior_computed = False

    def add_inferior_from(self, other: 'InferiorCells') -> None:
        for v in other.m_inf_graph.vertices:
            v_int: int = int(v)
            out_neighbors: npt.NDArray[np.int_] = other.m_inf_graph.out_set(v_int)
            self.add_inferior(v_int, set(int(u) for u in out_neighbors))

    def clear(self) -> None:
        self.clear_fillin(1)
        self.clear_fillin(2)
        self.clear_vulnerable()
        self.clear_s_reversible()
        self.clear_inferior()

    def clear_fillin(self, color: int) -> None:
        ci: int = 0 if color == 1 else 1
        self.m_fillin[ci][:] = False

    def clear_vulnerable(self) -> None:
        self.remove_vulnerable_mask(self.m_vulnerable.copy())
        self.m_inferior_computed = False

    def clear_s_reversible(self) -> None:
        self.remove_s_reversible_mask(self.m_s_reversible.copy())
        self.m_blockers[:] = False
        self.m_s_reversible_carriers[:] = False
        self.m_inferior_computed = False

    def clear_inferior(self) -> None:
        self.m_inf_graph.clear()
        self.m_inferior_computed = False

    def remove_vulnerable(self, vulnerable: int) -> None:
        if self.m_vulnerable[vulnerable]:
            self.m_killers[vulnerable].clear()
            self.m_vulnerable[vulnerable] = False
            self.m_inferior_computed = False

    def remove_vulnerable_mask(self, vulnerable: npt.NDArray[np.bool_]) -> None:
        to_remove: npt.NDArray[np.bool_] = vulnerable & self.m_vulnerable
        for v in np.nonzero(to_remove)[0]:
            self.m_killers[int(v)].clear()
        self.m_vulnerable = self.m_vulnerable & ~vulnerable
        self.m_inferior_computed = False

    def remove_s_reversible(self, reversible: int) -> None:
        if self.m_s_reversible[reversible]:
            self.m_s_reversers[reversible].clear()
            self.m_s_reversible[reversible] = False
            self.m_inferior_computed = False

    def remove_s_reversible_mask(self, reversible: npt.NDArray[np.bool_]) -> None:
        to_remove: npt.NDArray[np.bool_] = reversible & self.m_s_reversible
        for v in np.nonzero(to_remove)[0]:
            self.m_s_reversers[int(v)].clear()
        self.m_s_reversible = self.m_s_reversible & ~reversible
        self.m_inferior_computed = False

    def remove_inferior(self, inferior: int) -> None:
        if self.m_inf_graph.vertex_exists(inferior):
            self.m_inf_graph.remove_vertex(inferior)
            self.m_inferior_computed = False

    def remove_inferior_mask(self, inferior: npt.NDArray[np.bool_]) -> None:
        to_remove_indices: npt.NDArray[np.intp] = np.nonzero(inferior)[0]
        for v in to_remove_indices:
            v_int: int = int(v)
            if self.m_inf_graph.vertex_exists(v_int):
                self.m_inf_graph.remove_vertex(v_int)
        self.m_inferior_computed = False

    def find_presimplicial_pairs(self) -> npt.NDArray[np.bool_]:
        fillin: npt.NDArray[np.bool_] = np.zeros(self.n_cells, dtype=bool)
        vulnerable_indices: npt.NDArray[np.intp] = np.nonzero(self.m_vulnerable)[0]
        for x_idx in vulnerable_indices:
            x_int: int = int(x_idx)
            for y in self.m_killers[x_int]:
                if x_int in self.m_killers[y]:
                    fillin[x_int] = True
                    fillin[y] = True
                    break
        return fillin.copy()