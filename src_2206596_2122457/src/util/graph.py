import numpy as np
import numpy.typing as npt


class Graph:
    
    def __init__(self) -> None:
        self.vertices: npt.NDArray[np.int_] = np.array([], dtype=int)
        self.out_edges: npt.NDArray[np.bool_] = np.empty((0, 0), dtype=bool)
        self.in_edges: npt.NDArray[np.bool_] = np.empty((0, 0), dtype=bool)
    
    def vertex_exists(self, vertex: int) -> bool:
        return bool(np.isin(vertex, self.vertices))
    
    def add_vertex(self, vertex: int) -> None:
        if self.vertex_exists(vertex):
            return
        
        self.vertices = np.concatenate((self.vertices, np.array([vertex], dtype=int)))
        
        new_out_edges = np.pad(self.out_edges, ((0, 1), (0, 1)), constant_values=False)
        new_in_edges = np.pad(self.in_edges, ((0, 1), (0, 1)), constant_values=False)
        
        self.out_edges = new_out_edges
        self.in_edges = new_in_edges
    
    def remove_vertex(self, vertex: int) -> None:
        if not self.vertex_exists(vertex):
            return
        
        idx = np.where(self.vertices == vertex)[0][0]
        
        self.out_edges = np.delete(np.delete(self.out_edges, idx, axis=0), idx, axis=1)
        self.in_edges = np.delete(np.delete(self.in_edges, idx, axis=0), idx, axis=1)
        
        self.vertices = np.delete(self.vertices, idx)
    
    def add_edge(self, source: int, target: int) -> None:
        if not self.vertex_exists(source) or not self.vertex_exists(target):
            return
        
        source_idx: int = int(np.where(self.vertices == source)[0][0])
        target_idx: int = int(np.where(self.vertices == target)[0][0])
        
        self.out_edges[source_idx, target_idx] = True
        self.in_edges[target_idx, source_idx] = True
    
    def add_edges(self, source: int, targets: npt.NDArray[np.int_]) -> None:
        if not self.vertex_exists(source):
            return
        
        source_idx: int = int(np.where(self.vertices == source)[0][0])
        target_indices: npt.NDArray[np.intp] = np.where(np.isin(self.vertices, targets))[0]
        
        self.out_edges[source_idx, target_indices] = True
        self.in_edges[target_indices, source_idx] = True
    
    def remove_edge(self, source: int, target: int) -> None:
        if not self.vertex_exists(source) or not self.vertex_exists(target):
            return
        
        source_idx: int = int(np.where(self.vertices == source)[0][0])
        target_idx: int = int(np.where(self.vertices == target)[0][0])
        
        self.out_edges[source_idx, target_idx] = False
        self.in_edges[target_idx, source_idx] = False
    
    def num_vertices(self) -> int:
        return int(self.vertices.size)
    
    def is_edge(self, source: int, target: int) -> bool:
        if not self.vertex_exists(source) or not self.vertex_exists(target):
            return False
        
        source_idx: int = int(np.where(self.vertices == source)[0][0])
        target_idx: int = int(np.where(self.vertices == target)[0][0])
        
        return bool(self.out_edges[source_idx, target_idx])
    
    def is_isolated(self, vertex: int) -> bool:
        if not self.vertex_exists(vertex):
            return False
        
        idx: int = int(np.where(self.vertices == vertex)[0][0])
        
        return not (np.any(self.out_edges[idx, :]) or np.any(self.in_edges[idx, :]))
    
    def in_degree(self, vertex: int) -> int:
        if not self.vertex_exists(vertex):
            return 0
        
        idx: int = int(np.where(self.vertices == vertex)[0][0])
        return int(np.count_nonzero(self.in_edges[idx, :]))
    
    def out_degree(self, vertex: int) -> int:
        if not self.vertex_exists(vertex):
            return 0
        
        idx: int = int(np.where(self.vertices == vertex)[0][0])
        return int(np.count_nonzero(self.out_edges[idx, :]))
    
    def sources(self) -> npt.NDArray[np.int_]:
        has_out: npt.NDArray[np.bool_] = np.any(self.out_edges, axis=1)
        has_in: npt.NDArray[np.bool_] = np.any(self.in_edges, axis=1)
        return self.vertices[np.logical_and(has_out, ~has_in)]
    
    def sinks(self) -> npt.NDArray[np.int_]:
        has_in: npt.NDArray[np.bool_] = np.any(self.in_edges, axis=1)
        has_out: npt.NDArray[np.bool_] = np.any(self.out_edges, axis=1)
        return self.vertices[np.logical_and(has_in, ~has_out)]
    
    def out_set(self, vertex: int) -> npt.NDArray[np.int_]:
        if not self.vertex_exists(vertex):
            return np.array([], dtype=int)
        
        idx: int = int(np.where(self.vertices == vertex)[0][0])
        return self.vertices[self.out_edges[idx, :]]
    
    def in_set(self, vertex: int) -> npt.NDArray[np.int_]:
        if not self.vertex_exists(vertex):
            return np.array([], dtype=int)
        
        idx: int = int(np.where(self.vertices == vertex)[0][0])
        return self.vertices[self.in_edges[idx, :]]
    
    def out_set_multiple(self, vertices_subset: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        if vertices_subset.size == 0:
            return np.array([], dtype=int)
        
        subset_indices: npt.NDArray[np.intp] = np.where(np.isin(self.vertices, vertices_subset))[0]
        
        if subset_indices.size == 0:
            return np.array([], dtype=int)
        
        union_mask: npt.NDArray[np.bool_] = np.any(self.out_edges[subset_indices, :], axis=0)
        return self.vertices[union_mask]
    
    def in_set_multiple(self, vertices_subset: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        if vertices_subset.size == 0:
            return np.array([], dtype=int)
        
        subset_indices: npt.NDArray[np.intp] = np.where(np.isin(self.vertices, vertices_subset))[0]
        
        if subset_indices.size == 0:
            return np.array([], dtype=int)
        
        union_mask: npt.NDArray[np.bool_] = np.any(self.in_edges[subset_indices, :], axis=0)
        return self.vertices[union_mask]
    
    def transpose(self) -> 'Graph':
        new_graph: Graph = Graph()
        new_graph.vertices = self.vertices.copy()
        new_graph.out_edges = self.in_edges.copy()
        new_graph.in_edges = self.out_edges.copy()
        return new_graph
    
    def dfs(self, vertex: int, visited: npt.NDArray[np.bool_], killing: npt.NDArray[np.int_], 
            stack: list[int]) -> bool:
        if not self.vertex_exists(vertex):
            return False
        
        local_stack: list[int] = [vertex]
        
        while local_stack:
            current: int = local_stack[-1]
            current_idx: int = int(np.where(self.vertices == current)[0][0])
            
            if visited[current_idx]:
                local_stack.pop()
                continue
            
            visited[current_idx] = True
            
            if np.isin(current, killing):
                stack.append(current)
                return True
            
            neighbor_mask: npt.NDArray[np.bool_] = self.out_edges[current_idx, :] & ~visited
            neighbor_indices: npt.NDArray[np.intp] = np.where(neighbor_mask)[0]
            
            for neighbor_idx in neighbor_indices:
                neighbor: int = int(self.vertices[neighbor_idx])
                if not visited[neighbor_idx]:
                    local_stack.append(neighbor)
            
            if local_stack[-1] == current:
                stack.append(current)
                local_stack.pop()
        
        return False
    
    def find_two_cycles(self) -> npt.NDArray[np.int_]:
        cycles: list[list[int]] = []
        
        for i, x in enumerate(self.vertices):
            targets: npt.NDArray[np.intp] = np.where(self.out_edges[i, :])[0]
            
            for j in targets:
                if self.out_edges[j, i]:
                    cycles.append([int(x), int(self.vertices[j])])
        
        if not cycles:
            return np.array([], dtype=int)
        
        return np.unique(np.array(cycles, dtype=int).flatten())
    
    def clear(self) -> None:
        self.vertices = np.array([], dtype=int)
        self.out_edges = np.empty((0, 0), dtype=bool)
        self.in_edges = np.empty((0, 0), dtype=bool)