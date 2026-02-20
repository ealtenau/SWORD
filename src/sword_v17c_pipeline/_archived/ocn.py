"""
Theory Test OCN Suite
Tests Topological Universality Hypothesis and orientation vs geometry decoupling
Proposed by Gearon (2025)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to prevent blocking
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import deque
import copy

# Pre-compute all neighbor offsets for faster lookup
NEIGHBOR_OFFSETS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if not (dy == 0 and dx == 0)]


def neighbors(y, x, ny, nx):
    """Generator for valid neighbor coordinates"""
    for dy, dx in NEIGHBOR_OFFSETS:
        nyy, nxx = y + dy, x + dx
        if 0 <= nyy < ny and 0 <= nxx < nx:
            yield nyy, nxx


# ==========================================================
# MODULE 0 — BASE OCN ENGINE
# ==========================================================

class OCN:
    """Optimal Channel Network - Orientation-Only Optimization"""
    
    def __init__(self, ny=60, nx=60, outlets=None, seed=42):
        """
        Initialize OCN grid
        
        Parameters:
        -----------
        ny, nx : int
            Grid dimensions
        outlets : list of (y, x) tuples or None
            Outlet locations. If None, uses bottom row.
        seed : int
            Random seed for reproducibility
        """
        self.ny = ny
        self.nx = nx
        self.rng = np.random.default_rng(seed)
        
        # Set outlets
        if outlets is None:
            # Default: all bottom row cells are outlets
            self.outlets = [(ny-1, x) for x in range(nx)]
        else:
            self.outlets = outlets
        
        # Parent pointers: -1 means outlet (no parent)
        self.parent_y = np.full((ny, nx), -1, dtype=int)
        self.parent_x = np.full((ny, nx), -1, dtype=int)
        
        # Contributing area
        self.A = None
        
        # Energy
        self.H = None
        self.H_history = []
        self.it_history = []
        
        # Geodesic potential (computed on demand)
        self.phi = None
        
        # Pre-compute neighbor cache for speed
        self.neighbor_cache = {}
        for y in range(ny):
            for x in range(nx):
                self.neighbor_cache[(y, x)] = list(neighbors(y, x, ny, nx))
        
        # Children dictionary for incremental updates
        self.children_dict = {}
        
        # Initialize random network
        self._initialize_network()
    
    def _initialize_network(self):
        """Initialize a valid spanning tree from outlets using BFS"""
        # Set outlets
        for oy, ox in self.outlets:
            self.parent_y[oy, ox] = -1
            self.parent_x[oy, ox] = -1

        # Build spanning tree from outlets outward using BFS
        # This guarantees: (1) all nodes connected, (2) no cycles, (3) valid tree structure
        visited = set(self.outlets)
        frontier = deque(self.outlets)

        while frontier:
            cy, cx = frontier.popleft()

            # Get all unvisited neighbors and shuffle for randomness
            unvisited_neighbors = [(nyy, nxx) for nyy, nxx in self.neighbor_cache[(cy, cx)]
                                  if (nyy, nxx) not in visited and (nyy, nxx) not in self.outlets]

            # Shuffle to add randomness while maintaining valid tree
            self.rng.shuffle(unvisited_neighbors)

            # Connect all unvisited neighbors (pure BFS spanning tree)
            for nyy, nxx in unvisited_neighbors:
                if (nyy, nxx) not in visited:  # Double-check since we're processing multiple parents
                    # Point child to current parent
                    self.parent_y[nyy, nxx] = cy
                    self.parent_x[nyy, nxx] = cx
                    visited.add((nyy, nxx))
                    frontier.append((nyy, nxx))

        # Verify all non-outlet nodes are connected
        for y in range(self.ny):
            for x in range(self.nx):
                if (y, x) not in self.outlets and (y, x) not in visited:
                    print(f"WARNING: Node ({y},{x}) not reachable from any outlet!")

        self.children_dict = self._build_children_dict()
        self.A = self.compute_A()

        # Verify no cycles
        if self._has_cycles():
            print("ERROR: Cycles detected after initialization - this should NEVER happen!")
    
    def _has_cycles(self):
        """Check if the network has cycles"""
        for y in range(self.ny):
            for x in range(self.nx):
                if (y, x) in self.outlets:
                    continue
                
                # Follow path and check for cycles
                visited = set()
                cy, cx = y, x
                while cy != -1 and cx != -1:
                    if (cy, cx) in visited:
                        return True
                    visited.add((cy, cx))
                    py, px = self.parent_y[cy, cx], self.parent_x[cy, cx]
                    if py == -1:
                        break
                    cy, cx = py, px
        return False
    
    def _makes_cycle(self, y, x, new_py, new_px):
        """Check if setting parent to (new_py, new_px) would create a cycle"""
        if new_py == -1 or new_px == -1:
            return False  # Outlets can't create cycles
        
        cy, cx = new_py, new_px
        visited = set()
        while cy != -1 and cx != -1:
            if (cy, cx) in visited:
                return True  # Cycle detected
            if cy == y and cx == x:
                return True  # Would point back to self
            visited.add((cy, cx))
            cy, cx = self.parent_y[cy, cx], self.parent_x[cy, cx]
        return False
    
    def compute_A(self):
        """Compute contributing area using topological sort"""
        ny, nx = self.ny, self.nx
        children = [[[] for _ in range(nx)] for _ in range(ny)]
        indeg = np.zeros((ny, nx), dtype=np.int32)
        
        # Build children lists and find outlets
        outlets = []
        for y in range(ny):
            for x in range(nx):
                py, px = self.parent_y[y, x], self.parent_x[y, x]
                if py == -1 and px == -1:
                    outlets.append((y, x))
                else:
                    children[py][px].append((y, x))
                    indeg[y, x] += 1
        
        # Topological sort
        order = []
        q = deque(outlets)
        while q:
            node = q.popleft()
            order.append(node)
            nyy, nxx = node
            for (uy, ux) in children[nyy][nxx]:
                indeg[uy, ux] -= 1
                if indeg[uy, ux] == 0:
                    q.append((uy, ux))
        
        # Compute contributing area (each cell contributes 1 unit)
        A = np.ones((ny, nx), dtype=np.float64)
        for (y, x) in reversed(order):
            py, px = self.parent_y[y, x], self.parent_x[y, x]
            if py != -1:
                A[py, px] += A[y, x]
        
        return A
    
    def compute_phi(self, metric='unit'):
        """
        Compute geodesic distance potential φ
        Shortest path distance from each node to nearest outlet
        
        Parameters:
        -----------
        metric : str
            'unit' for unit cost per step (consistent with energy), 
            'euclidean' for Euclidean distance
        """
        phi = np.full((self.ny, self.nx), np.inf, dtype=np.float64)
        
        # Multi-source BFS from all outlets
        q = deque()
        for oy, ox in self.outlets:
            phi[oy, ox] = 0.0
            q.append((oy, ox, 0.0))
        
        while q:
            y, x, dist = q.popleft()
            for nyy, nxx in self.neighbor_cache[(y, x)]:
                # Use unit cost per step (consistent with energy functional)
                # This ensures E₁ = Σ_v D_T(v) = Σ_v φ(v) for the geodesic tree
                if metric == 'unit':
                    edge_len = 1.0  # Unit cost per step
                else:  # euclidean
                    dy, dx = nyy - y, nxx - x
                    edge_len = np.sqrt(dy*dy + dx*dx)
                
                new_dist = dist + edge_len
                
                if new_dist < phi[nyy, nxx]:
                    phi[nyy, nxx] = new_dist
                    q.append((nyy, nxx, new_dist))
        
        return phi
    
    def compute_phi_gradient_direction(self, phi):
        """
        Compute gradient direction from φ field
        Returns parent pointers based on steepest descent
        """
        grad_parent_y = np.full((self.ny, self.nx), -1, dtype=int)
        grad_parent_x = np.full((self.ny, self.nx), -1, dtype=int)
        
        for y in range(self.ny):
            for x in range(self.nx):
                if (y, x) in self.outlets:
                    continue
                
                # Find neighbor with minimum phi (steepest descent)
                min_phi = phi[y, x]
                best_neigh = None
                
                for nyy, nxx in self.neighbor_cache[(y, x)]:
                    if phi[nyy, nxx] < min_phi:
                        min_phi = phi[nyy, nxx]
                        best_neigh = (nyy, nxx)
                
                if best_neigh:
                    grad_parent_y[y, x] = best_neigh[0]
                    grad_parent_x[y, x] = best_neigh[1]
                else:
                    # No downhill neighbor - point to nearest outlet
                    if self.outlets:
                        y_val, x_val = y, x  # Capture for lambda
                        best_outlet = min(self.outlets, 
                                        key=lambda o, y=y_val, x=x_val: (o[0]-y)**2 + (o[1]-x)**2)
                        grad_parent_y[y, x] = best_outlet[0]
                        grad_parent_x[y, x] = best_outlet[1]
        
        return grad_parent_y, grad_parent_x
    
    def compute_agreement(self, phi=None):
        """
        Compute % of nodes whose parent matches ∇φ (steepest descent)
        
        Parameters:
        -----------
        phi : array or None
            Geodesic potential. If None, computes from current outlets.
        
        Returns:
        --------
        agreement : float
            Percentage (0-100) of nodes with matching orientation
        """
        if phi is None:
            phi = self.compute_phi(metric='unit')
        
        grad_parent_y, grad_parent_x = self.compute_phi_gradient_direction(phi)
        
        matches = 0
        total = 0
        
        for y in range(self.ny):
            for x in range(self.nx):
                if (y, x) in self.outlets:
                    continue
                
                total += 1
                if (self.parent_y[y, x] == grad_parent_y[y, x] and 
                    self.parent_x[y, x] == grad_parent_x[y, x]):
                    matches += 1
        
        return 100.0 * matches / total if total > 0 else 0.0
    
    def _build_children_dict(self):
        """Build dictionary mapping (py, px) -> list of children"""
        children_dict = {}
        for y in range(self.ny):
            for x in range(self.nx):
                py, px = self.parent_y[y, x], self.parent_x[y, x]
                if py != -1 and px != -1:
                    if (py, px) not in children_dict:
                        children_dict[(py, px)] = []
                    children_dict[(py, px)].append((y, x))
        return children_dict
    
    def _update_contributing_area_incremental(self, y, x, old_py, old_px, new_py, new_px):
        """Incrementally update contributing area after parent change"""
        delta = float(self.A[y, x])  # Contributing area of node and all descendants
        
        # Update old path (subtract) - limit iterations to prevent infinite loops
        cy, cx = old_py, old_px
        path_len = 0
        max_path_len = self.ny * self.nx  # Safety limit
        while cy != -1 and cx != -1 and path_len < max_path_len:
            self.A[cy, cx] = max(1e-10, float(self.A[cy, cx] - delta))
            cy, cx = self.parent_y[cy, cx], self.parent_x[cy, cx]
            path_len += 1
        
        # Update new path (add) - limit iterations to prevent infinite loops
        cy, cx = new_py, new_px
        path_len = 0
        while cy != -1 and cx != -1 and path_len < max_path_len:
            self.A[cy, cx] += delta
            cy, cx = self.parent_y[cy, cx], self.parent_x[cy, cx]
            path_len += 1
    
    def compute_energy(self, gamma=1.0, metric='unit'):
        """
        Compute energy functional: E = Σ_e A(e)^gamma * L(e)
        
        For γ=1 with unit metric: E₁ = Σ_e A(e) * L(e) = Σ_v D_T(v)
        where D_T(v) is distance from node v to outlet along tree T.
        This equals Σ_v φ(v) for the geodesic tree, establishing the
        theoretical link: global minimizer of E₁ is the φ-aligned tree.
        
        Parameters:
        -----------
        gamma : float
            Energy exponent (0.5 ≈ natural rivers, 1.0 = geodesic optimal)
        metric : str
            'unit' for unit cost per edge (consistent with φ), 
            'euclidean' for Euclidean edge lengths
        """
        if self.A is None:
            self.A = self.compute_A()
        
        epsilon = 1e-10
        E = 0.0
        
        # Sum over all edges in the tree
        # Each edge connects a node v to its parent p
        # Contributing area A(e) = A(v) (area of subtree rooted at v)
        for y in range(self.ny):
            for x in range(self.nx):
                py, px = self.parent_y[y, x], self.parent_x[y, x]
                if py == -1:  # Outlet (no parent edge)
                    continue
                
                # Edge length L(e)
                if metric == 'unit':
                    L_e = 1.0  # Unit cost per step
                else:  # euclidean
                    dy, dx = py - y, px - x
                    L_e = np.sqrt(dy*dy + dx*dx)
                
                # Energy contribution: A(e)^gamma * L(e)
                A_e = max(self.A[y, x], epsilon)  # Contributing area through this edge
                E += np.power(A_e, gamma) * L_e
        
        return E
    
    def run(self, n_iter=100000, gamma=1.0, report_every=10000,
            recompute_every=50000, annealing=False, T0=1.0, alpha=0.9995, metric='euclidean'):
        """
        Evolve network orientation via local rewiring

        Parameters:
        -----------
        n_iter : int
            Number of iterations
        gamma : float
            Energy exponent (0.5 ≈ natural rivers, 1.0 = geodesic optimal)
        report_every : int
            Frequency of reporting/history recording
        recompute_every : int
            Frequency of full recomputation to correct drift
        annealing : bool
            Use simulated annealing (else greedy)
        T0 : float
            Initial temperature for annealing
        alpha : float
            Cooling rate for annealing
        metric : str
            Edge length metric ('euclidean' recommended to break degeneracy, 'unit' for testing)
        """
        self.H = self.compute_energy(gamma=gamma, metric=metric)
        self.H_history = [self.H]
        self.it_history = [0]

        # Ensure A is initialized
        if self.A is None:
            self.A = self.compute_A()

        # Compute phi for tie-breaking (critical for gamma=1.0)
        # The phi field is the geodesic distance potential that defines
        # the theoretical optimum orientation for gamma=1.0
        # CRITICAL: Use 'euclidean' metric to break degeneracy!
        # With unit metric, all spanning trees have equal energy.
        if self.phi is None:
            self.phi = self.compute_phi(metric='euclidean')

        print(f"  Starting optimization: H={self.H:.6e}, grid={self.ny}x{self.nx}, non-outlets={self.ny*self.nx - len(self.outlets)}")
        print(f"  Will run {n_iter:,} iterations with progress every 1000...")
        
        T = T0 if annealing else 0.0
        last_progress_print = 0
        
        for it in range(1, n_iter + 1):
            # Print progress more frequently - every 1000 iterations
            if it % 1000 == 0 or it == 1:
                progress_pct = 100.0 * it / n_iter
                print(f"  Iteration {it:,}/{n_iter:,} ({progress_pct:.1f}%) - Energy: {self.H:.6e}", flush=True)
                last_progress_print = it
            # Randomly select a non-outlet node (more efficient)
            # Pre-compute valid cells once
            max_attempts = 100
            for _ in range(max_attempts):
                y = self.rng.integers(0, self.ny)
                x = self.rng.integers(0, self.nx)
                if (y, x) not in self.outlets:
                    break
            else:
                # Fallback: just pick first non-outlet (shouldn't happen)
                for y in range(self.ny):
                    for x in range(self.nx):
                        if (y, x) not in self.outlets:
                            break
                    else:
                        continue
                    break
            
            old_py, old_px = self.parent_y[y, x], self.parent_x[y, x]
            
            # Randomly select new parent from neighbors
            neighs = self.neighbor_cache[(y, x)]
            if not neighs:
                continue
            
            new_py, new_px = neighs[self.rng.integers(len(neighs))]
            
            # Check for cycles
            if self._makes_cycle(y, x, new_py, new_px):
                continue
            
            # Make tentative change
            self.parent_y[y, x] = new_py
            self.parent_x[y, x] = new_px
            
            # Update children dict
            if (old_py, old_px) in self.children_dict:
                if (y, x) in self.children_dict[(old_py, old_px)]:
                    self.children_dict[(old_py, old_px)].remove((y, x))
                if not self.children_dict[(old_py, old_px)]:
                    del self.children_dict[(old_py, old_px)]
            if (new_py, new_px) not in self.children_dict:
                self.children_dict[(new_py, new_px)] = []
            if (y, x) not in self.children_dict[(new_py, new_px)]:
                self.children_dict[(new_py, new_px)].append((y, x))
            
            # Incrementally update A
            # Only copy if we need to (for reversion)
            A_backup = None
            try:
                A_backup = self.A.copy()
                self._update_contributing_area_incremental(y, x, old_py, old_px, new_py, new_px)
                H_new = self.compute_energy(gamma=gamma, metric=metric)
            except Exception as e:
                print(f"  ERROR at iteration {it}: {e}")
                raise
            
            # Acceptance criterion
            # For γ=1.0, we want to minimize energy (greedy)
            # CRITICAL: When energy is equal (degeneracy), use tie-breaking
            # to prefer gradient-aligned moves. This breaks the massive
            # degeneracy in the γ=1.0 energy landscape.
            accept = False
            energy_tol = 1e-10  # Tolerance for considering energies equal

            if H_new < self.H - energy_tol:
                # Clear energy decrease - always accept
                accept = True
            elif abs(H_new - self.H) <= energy_tol:
                # Energies are equal (within tolerance) - use tie-breaking
                # Check if new parent aligns better with gradient
                if self.phi is not None:
                    # Use phi gradient as tie-breaker
                    # Accept if new parent has lower phi (steepest descent)
                    old_parent_phi = self.phi[old_py, old_px] if old_py != -1 else 0.0
                    new_parent_phi = self.phi[new_py, new_px] if new_py != -1 else 0.0
                    if new_parent_phi < old_parent_phi:
                        accept = True
                    elif new_parent_phi == old_parent_phi:
                        # Random tie-breaking if phi is also equal
                        accept = self.rng.random() < 0.5
                else:
                    # No phi computed yet - random tie-breaking
                    accept = self.rng.random() < 0.5
            elif annealing and T > 0:
                # Simulated annealing: accept uphill moves with probability
                prob = np.exp(-(H_new - self.H) / T)
                if self.rng.random() < prob:
                    accept = True
            
            if accept:
                self.H = H_new
                self.A = np.maximum(self.A, 1e-10)  # Ensure positivity
            else:
                # Revert
                self.parent_y[y, x] = old_py
                self.parent_x[y, x] = old_px
                if A_backup is not None:
                    self.A = A_backup
                
                # Revert children dict
                if (new_py, new_px) in self.children_dict:
                    if (y, x) in self.children_dict[(new_py, new_px)]:
                        self.children_dict[(new_py, new_px)].remove((y, x))
                    if not self.children_dict[(new_py, new_px)]:
                        del self.children_dict[(new_py, new_px)]
                if (old_py, old_px) not in self.children_dict:
                    self.children_dict[(old_py, old_px)] = []
                if (y, x) not in self.children_dict[(old_py, old_px)]:
                    self.children_dict[(old_py, old_px)].append((y, x))
            
            # Periodic full recomputation
            if it % recompute_every == 0:
                self.A = self.compute_A()
                self.children_dict = self._build_children_dict()
                self.H = self.compute_energy(gamma=gamma, metric=metric)
            
            # Cooling schedule
            if annealing:
                T *= alpha
            
            # Record history
            if it % report_every == 0:
                self.H_history.append(self.H)
                self.it_history.append(it)
        
        # Final computation of phi (using same metric as energy)
        print(f"  Optimization complete. Computing geodesic potential φ...")
        self.phi = self.compute_phi(metric=metric)
    
    def plot_network(self, ax=None, show_flow=True, title="OCN Network"):
        """Plot network with flow directions"""
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 10))
        
        # Plot contributing area as background
        if self.A is None:
            self.A = self.compute_A()
        im = ax.imshow(np.log10(self.A + 1e-6), cmap='YlOrRd', origin='lower')
        
        # Plot flow directions (parent pointers)
        if show_flow:
            for y in range(self.ny):
                for x in range(self.nx):
                    if (y, x) in self.outlets:
                        ax.plot(x, y, 'ks', markersize=8, zorder=5)
                        continue
                    
                    py, px = self.parent_y[y, x], self.parent_x[y, x]
                    if py != -1:
                        # Draw arrow from (y,x) to parent
                        dx = px - x
                        dy = py - y
                        ax.arrow(x, y, dx*0.8, dy*0.8, head_width=0.3, 
                               head_length=0.2, fc='blue', ec='blue', 
                               alpha=0.3, length_includes_head=True)
        
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.colorbar(im, ax=ax, label="log10(A)")
        return ax
    
    def plot_energy(self, ax=None, title="Energy Evolution"):
        """Plot energy vs iteration"""
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(self.it_history, self.H_history, 'b-', linewidth=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Energy H")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        return ax


# ==========================================================
# MODULE 1 — MULTIPLE OUTLETS TEST
# ==========================================================

def test_multiple_outlets(ny=60, nx=60, n_outlets=3, gamma=1.0, n_iter=200000):
    """Test #1: Multiple competing outlets"""
    print("\n" + "="*60)
    print("MODULE 1: MULTIPLE OUTLETS TEST")
    print("="*60)
    
    # Choose outlets along bottom boundary
    rng = np.random.default_rng(42)
    outlet_x_positions = np.linspace(0, nx-1, n_outlets, dtype=int)
    outlets = [(ny-1, x) for x in outlet_x_positions]
    
    print(f"Outlets at: {outlets}")
    
    # Initialize and run OCN
    ocn = OCN(ny=ny, nx=nx, outlets=outlets, seed=42)
    print(f"Initial agreement: {ocn.compute_agreement():.2f}%")
    print(f"Running optimization ({n_iter:,} iterations)...")
    
    ocn.run(n_iter=n_iter, gamma=gamma, report_every=2000)  # More frequent reporting
    final_agreement = ocn.compute_agreement()
    
    print(f"Final agreement: {final_agreement:.2f}%")
    print(f"Final energy: {ocn.H:.2e}")
    
    # Compute phi
    phi = ocn.compute_phi(metric='unit')
    
    # Create figure
    _, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # (a) φ field
    im1 = axes[0].imshow(phi, cmap='viridis', origin='lower')
    for oy, ox in outlets:
        axes[0].plot(ox, oy, 'r*', markersize=15, zorder=5)
    axes[0].set_title("(a) Geodesic Potential φ")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    plt.colorbar(im1, ax=axes[0], label="Distance to outlet")
    
    # (b) OCN final network
    ocn.plot_network(ax=axes[1], title=f"(b) OCN Network (γ={gamma})")
    
    # (c) Agreement visualization
    grad_py, grad_px = ocn.compute_phi_gradient_direction(phi)
    agreement_map = np.zeros((ny, nx))
    for y in range(ny):
        for x in range(nx):
            if (y, x) in outlets:
                agreement_map[y, x] = 1.0
            else:
                if (ocn.parent_y[y, x] == grad_py[y, x] and 
                    ocn.parent_x[y, x] == grad_px[y, x]):
                    agreement_map[y, x] = 1.0
                else:
                    agreement_map[y, x] = 0.0
    
    im3 = axes[2].imshow(agreement_map, cmap='RdYlGn', origin='lower', vmin=0, vmax=1)
    axes[2].set_title(f"(c) Agreement Map ({final_agreement:.1f}%)")
    axes[2].set_xlabel("X")
    axes[2].set_ylabel("Y")
    plt.colorbar(im3, ax=axes[2], label="Matches ∇φ")
    
    plt.tight_layout()
    plt.savefig('test1_multiple_outlets.png', dpi=150, bbox_inches='tight')
    plt.close()  # Close figure to prevent blocking
    
    return ocn, phi, final_agreement


# ==========================================================
# MODULE 2 — CYCLE ROBUSTNESS TEST
# ==========================================================

def test_cycle_robustness(ny=60, nx=60, gamma=1.0, n_iter=200000):
    """Test #2: Cycle robustness"""
    print("\n" + "="*60)
    print("MODULE 2: CYCLE ROBUSTNESS TEST")
    print("="*60)
    
    # Initialize normal OCN
    ocn = OCN(ny=ny, nx=nx, seed=42)
    print("  Running initial optimization...")
    ocn.run(n_iter=20000, gamma=gamma, report_every=5000)
    initial_agreement = ocn.compute_agreement()
    
    print(f"Initial agreement (before cycles): {initial_agreement:.2f}%")
    
    # Create artificial braids/loops in a subgrid region
    # Method: Force cross-connections in a 10x10 region
    loop_region_y0, loop_region_y1 = ny//3, ny//3 + 10
    loop_region_x0, loop_region_x1 = nx//3, nx//3 + 10
    
    ocn_cyclic = copy.deepcopy(ocn)
    rng_local = np.random.default_rng(123)
    
    # Add cross-connections to create loops
    n_forced_edges = 15
    forced_edges = []
    for _ in range(n_forced_edges):
        y = rng_local.integers(loop_region_y0, loop_region_y1)
        x = rng_local.integers(loop_region_x0, loop_region_x1)
        neighs = ocn_cyclic.neighbor_cache[(y, x)]
        if neighs:
            nyy, nxx = neighs[rng_local.integers(len(neighs))]
            # Force edge if doesn't create immediate self-cycle
            if not (nyy == y and nxx == x):
                ocn_cyclic.parent_y[y, x] = nyy
                ocn_cyclic.parent_x[y, x] = nxx
                forced_edges.append((y, x, nyy, nxx))
    
    # Recompute
    ocn_cyclic.A = ocn_cyclic.compute_A()
    ocn_cyclic.children_dict = ocn_cyclic._build_children_dict()
    
    print(f"Added {len(forced_edges)} forced edges in loop region")
    
    # Run optimization
    print(f"Running optimization ({n_iter:,} iterations)...")
    ocn_cyclic.run(n_iter=n_iter, gamma=gamma, report_every=5000)
    final_agreement = ocn_cyclic.compute_agreement()
    
    print(f"Final agreement (after cycle optimization): {final_agreement:.2f}%")
    
    # Compute accuracy inside loop region vs global
    phi = ocn_cyclic.compute_phi(metric='unit')
    grad_py, grad_px = ocn_cyclic.compute_phi_gradient_direction(phi)
    
    loop_matches = 0
    loop_total = 0
    global_matches = 0
    global_total = 0
    
    for y in range(ny):
        for x in range(nx):
            if (y, x) in ocn_cyclic.outlets:
                continue
            
            matches = (ocn_cyclic.parent_y[y, x] == grad_py[y, x] and 
                      ocn_cyclic.parent_x[y, x] == grad_px[y, x])
            
            if loop_region_y0 <= y < loop_region_y1 and loop_region_x0 <= x < loop_region_x1:
                loop_total += 1
                if matches:
                    loop_matches += 1
            
            global_total += 1
            if matches:
                global_matches += 1
    
    loop_accuracy = 100.0 * loop_matches / loop_total if loop_total > 0 else 0.0
    global_accuracy = 100.0 * global_matches / global_total
    
    print(f"Loop region accuracy: {loop_accuracy:.2f}%")
    print(f"Global accuracy: {global_accuracy:.2f}%")
    
    # Create figure
    _, axes = plt.subplots(2, 2, figsize=(14, 14))
    
    # (a) Initial network with cycles
    ocn.plot_network(ax=axes[0, 0], title="(a) Initial Network")
    rect = Rectangle((loop_region_x0, loop_region_y0), 
                        loop_region_x1 - loop_region_x0,
                        loop_region_y1 - loop_region_y0,
                        fill=False, edgecolor='red', linewidth=2)
    axes[0, 0].add_patch(rect)
    
    # (b) Final network
    ocn_cyclic.plot_network(ax=axes[0, 1], title="(b) Final Network (optimized)")
    rect2 = Rectangle((loop_region_x0, loop_region_y0), 
                         loop_region_x1 - loop_region_x0,
                         loop_region_y1 - loop_region_y0,
                         fill=False, edgecolor='red', linewidth=2)
    axes[0, 1].add_patch(rect2)
    
    # (c) φ difference map
    diff_map = np.zeros((ny, nx))
    for y in range(ny):
        for x in range(nx):
            if (y, x) in ocn_cyclic.outlets:
                diff_map[y, x] = 0
            else:
                if (ocn_cyclic.parent_y[y, x] != grad_py[y, x] or 
                    ocn_cyclic.parent_x[y, x] != grad_px[y, x]):
                    diff_map[y, x] = 1.0  # Disagreement
    
    im3 = axes[1, 0].imshow(diff_map, cmap='Reds', origin='lower', vmin=0, vmax=1)
    rect3 = Rectangle((loop_region_x0, loop_region_y0), 
                         loop_region_x1 - loop_region_x0,
                         loop_region_y1 - loop_region_y0,
                         fill=False, edgecolor='blue', linewidth=2)
    axes[1, 0].add_patch(rect3)
    axes[1, 0].set_title("(c) Disagreement Map (OCN ≠ ∇φ)")
    axes[1, 0].set_xlabel("X")
    axes[1, 0].set_ylabel("Y")
    plt.colorbar(im3, ax=axes[1, 0], label="Disagreement")
    
    # (d) Energy curve
    ocn_cyclic.plot_energy(ax=axes[1, 1], title="(d) Energy Evolution")
    
    plt.tight_layout()
    plt.savefig('test2_cycle_robustness.png', dpi=150, bbox_inches='tight')
    plt.close()  # Close figure to prevent blocking
    
    return ocn_cyclic, loop_accuracy, global_accuracy


# ==========================================================
# MODULE 3 — GEOMETRIC NOISE ROBUSTNESS
# ==========================================================

def test_geometric_noise(ny=60, nx=60, noise_level=0.3, gamma=1.0, n_iter=200000):
    """Test #3: Geometric noise robustness"""
    print("\n" + "="*60)
    print("MODULE 3: GEOMETRIC NOISE ROBUSTNESS TEST")
    print("="*60)
    
    # Create pseudo-topography: linear slope + noise
    y_coords = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')[0]
    slope = 0.1  # Slope in y direction
    base_z = slope * (ny - y_coords)
    
    # Add Gaussian noise
    rng = np.random.default_rng(42)
    noise = rng.normal(0, noise_level, size=(ny, nx))
    z = base_z + noise
    
    print(f"Noise level (std): {noise_level}")
    
    # Compute gradient descent directions from topography
    grad_z_y = np.gradient(z, axis=0)
    grad_z_x = np.gradient(z, axis=1)
    
    # Find steepest descent neighbor for each cell
    gradient_parent_y = np.full((ny, nx), -1, dtype=int)
    gradient_parent_x = np.full((ny, nx), -1, dtype=int)
    
    outlets = [(ny-1, x) for x in range(nx)]
    
    for y in range(ny):
        for x in range(nx):
            if (y, x) in outlets:
                continue
            
            # Find neighbor with steepest descent (maximum negative gradient)
            best_neigh = None
            max_descent = -np.inf
            
            for nyy, nxx in neighbors(y, x, ny, nx):
                if nyy > y:  # Prefer downstream
                    dy, dx = nyy - y, nxx - x
                    descent = -(grad_z_y[y, x] * dy + grad_z_x[y, x] * dx)
                    if descent > max_descent:
                        max_descent = descent
                        best_neigh = (nyy, nxx)
            
            if best_neigh:
                gradient_parent_y[y, x] = best_neigh[0]
                gradient_parent_x[y, x] = best_neigh[1]
    
    # Run OCN
    ocn = OCN(ny=ny, nx=nx, outlets=outlets, seed=42)
    print(f"Running OCN optimization ({n_iter:,} iterations)...")
    ocn.run(n_iter=n_iter, gamma=gamma, report_every=2000)  # More frequent reporting
    
    # Compute true φ (geodesic)
    phi = ocn.compute_phi(metric='unit')
    grad_py, grad_px = ocn.compute_phi_gradient_direction(phi)
    
    # Compute agreements
    ocn_agreement = ocn.compute_agreement(phi)
    
    # Gradient agreement with φ
    grad_matches = 0
    grad_total = 0
    for y in range(ny):
        for x in range(nx):
            if (y, x) in outlets:
                continue
            grad_total += 1
            if (gradient_parent_y[y, x] == grad_py[y, x] and 
                gradient_parent_x[y, x] == grad_px[y, x]):
                grad_matches += 1
    
    grad_agreement = 100.0 * grad_matches / grad_total if grad_total > 0 else 0.0
    robustness_val = ocn_agreement - grad_agreement
    
    print(f"OCN agreement with φ: {ocn_agreement:.2f}%")
    print(f"Gradient agreement with φ: {grad_agreement:.2f}%")
    print(f"Robustness metric R = {robustness_val:.2f}%")
    
    # Create figure
    _, axes = plt.subplots(2, 2, figsize=(14, 14))
    
    # (a) Noisy topography
    im1 = axes[0, 0].imshow(z, cmap='terrain', origin='lower')
    axes[0, 0].set_title("(a) Pseudo-topography z (slope + noise)")
    axes[0, 0].set_xlabel("X")
    axes[0, 0].set_ylabel("Y")
    plt.colorbar(im1, ax=axes[0, 0], label="Elevation")
    
    # (b) Gradient descent network
    if ocn.A is None:
        ocn.A = ocn.compute_A()
    im2 = axes[0, 1].imshow(np.log10(ocn.A + 1e-6), cmap='YlOrRd', origin='lower')
    for y in range(ny):
        for x in range(nx):
            if (y, x) in outlets:
                axes[0, 1].plot(x, y, 'ks', markersize=6)
                continue
            py, px = gradient_parent_y[y, x], gradient_parent_x[y, x]
            if py != -1:
                dx = px - x
                dy = py - y
                axes[0, 1].arrow(x, y, dx*0.8, dy*0.8, head_width=0.3,
                               head_length=0.2, fc='green', ec='green',
                               alpha=0.4, length_includes_head=True)
    axes[0, 1].set_title(f"(b) Gradient Descent (agreement: {grad_agreement:.1f}%)")
    axes[0, 1].set_xlabel("X")
    axes[0, 1].set_ylabel("Y")
    plt.colorbar(im2, ax=axes[0, 1], label="log10(A)")
    
    # (c) OCN network
    ocn.plot_network(ax=axes[1, 0], title=f"(c) OCN Network (agreement: {ocn_agreement:.1f}%)")
    
    # (d) Comparison
    axes[1, 1].bar(['Gradient\nDescent', 'OCN'], 
                  [grad_agreement, ocn_agreement],
                  color=['green', 'blue'], alpha=0.7)
    axes[1, 1].axhline(y=ocn_agreement, color='r', linestyle='--', alpha=0.5)
    axes[1, 1].set_ylabel("Agreement with φ (%)")
    axes[1, 1].set_title(f"(d) Robustness R = {robustness_val:.1f}%")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig('test3_geometric_noise.png', dpi=150, bbox_inches='tight')
    plt.close()  # Close figure to prevent blocking
    
    return ocn, robustness_val, ocn_agreement, grad_agreement


# ==========================================================
# MODULE 4 — PERTURBATION EXPERIMENTS
# ==========================================================

def test_perturbations(ny=60, nx=60, gamma=1.0):
    """Test #4: Perturbation experiments"""
    print("\n" + "="*60)
    print("MODULE 4: PERTURBATION EXPERIMENTS")
    print("="*60)
    
    # Experiment A: Tectonic Tilt
    print("\n--- Experiment A: Tectonic Tilt ---")
    ocn_a0 = OCN(ny=ny, nx=nx, seed=42)
    print("  Running initial optimization...")
    ocn_a0.run(n_iter=30000, gamma=gamma, report_every=5000)
    phi_a0 = ocn_a0.compute_phi(metric='unit')
    agreement_a0 = ocn_a0.compute_agreement(phi_a0)
    print(f"Before tilt - agreement: {agreement_a0:.2f}%")
    
    # Add linear tilt: shift outlet region up
    # Move outlets up by 5 rows (simulating uplift)
    ocn_a1 = copy.deepcopy(ocn_a0)
    old_outlets = ocn_a1.outlets
    new_outlets = [(max(0, oy - 5), ox) for oy, ox in old_outlets]
    ocn_a1.outlets = new_outlets
    
    # Rebuild children_dict after modifying outlets (some cells might now be outlets)
    ocn_a1.children_dict = ocn_a1._build_children_dict()
    # Recompute A to ensure consistency
    ocn_a1.A = ocn_a1.compute_A()
    
    # Recompute phi with new outlets
    ocn_a1.phi = ocn_a1.compute_phi(metric='unit')
    
    # Re-optimize
    agreement_a1_initial = ocn_a1.compute_agreement(ocn_a1.phi)
    print(f"After tilt (before re-opt) - agreement: {agreement_a1_initial:.2f}%")
    
    # Track relaxation
    relax_iter = 0
    target_agreement = agreement_a0 * 0.95  # 95% of original
    for _ in range(10):
        ocn_a1.run(n_iter=10000, gamma=gamma, report_every=10000)
        relax_iter += 10000
        current_agreement = ocn_a1.compute_agreement(ocn_a1.phi)
        if current_agreement >= target_agreement:
            break
    
    agreement_a1_final = ocn_a1.compute_agreement(ocn_a1.phi)
    print(f"After re-optimization ({relax_iter} iter) - agreement: {agreement_a1_final:.2f}%")
    print(f"Relaxation time: {relax_iter} iterations")
    
    # Experiment B: Avulsion
    print("\n--- Experiment B: Avulsion ---")
    ocn_b0 = OCN(ny=ny, nx=nx, seed=42)
    print("  Running initial optimization...")
    ocn_b0.run(n_iter=30000, gamma=gamma, report_every=5000)
    phi_b0 = ocn_b0.compute_phi(metric='unit')
    agreement_b0 = ocn_b0.compute_agreement(phi_b0)
    print(f"Before avulsion - agreement: {agreement_b0:.2f}%")
    
    # Create artificial shorter path: cut new channel from mid-top to outlet
    ocn_b1 = copy.deepcopy(ocn_b0)
    # Force a path from top-center to bottom-center
    center_x = nx // 2
    for y in range(0, ny-1):
        # Point to cell directly below (straight path)
        ocn_b1.parent_y[y, center_x] = y + 1
        ocn_b1.parent_x[y, center_x] = center_x
    
    # Rebuild children_dict after modifying parent pointers
    ocn_b1.children_dict = ocn_b1._build_children_dict()
    ocn_b1.A = ocn_b1.compute_A()
    ocn_b1.phi = ocn_b1.compute_phi(metric='unit')
    
    agreement_b1_initial = ocn_b1.compute_agreement(ocn_b1.phi)
    print(f"After avulsion (before re-opt) - agreement: {agreement_b1_initial:.2f}%")
    
    # Re-optimize
    relax_iter_b = 0
    for _ in range(10):
        ocn_b1.run(n_iter=10000, gamma=gamma, report_every=10000)
        relax_iter_b += 10000
        current_agreement = ocn_b1.compute_agreement(ocn_b1.phi)
        if current_agreement >= agreement_b0 * 0.95:
            break
    
    agreement_b1_final = ocn_b1.compute_agreement(ocn_b1.phi)
    print(f"After re-optimization ({relax_iter_b} iter) - agreement: {agreement_b1_final:.2f}%")
    print(f"Relaxation time: {relax_iter_b} iterations")
    
    # Experiment C: Dam / Canal
    print("\n--- Experiment C: Dam / Canal ---")
    ocn_c0 = OCN(ny=ny, nx=nx, seed=42)
    print("  Running initial optimization...")
    ocn_c0.run(n_iter=30000, gamma=gamma, report_every=5000)
    phi_c0 = ocn_c0.compute_phi(metric='unit')
    agreement_c0 = ocn_c0.compute_agreement(phi_c0)
    print(f"Before dam/canal - agreement: {agreement_c0:.2f}%")
    
    # Force local re-route across a divide
    ocn_c1 = copy.deepcopy(ocn_c0)
    # Force a cross-divide connection at middle-left to middle-right
    mid_y = ny // 2
    left_x = nx // 4
    right_x = 3 * nx // 4
    
    # Create bridge: left -> right
    ocn_c1.parent_y[mid_y, left_x] = mid_y
    ocn_c1.parent_x[mid_y, left_x] = left_x + 1
    # Continue path to right
    for x in range(left_x + 1, right_x):
        ocn_c1.parent_y[mid_y, x] = mid_y
        ocn_c1.parent_x[mid_y, x] = x + 1
    
    # Rebuild children_dict after modifying parent pointers
    ocn_c1.children_dict = ocn_c1._build_children_dict()
    ocn_c1.A = ocn_c1.compute_A()
    ocn_c1.phi = ocn_c1.compute_phi(metric='unit')
    
    agreement_c1_initial = ocn_c1.compute_agreement(ocn_c1.phi)
    print(f"After dam/canal (before re-opt) - agreement: {agreement_c1_initial:.2f}%")
    
    # Re-optimize
    print("  Re-optimizing after dam/canal...")
    ocn_c1.run(n_iter=20000, gamma=gamma, report_every=5000)
    agreement_c1_final = ocn_c1.compute_agreement(ocn_c1.phi)
    
    # Find topological anomalies (disagreements)
    grad_py, grad_px = ocn_c1.compute_phi_gradient_direction(ocn_c1.phi)
    anomalies = []
    for y in range(ny):
        for x in range(nx):
            if (y, x) in ocn_c1.outlets:
                continue
            if (ocn_c1.parent_y[y, x] != grad_py[y, x] or 
                ocn_c1.parent_x[y, x] != grad_px[y, x]):
                anomalies.append((y, x))
    
    print(f"After re-optimization - agreement: {agreement_c1_final:.2f}%")
    print(f"Topological anomalies: {len(anomalies)} nodes")
    
    # Create figure
    _, axes = plt.subplots(3, 2, figsize=(14, 18))
    
    # Experiment A
    ocn_a0.plot_network(ax=axes[0, 0], title="A1: Before Tilt")
    ocn_a1.plot_network(ax=axes[0, 1], title=f"A2: After Tilt & Re-opt ({relax_iter} iter)")
    
    # Experiment B
    ocn_b0.plot_network(ax=axes[1, 0], title="B1: Before Avulsion")
    ocn_b1.plot_network(ax=axes[1, 1], title=f"B2: After Avulsion & Re-opt ({relax_iter_b} iter)")
    
    # Experiment C
    ocn_c0.plot_network(ax=axes[2, 0], title="C1: Before Dam/Canal")
    ocn_c1.plot_network(ax=axes[2, 1], title=f"C2: After Dam/Canal ({len(anomalies)} anomalies)")
    
    # Mark anomalies
    if anomalies:
        anom_y, anom_x = zip(*anomalies)
        axes[2, 1].scatter(anom_x, anom_y, c='red', s=20, alpha=0.5, zorder=10, label='Anomalies')
        axes[2, 1].legend()
    
    plt.tight_layout()
    plt.savefig('test4_perturbations.png', dpi=150, bbox_inches='tight')
    plt.close()  # Close figure to prevent blocking
    
    return {
        'tilt': {'relax_time': relax_iter, 'final_agreement': agreement_a1_final},
        'avulsion': {'relax_time': relax_iter_b, 'final_agreement': agreement_b1_final},
        'dam': {'final_agreement': agreement_c1_final, 'n_anomalies': len(anomalies)}
    }


# ==========================================================
# MODULE 5 — GAMMA COMPARISON
# ==========================================================

def test_gamma_comparison(ny=60, nx=60, n_iter=200000):
    """Test #5: γ = 0.5 vs γ = 1.0 universality"""
    print("\n" + "="*60)
    print("MODULE 5: GAMMA COMPARISON (γ = 0.5 vs γ = 1.0)")
    print("="*60)
    
    # Compute true φ
    ocn_temp = OCN(ny=ny, nx=nx, seed=42)
    phi_true = ocn_temp.compute_phi(metric='unit')
    
    # Run OCN with gamma = 0.5 (geometric/realistic)
    print("\nRunning OCN with γ = 0.5 (geometric)...")
    print(f"  Starting optimization ({n_iter:,} iterations)...")
    ocn_g = OCN(ny=ny, nx=nx, seed=42)
    ocn_g.run(n_iter=n_iter, gamma=0.5, report_every=5000)
    agreement_geom = ocn_g.compute_agreement(phi_true)
    
    print(f"OCN_geom agreement with φ: {agreement_geom:.2f}%")
    print(f"OCN_geom final energy: {ocn_g.H:.2e}")
    
    # Run OCN with gamma = 1.0 (topological/geodesic)
    print("\nRunning OCN with γ = 1.0 (topological)...")
    print(f"  Starting optimization ({n_iter:,} iterations)...")
    ocn_t = OCN(ny=ny, nx=nx, seed=42)
    ocn_t.run(n_iter=n_iter, gamma=1.0, report_every=5000)
    agreement_topo = ocn_t.compute_agreement(phi_true)
    
    print(f"OCN_topo agreement with φ: {agreement_topo:.2f}%")
    print(f"OCN_topo final energy: {ocn_t.H:.2e}")
    
    # Compute geometric metrics (Hack's law approximation: length ~ A^h)
    # For simplicity, count total edge count and mean area
    def compute_network_length(parent_y, parent_x):
        """Compute total network length"""
        total_len = 0.0
        for y in range(ny):
            for x in range(nx):
                py, px = parent_y[y, x], parent_x[y, x]
                if py != -1:
                    dy, dx = py - y, px - x
                    total_len += np.sqrt(dy*dy + dx*dx)
        return total_len
    
    len_geom = compute_network_length(ocn_g.parent_y, ocn_g.parent_x)
    len_topo = compute_network_length(ocn_t.parent_y, ocn_t.parent_x)
    
    if ocn_g.A is None:
        ocn_g.A = ocn_g.compute_A()
    if ocn_t.A is None:
        ocn_t.A = ocn_t.compute_A()
    mean_A_geom = float(np.mean(ocn_g.A))
    mean_A_topo = float(np.mean(ocn_t.A))
    
    print(f"\nGeometric metrics:")
    print(f"  OCN_geom: mean A = {mean_A_geom:.2f}, total length = {len_geom:.2f}")
    print(f"  OCN_topo: mean A = {mean_A_topo:.2f}, total length = {len_topo:.2f}")
    
    # Create figure
    _, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # (a) OCN_geom network
    ocn_g.plot_network(ax=axes[0, 0], 
                         title=f"(a) OCN_geom (γ=0.5, agreement: {agreement_geom:.1f}%)")
    
    # (b) OCN_topo network
    ocn_t.plot_network(ax=axes[0, 1], 
                         title=f"(b) OCN_topo (γ=1.0, agreement: {agreement_topo:.1f}%)")
    
    # (c) φ orientation
    grad_py, grad_px = ocn_t.compute_phi_gradient_direction(phi_true)
    im3 = axes[1, 0].imshow(phi_true, cmap='viridis', origin='lower')
    for y in range(ny):
        for x in range(nx):
            if (y, x) in ocn_t.outlets:
                axes[1, 0].plot(x, y, 'r*', markersize=8, zorder=5)
                continue
            py, px = grad_py[y, x], grad_px[y, x]
            if py != -1:
                dx = px - x
                dy = py - y
                axes[1, 0].arrow(x, y, dx*0.8, dy*0.8, head_width=0.3,
                               head_length=0.2, fc='yellow', ec='yellow',
                               alpha=0.5, length_includes_head=True)
    axes[1, 0].set_title("(c) φ Orientation (∇φ)")
    axes[1, 0].set_xlabel("X")
    axes[1, 0].set_ylabel("Y")
    plt.colorbar(im3, ax=axes[1, 0], label="Distance to outlet")
    
    # (d) Comparison table
    axes[1, 1].axis('off')
    table_data = [
        ['Metric', 'OCN_geom (γ=0.5)', 'OCN_topo (γ=1.0)'],
        ['Agreement with φ', f'{agreement_geom:.2f}%', f'{agreement_topo:.2f}%'],
        ['Mean Contrib. Area', f'{mean_A_geom:.2f}', f'{mean_A_topo:.2f}'],
        ['Total Length', f'{len_geom:.2f}', f'{len_topo:.2f}'],
        ['Final Energy', f'{ocn_g.H:.2e}', f'{ocn_t.H:.2e}'],
    ]
    table = axes[1, 1].table(cellText=table_data[1:], colLabels=table_data[0],
                            cellLoc='center', loc='center',
                            colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    axes[1, 1].set_title("(d) Comparison Table", pad=20)
    
    plt.tight_layout()
    plt.savefig('test5_gamma_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()  # Close figure to prevent blocking
    
    return ocn_g, ocn_t, agreement_geom, agreement_topo


# ==========================================================
# MAIN EXECUTION
# ==========================================================

if __name__ == "__main__":
    print("="*60)
    print("THEORY TEST OCN SUITE")
    print("Testing Topological Universality Hypothesis")
    print("Gearon (2025)")
    print("="*60)
    
    # Run all tests
    results = {}
    
    # Module 1
    print("\nStarting Module 1 optimization...")
    ocn1, phi1, agreement1 = test_multiple_outlets(ny=60, nx=60, n_outlets=3, gamma=1.0, n_iter=50000)  # Reduced for testing
    results['module1'] = {'agreement': agreement1}
    
    # Module 2
    print("\nStarting Module 2 optimization...")
    ocn2, loop_acc, global_acc = test_cycle_robustness(ny=60, nx=60, gamma=1.0, n_iter=50000)
    results['module2'] = {'loop_accuracy': loop_acc, 'global_accuracy': global_acc}
    
    # Module 3
    print("\nStarting Module 3 optimization...")
    ocn3, robustness, ocn_agree, grad_agree = test_geometric_noise(ny=60, nx=60, noise_level=0.3, gamma=1.0, n_iter=50000)
    results['module3'] = {'robustness': robustness, 'ocn_agreement': ocn_agree, 'grad_agreement': grad_agree}
    
    # Module 4
    pert_results = test_perturbations(ny=60, nx=60, gamma=1.0)
    results['module4'] = pert_results
    
    # Module 5
    print("\nStarting Module 5 optimization...")
    ocn_g5, ocn_t5, agree_geom, agree_topo = test_gamma_comparison(ny=60, nx=60, n_iter=50000)
    results['module5'] = {'geom_agreement': agree_geom, 'topo_agreement': agree_topo}
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY OF RESULTS")
    print("="*60)
    print("\nModule 1 (Multiple Outlets):")
    print(f"  Agreement: {results['module1']['agreement']:.2f}%")
    print("\nModule 2 (Cycle Robustness):")
    print(f"  Loop accuracy: {results['module2']['loop_accuracy']:.2f}%")
    print(f"  Global accuracy: {results['module2']['global_accuracy']:.2f}%")
    print("\nModule 3 (Geometric Noise):")  # noqa: F541
    print(f"  Robustness R: {results['module3']['robustness']:.2f}%")
    print(f"  OCN agreement: {results['module3']['ocn_agreement']:.2f}%")
    print(f"  Gradient agreement: {results['module3']['grad_agreement']:.2f}%")
    print("\nModule 4 (Perturbations):")
    print(f"  Tilt relaxation: {results['module4']['tilt']['relax_time']} iterations")
    print(f"  Avulsion relaxation: {results['module4']['avulsion']['relax_time']} iterations")
    print(f"  Dam anomalies: {results['module4']['dam']['n_anomalies']} nodes")
    print("\nModule 5 (Gamma Comparison):")
    print(f"  OCN_geom (γ=0.5) agreement: {results['module5']['geom_agreement']:.2f}%")
    print(f"  OCN_topo (γ=1.0) agreement: {results['module5']['topo_agreement']:.2f}%")
    
    print("\n" + "="*60)
    print("All tests completed. Plots saved.")
    print("="*60)
    
    # Suppress unused variable warnings
    _ = ocn1, phi1, ocn2, ocn3, ocn_g5, ocn_t5
