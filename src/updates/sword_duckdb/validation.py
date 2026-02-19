"""
SWORD Topology Validation Module

Provides automated checks for topology quality and physical plausibility.
Run these checks periodically to identify issues for review.
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of a validation check."""
    check_name: str
    passed: bool
    total_checked: int
    issues_found: int
    issue_pct: float
    details: pd.DataFrame
    description: str


class SWORDValidator:
    """Validation checks for SWORD topology and physical plausibility."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
    
    def connect(self):
        if self.conn is None:
            self.conn = duckdb.connect(self.db_path, read_only=True)
    
    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, *args):
        self.close()
    
    # =========================================================================
    # CHECK 1: WSE Gradient (flow direction validation)
    # =========================================================================
    def check_wse_gradient(self, region: Optional[str] = None) -> ValidationResult:
        """
        Check if WSE decreases downstream along each path.
        
        Flags reaches where downstream neighbor has higher WSE,
        which suggests flow direction error.
        """
        self.connect()
        
        where_clause = f"AND r1.region = '{region}'" if region else ""
        
        query = f"""
        WITH reach_pairs AS (
            SELECT 
                r1.reach_id,
                r1.region,
                r1.wse as wse_up,
                r2.wse as wse_down,
                r1.river_name,
                r1.x, r1.y,
                r1.lakeflag
            FROM reaches r1
            JOIN reach_topology rt ON r1.reach_id = rt.reach_id AND r1.region = rt.region
            JOIN reaches r2 ON rt.neighbor_reach_id = r2.reach_id AND rt.region = r2.region
            WHERE rt.direction = 'down'
                AND r1.wse > 0 AND r1.wse != -9999
                AND r2.wse > 0 AND r2.wse != -9999
                AND r1.lakeflag = 0  -- rivers only
                AND r2.lakeflag = 0
                {where_clause}
        )
        SELECT 
            reach_id, region, river_name, x, y,
            wse_up, wse_down,
            (wse_down - wse_up) as wse_increase
        FROM reach_pairs
        WHERE wse_down > wse_up + 0.5  -- >0.5m increase downstream
        ORDER BY wse_increase DESC
        """
        
        issues = self.conn.execute(query).fetchdf()
        
        # Get total count
        total_query = f"""
        SELECT COUNT(*) FROM reaches 
        WHERE wse > 0 AND wse != -9999 AND lakeflag = 0
        {where_clause.replace('r1.', '')}
        """
        total = self.conn.execute(total_query).fetchone()[0]
        
        return ValidationResult(
            check_name="WSE Gradient",
            passed=len(issues) == 0,
            total_checked=total,
            issues_found=len(issues),
            issue_pct=100 * len(issues) / total if total > 0 else 0,
            details=issues,
            description="Reaches where WSE increases downstream (potential flow direction error)"
        )
    
    # =========================================================================
    # CHECK 2: Width Monotonicity
    # =========================================================================
    def check_width_monotonicity(self, region: Optional[str] = None, 
                                  threshold: float = 0.3) -> ValidationResult:
        """
        Check if width generally increases downstream.
        
        Flags reaches where downstream width is less than threshold * upstream width.
        Some decrease is normal, but dramatic decreases suggest issues.
        """
        self.connect()
        
        where_clause = f"AND r1.region = '{region}'" if region else ""
        
        query = f"""
        WITH reach_pairs AS (
            SELECT 
                r1.reach_id,
                r1.region,
                r1.width as width_up,
                r2.width as width_down,
                r1.river_name,
                r1.x, r1.y,
                r1.lakeflag as lakeflag_up,
                r2.lakeflag as lakeflag_down
            FROM reaches r1
            JOIN reach_topology rt ON r1.reach_id = rt.reach_id AND r1.region = rt.region
            JOIN reaches r2 ON rt.neighbor_reach_id = r2.reach_id AND rt.region = r2.region
            WHERE rt.direction = 'down'
                AND r1.width > 0 AND r1.width != -9999
                AND r2.width > 0 AND r2.width != -9999
                AND r1.lakeflag = 0 AND r2.lakeflag = 0  -- rivers only
                {where_clause}
        )
        SELECT 
            reach_id, region, river_name, x, y,
            width_up, width_down,
            ROUND(width_down / width_up, 3) as width_ratio
        FROM reach_pairs
        WHERE width_down < {threshold} * width_up
            AND width_up > 100  -- ignore small streams
        ORDER BY width_ratio ASC
        """
        
        issues = self.conn.execute(query).fetchdf()
        
        total_query = f"""
        SELECT COUNT(*) FROM reaches 
        WHERE width > 100 AND width != -9999 AND lakeflag = 0
        {where_clause.replace('r1.', '')}
        """
        total = self.conn.execute(total_query).fetchone()[0]
        
        return ValidationResult(
            check_name="Width Monotonicity",
            passed=len(issues) == 0,
            total_checked=total,
            issues_found=len(issues),
            issue_pct=100 * len(issues) / total if total > 0 else 0,
            details=issues,
            description=f"Reaches where downstream width < {threshold*100:.0f}% of upstream width"
        )
    
    # =========================================================================
    # CHECK 3: dist_out Monotonicity
    # =========================================================================
    def check_dist_out_monotonicity(self, region: Optional[str] = None) -> ValidationResult:
        """
        Check if dist_out decreases downstream (closer to outlet).
        
        Flags reaches where downstream neighbor has higher dist_out.
        """
        self.connect()
        
        where_clause = f"AND r1.region = '{region}'" if region else ""
        
        query = f"""
        WITH reach_pairs AS (
            SELECT 
                r1.reach_id,
                r1.region,
                r1.dist_out as dist_out_up,
                r2.dist_out as dist_out_down,
                r1.river_name,
                r1.x, r1.y
            FROM reaches r1
            JOIN reach_topology rt ON r1.reach_id = rt.reach_id AND r1.region = rt.region
            JOIN reaches r2 ON rt.neighbor_reach_id = r2.reach_id AND rt.region = r2.region
            WHERE rt.direction = 'down'
                AND r1.dist_out > 0 AND r1.dist_out != -9999
                AND r2.dist_out > 0 AND r2.dist_out != -9999
                {where_clause}
        )
        SELECT 
            reach_id, region, river_name, x, y,
            dist_out_up, dist_out_down,
            (dist_out_down - dist_out_up) as dist_out_increase
        FROM reach_pairs
        WHERE dist_out_down > dist_out_up + 100  -- >100m increase
        ORDER BY dist_out_increase DESC
        """
        
        issues = self.conn.execute(query).fetchdf()
        
        total_query = f"""
        SELECT COUNT(*) FROM reaches 
        WHERE dist_out > 0 AND dist_out != -9999
        {where_clause.replace('r1.', '')}
        """
        total = self.conn.execute(total_query).fetchone()[0]
        
        return ValidationResult(
            check_name="dist_out Monotonicity",
            passed=len(issues) == 0,
            total_checked=total,
            issues_found=len(issues),
            issue_pct=100 * len(issues) / total if total > 0 else 0,
            details=issues,
            description="Reaches where dist_out increases downstream (topology error)"
        )
    
    # =========================================================================
    # CHECK 4: path_freq Consistency
    # =========================================================================
    def check_path_freq_consistency(self, region: Optional[str] = None) -> ValidationResult:
        """
        Check if path_freq increases toward outlets.
        
        At confluences, downstream path_freq should >= max(upstream path_freqs).
        """
        self.connect()
        
        where_clause = f"AND r1.region = '{region}'" if region else ""
        
        query = f"""
        WITH downstream_freqs AS (
            SELECT 
                r1.reach_id,
                r1.region,
                r1.path_freq as pf_up,
                r2.path_freq as pf_down,
                r1.river_name,
                r1.x, r1.y
            FROM reaches r1
            JOIN reach_topology rt ON r1.reach_id = rt.reach_id AND r1.region = rt.region
            JOIN reaches r2 ON rt.neighbor_reach_id = r2.reach_id AND rt.region = r2.region
            WHERE rt.direction = 'down'
                AND r1.path_freq > 0 AND r1.path_freq != -9999
                AND r2.path_freq > 0 AND r2.path_freq != -9999
                {where_clause}
        )
        SELECT 
            reach_id, region, river_name, x, y,
            pf_up, pf_down,
            (pf_up - pf_down) as pf_decrease
        FROM downstream_freqs
        WHERE pf_down < pf_up  -- downstream should be >= upstream
        ORDER BY pf_decrease DESC
        """
        
        issues = self.conn.execute(query).fetchdf()
        
        total_query = f"""
        SELECT COUNT(*) FROM reaches 
        WHERE path_freq > 0 AND path_freq != -9999
        {where_clause.replace('r1.', '')}
        """
        total = self.conn.execute(total_query).fetchone()[0]
        
        return ValidationResult(
            check_name="path_freq Consistency",
            passed=len(issues) == 0,
            total_checked=total,
            issues_found=len(issues),
            issue_pct=100 * len(issues) / total if total > 0 else 0,
            details=issues,
            description="Reaches where path_freq decreases downstream"
        )
    
    # =========================================================================
    # CHECK 5: Orphaned Reaches
    # =========================================================================
    def check_orphaned_reaches(self, region: Optional[str] = None) -> ValidationResult:
        """
        Find reaches with no upstream AND no downstream neighbors.
        
        These are disconnected from the network (unless single-reach networks).
        """
        self.connect()
        
        where_clause = f"AND r.region = '{region}'" if region else ""
        
        query = f"""
        SELECT 
            r.reach_id, r.region, r.river_name, r.x, r.y,
            r.n_rch_up, r.n_rch_down, r.network, r.reach_length, r.width
        FROM reaches r
        WHERE r.n_rch_up = 0 AND r.n_rch_down = 0
            {where_clause}
        ORDER BY r.reach_length DESC
        """
        
        issues = self.conn.execute(query).fetchdf()
        
        total_query = f"""
        SELECT COUNT(*) FROM reaches r WHERE 1=1 {where_clause}
        """
        total = self.conn.execute(total_query).fetchone()[0]
        
        return ValidationResult(
            check_name="Orphaned Reaches",
            passed=len(issues) == 0,
            total_checked=total,
            issues_found=len(issues),
            issue_pct=100 * len(issues) / total if total > 0 else 0,
            details=issues,
            description="Reaches with no upstream or downstream neighbors (disconnected)"
        )
    
    # =========================================================================
    # CHECK 6: Lake Sandwich Detection
    # =========================================================================
    def check_lake_sandwiches(self, region: Optional[str] = None) -> ValidationResult:
        """
        Find river reaches sandwiched between lake reaches.
        
        These may be misclassified lake sections.
        """
        self.connect()
        
        where_clause = f"AND r.region = '{region}'" if region else ""
        
        query = f"""
        WITH river_reaches AS (
            SELECT reach_id, region, x, y, reach_length, width, river_name
            FROM reaches WHERE lakeflag = 0 {where_clause.replace('r.', '')}
        ),
        has_lake_upstream AS (
            SELECT DISTINCT rt.reach_id, rt.region
            FROM reach_topology rt
            JOIN reaches r ON rt.neighbor_reach_id = r.reach_id AND rt.region = r.region
            WHERE rt.direction = 'up' AND r.lakeflag = 1
        ),
        has_lake_downstream AS (
            SELECT DISTINCT rt.reach_id, rt.region
            FROM reach_topology rt
            JOIN reaches r ON rt.neighbor_reach_id = r.reach_id AND rt.region = r.region
            WHERE rt.direction = 'down' AND r.lakeflag = 1
        )
        SELECT rr.reach_id, rr.region, rr.river_name, rr.x, rr.y,
               rr.reach_length, rr.width
        FROM river_reaches rr
        JOIN has_lake_upstream hu ON rr.reach_id = hu.reach_id AND rr.region = hu.region
        JOIN has_lake_downstream hd ON rr.reach_id = hd.reach_id AND rr.region = hd.region
        ORDER BY rr.reach_length DESC
        """
        
        issues = self.conn.execute(query).fetchdf()
        
        total_query = f"""
        SELECT COUNT(*) FROM reaches r WHERE lakeflag = 0 {where_clause}
        """
        total = self.conn.execute(total_query).fetchone()[0]
        
        return ValidationResult(
            check_name="Lake Sandwiches",
            passed=len(issues) == 0,
            total_checked=total,
            issues_found=len(issues),
            issue_pct=100 * len(issues) / total if total > 0 else 0,
            details=issues,
            description="River reaches sandwiched between lake reaches (potential misclassification)"
        )
    
    # =========================================================================
    # RUN ALL CHECKS
    # =========================================================================
    def run_all_checks(self, region: Optional[str] = None) -> Dict[str, ValidationResult]:
        """Run all validation checks and return results."""
        
        checks = {
            'wse_gradient': self.check_wse_gradient,
            'width_monotonicity': self.check_width_monotonicity,
            'dist_out_monotonicity': self.check_dist_out_monotonicity,
            'path_freq_consistency': self.check_path_freq_consistency,
            'orphaned_reaches': self.check_orphaned_reaches,
            'lake_sandwiches': self.check_lake_sandwiches,
        }
        
        results = {}
        for name, check_fn in checks.items():
            try:
                results[name] = check_fn(region)
            except Exception as e:
                print(f"Error running {name}: {e}")
        
        return results
    
    def print_report(self, results: Dict[str, ValidationResult]):
        """Print a summary report of validation results."""
        
        print("=" * 70)
        print("SWORD TOPOLOGY VALIDATION REPORT")
        print("=" * 70)
        
        total_issues = 0
        for name, result in results.items():
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"\n{status} {result.check_name}")
            print(f"   {result.description}")
            print(f"   Checked: {result.total_checked:,} | Issues: {result.issues_found:,} ({result.issue_pct:.2f}%)")
            total_issues += result.issues_found
        
        print("\n" + "=" * 70)
        print(f"TOTAL ISSUES FOUND: {total_issues:,}")
        print("=" * 70)


# =============================================================================
# CLI ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run SWORD topology validation checks')
    parser.add_argument('db_path', help='Path to SWORD DuckDB database')
    parser.add_argument('--region', '-r', help='Region to check (default: all)')
    parser.add_argument('--check', '-c', help='Specific check to run (default: all)')
    parser.add_argument('--output', '-o', help='Output CSV for issues')
    args = parser.parse_args()
    
    with SWORDValidator(args.db_path) as validator:
        if args.check:
            check_fn = getattr(validator, f'check_{args.check}', None)
            if check_fn:
                result = check_fn(args.region)
                validator.print_report({args.check: result})
                if args.output and len(result.details) > 0:
                    result.details.to_csv(args.output, index=False)
                    print(f"\nIssues saved to: {args.output}")
            else:
                print(f"Unknown check: {args.check}")
        else:
            results = validator.run_all_checks(args.region)
            validator.print_report(results)
            
            if args.output:
                # Combine all issues
                all_issues = []
                for name, result in results.items():
                    if len(result.details) > 0:
                        result.details['check'] = name
                        all_issues.append(result.details)
                if all_issues:
                    combined = pd.concat(all_issues, ignore_index=True)
                    combined.to_csv(args.output, index=False)
                    print(f"\nAll issues saved to: {args.output}")
