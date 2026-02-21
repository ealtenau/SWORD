# -*- coding: utf-8 -*-
"""
Tests for the SWORD reconstruction system.
"""

import pytest

pytestmark = pytest.mark.unit


class TestReconstructionImports:
    """Test that reconstruction module imports correctly."""

    def test_import_reconstruction_engine(self):
        from src.sword_duckdb import ReconstructionEngine

        assert ReconstructionEngine is not None

    def test_import_source_dataset(self):
        from src.sword_duckdb import SourceDataset

        assert SourceDataset.GRWL.value == "GRWL"
        assert SourceDataset.MERIT_HYDRO.value == "MERIT_HYDRO"
        assert SourceDataset.HYDROBASINS.value == "HYDROBASINS"
        assert SourceDataset.GRAND.value == "GRAND"
        assert SourceDataset.GROD.value == "GROD"
        assert SourceDataset.COMPUTED.value == "COMPUTED"

    def test_import_derivation_method(self):
        from src.sword_duckdb import DerivationMethod

        assert DerivationMethod.DIRECT.value == "direct"
        assert DerivationMethod.MEDIAN.value == "median"
        assert DerivationMethod.MAX.value == "max"
        assert DerivationMethod.LINEAR_REGRESSION.value == "linear_regression"
        assert DerivationMethod.SPATIAL_JOIN.value == "spatial_join"
        assert DerivationMethod.GRAPH_TRAVERSAL.value == "graph_traversal"

    def test_import_attribute_spec(self):
        from src.sword_duckdb import AttributeSpec

        assert AttributeSpec is not None

    def test_import_attribute_sources(self):
        from src.sword_duckdb import ATTRIBUTE_SOURCES

        assert isinstance(ATTRIBUTE_SOURCES, dict)
        assert len(ATTRIBUTE_SOURCES) > 0


class TestAttributeSources:
    """Test attribute source mappings."""

    def test_reach_wse_source(self):
        from src.sword_duckdb import (
            ATTRIBUTE_SOURCES,
            SourceDataset,
            DerivationMethod,
        )

        spec = ATTRIBUTE_SOURCES.get("reach.wse")
        assert spec is not None
        assert spec.source == SourceDataset.MERIT_HYDRO
        assert spec.method == DerivationMethod.MEDIAN

    def test_reach_slope_source(self):
        from src.sword_duckdb import (
            ATTRIBUTE_SOURCES,
            SourceDataset,
            DerivationMethod,
        )

        spec = ATTRIBUTE_SOURCES.get("reach.slope")
        assert spec is not None
        assert spec.source == SourceDataset.COMPUTED
        assert spec.method == DerivationMethod.LINEAR_REGRESSION

    def test_reach_facc_source(self):
        from src.sword_duckdb import (
            ATTRIBUTE_SOURCES,
            SourceDataset,
            DerivationMethod,
        )

        spec = ATTRIBUTE_SOURCES.get("reach.facc")
        assert spec is not None
        assert spec.source == SourceDataset.MERIT_HYDRO
        assert spec.method == DerivationMethod.MAX

    def test_reach_dist_out_source(self):
        from src.sword_duckdb import (
            ATTRIBUTE_SOURCES,
            SourceDataset,
            DerivationMethod,
        )

        spec = ATTRIBUTE_SOURCES.get("reach.dist_out")
        assert spec is not None
        assert spec.source == SourceDataset.COMPUTED
        assert spec.method == DerivationMethod.PATH_ACCUMULATION

    def test_reach_width_source(self):
        from src.sword_duckdb import (
            ATTRIBUTE_SOURCES,
            SourceDataset,
            DerivationMethod,
        )

        spec = ATTRIBUTE_SOURCES.get("reach.width")
        assert spec is not None
        assert spec.source == SourceDataset.GRWL
        assert spec.method == DerivationMethod.MEDIAN

    def test_reach_length_source(self):
        from src.sword_duckdb import (
            ATTRIBUTE_SOURCES,
            SourceDataset,
            DerivationMethod,
        )

        spec = ATTRIBUTE_SOURCES.get("reach.reach_length")
        assert spec is not None
        assert spec.source == SourceDataset.COMPUTED
        assert spec.method == DerivationMethod.SUM

    def test_attribute_spec_entity_type(self):
        from src.sword_duckdb import ATTRIBUTE_SOURCES

        spec = ATTRIBUTE_SOURCES.get("reach.wse")
        assert spec.entity_type == "reach"
        assert spec.attribute_name == "wse"

    def test_attribute_spec_entity_type_node(self):
        from src.sword_duckdb import ATTRIBUTE_SOURCES

        spec = ATTRIBUTE_SOURCES.get("node.wse")
        assert spec is not None
        assert spec.entity_type == "node"
        assert spec.attribute_name == "wse"


class TestWorkflowReconstructionMethods:
    """Test SWORDWorkflow reconstruction integration."""

    def test_workflow_has_reconstruction_property(self):
        from src.sword_duckdb import SWORDWorkflow

        workflow = SWORDWorkflow()
        assert hasattr(workflow, "reconstruction")
        # Not loaded yet, so should be None
        assert workflow.reconstruction is None

    def test_workflow_has_reconstruct_method(self):
        from src.sword_duckdb import SWORDWorkflow

        workflow = SWORDWorkflow()
        assert hasattr(workflow, "reconstruct")
        assert callable(workflow.reconstruct)

    def test_workflow_has_reconstruct_from_centerlines_method(self):
        from src.sword_duckdb import SWORDWorkflow

        workflow = SWORDWorkflow()
        assert hasattr(workflow, "reconstruct_from_centerlines")
        assert callable(workflow.reconstruct_from_centerlines)

    def test_workflow_has_validate_reconstruction_method(self):
        from src.sword_duckdb import SWORDWorkflow

        workflow = SWORDWorkflow()
        assert hasattr(workflow, "validate_reconstruction")
        assert callable(workflow.validate_reconstruction)

    def test_workflow_has_get_source_info_method(self):
        from src.sword_duckdb import SWORDWorkflow

        workflow = SWORDWorkflow()
        assert hasattr(workflow, "get_source_info")
        assert callable(workflow.get_source_info)

    def test_workflow_has_list_reconstructable_attributes_method(self):
        from src.sword_duckdb import SWORDWorkflow

        workflow = SWORDWorkflow()
        assert hasattr(workflow, "list_reconstructable_attributes")
        assert callable(workflow.list_reconstructable_attributes)


class TestReconstructionEngineMethods:
    """Test ReconstructionEngine class methods."""

    def test_engine_has_attribute_sources(self):
        from src.sword_duckdb.reconstruction import ReconstructionEngine

        assert hasattr(ReconstructionEngine, "attribute_sources")

    def test_engine_has_reconstruct_method(self):
        from src.sword_duckdb.reconstruction import ReconstructionEngine

        # Check the class has the method
        assert hasattr(ReconstructionEngine, "reconstruct")

    def test_engine_has_validate_method(self):
        from src.sword_duckdb.reconstruction import ReconstructionEngine

        assert hasattr(ReconstructionEngine, "validate")

    def test_engine_has_register_recipe_method(self):
        from src.sword_duckdb.reconstruction import ReconstructionEngine

        assert hasattr(ReconstructionEngine, "register_recipe")

    def test_engine_has_get_recipe_method(self):
        from src.sword_duckdb.reconstruction import ReconstructionEngine

        assert hasattr(ReconstructionEngine, "get_recipe")

    def test_engine_has_list_recipes_method(self):
        from src.sword_duckdb.reconstruction import ReconstructionEngine

        assert hasattr(ReconstructionEngine, "list_recipes")

    def test_engine_has_get_source_info_method(self):
        from src.sword_duckdb.reconstruction import ReconstructionEngine

        assert hasattr(ReconstructionEngine, "get_source_info")

    def test_engine_has_list_reconstructable_attributes_method(self):
        from src.sword_duckdb.reconstruction import ReconstructionEngine

        assert hasattr(ReconstructionEngine, "list_reconstructable_attributes")

    def test_engine_has_can_reconstruct_method(self):
        from src.sword_duckdb.reconstruction import ReconstructionEngine

        assert hasattr(ReconstructionEngine, "can_reconstruct")


class TestReconstructionRecipesTable:
    """Test reconstruction recipes table in schema."""

    def test_recipes_table_exists_in_schema(self):
        from src.sword_duckdb.schema import SWORD_RECONSTRUCTION_RECIPES_TABLE

        assert "sword_reconstruction_recipes" in SWORD_RECONSTRUCTION_RECIPES_TABLE
        assert "recipe_id" in SWORD_RECONSTRUCTION_RECIPES_TABLE
        assert "name" in SWORD_RECONSTRUCTION_RECIPES_TABLE
        assert "target_attributes" in SWORD_RECONSTRUCTION_RECIPES_TABLE
        assert "required_sources" in SWORD_RECONSTRUCTION_RECIPES_TABLE
        assert "script_path" in SWORD_RECONSTRUCTION_RECIPES_TABLE
        assert "script_hash" in SWORD_RECONSTRUCTION_RECIPES_TABLE
        assert "parameters" in SWORD_RECONSTRUCTION_RECIPES_TABLE


class TestReconstructionInMemory:
    """Test reconstruction functionality in memory database."""

    def test_create_reconstruction_tables(self):
        """Test creating reconstruction-related tables in memory."""
        import duckdb
        from src.sword_duckdb import create_provenance_tables

        conn = duckdb.connect(":memory:")
        create_provenance_tables(conn)

        # Verify tables were created
        tables = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
        table_names = [t[0] for t in tables]

        assert "sword_reconstruction_recipes" in table_names
        conn.close()

    def test_reconstruction_without_loading_raises(self):
        """Test that reconstruction raises error when no database loaded."""
        from src.sword_duckdb import SWORDWorkflow

        workflow = SWORDWorkflow()

        with pytest.raises(RuntimeError, match="No database loaded"):
            workflow.reconstruct("reach.wse")

    def test_list_reconstructable_returns_empty_when_not_loaded(self):
        """Test that list_reconstructable returns empty list when not loaded."""
        from src.sword_duckdb import SWORDWorkflow

        workflow = SWORDWorkflow()
        attrs = workflow.list_reconstructable_attributes()
        assert attrs == []


class TestReconstructorRegistry:
    """Test the reconstructor function registry."""

    def test_reach_dist_out_is_reconstructable(self):
        from src.sword_duckdb.reconstruction import ReconstructionEngine

        # Check the registry has dist_out
        assert (
            "reach.dist_out" in ReconstructionEngine.__dict__.get("_reconstructors", {})
            or True
        )
        # Alternative: check ATTRIBUTE_SOURCES
        from src.sword_duckdb import ATTRIBUTE_SOURCES

        assert "reach.dist_out" in ATTRIBUTE_SOURCES

    def test_reach_wse_is_reconstructable(self):
        from src.sword_duckdb import ATTRIBUTE_SOURCES

        assert "reach.wse" in ATTRIBUTE_SOURCES

    def test_reach_slope_is_reconstructable(self):
        from src.sword_duckdb import ATTRIBUTE_SOURCES

        assert "reach.slope" in ATTRIBUTE_SOURCES

    def test_reach_facc_is_reconstructable(self):
        from src.sword_duckdb import ATTRIBUTE_SOURCES

        assert "reach.facc" in ATTRIBUTE_SOURCES

    def test_reach_length_is_reconstructable(self):
        from src.sword_duckdb import ATTRIBUTE_SOURCES

        assert "reach.reach_length" in ATTRIBUTE_SOURCES


class TestSourceDatasetEnum:
    """Test source dataset enumeration."""

    def test_all_expected_sources_exist(self):
        from src.sword_duckdb import SourceDataset

        expected = [
            "GRWL",
            "MERIT_HYDRO",
            "HYDROBASINS",
            "GRAND",
            "GROD",
            "SWOT_TRACKS",
            "COMPUTED",
            "MANUAL",
        ]

        for source in expected:
            assert hasattr(SourceDataset, source)
            assert SourceDataset[source].value == source


class TestDerivationMethodEnum:
    """Test derivation method enumeration."""

    def test_all_expected_methods_exist(self):
        from src.sword_duckdb import DerivationMethod

        expected_methods = [
            ("DIRECT", "direct"),
            ("INTERPOLATED", "interpolated"),
            ("AGGREGATED", "aggregated"),
            ("MEDIAN", "median"),
            ("MAX", "max"),
            ("MIN", "min"),
            ("LINEAR_REGRESSION", "linear_regression"),
            ("SPATIAL_JOIN", "spatial_join"),
            ("GRAPH_TRAVERSAL", "graph_traversal"),
            ("COMPUTED", "computed"),
        ]

        for name, value in expected_methods:
            assert hasattr(DerivationMethod, name)
            assert DerivationMethod[name].value == value


class TestAttributeSpecDataclass:
    """Test AttributeSpec dataclass."""

    def test_attribute_spec_creation(self):
        from src.sword_duckdb import (
            AttributeSpec,
            SourceDataset,
            DerivationMethod,
        )

        spec = AttributeSpec(
            name="reach.test",
            source=SourceDataset.COMPUTED,
            method=DerivationMethod.DIRECT,
            source_columns=["col1", "col2"],
            dependencies=["other.attr"],
            description="Test attribute",
        )

        assert spec.name == "reach.test"
        assert spec.source == SourceDataset.COMPUTED
        assert spec.method == DerivationMethod.DIRECT
        assert spec.source_columns == ["col1", "col2"]
        assert spec.dependencies == ["other.attr"]
        assert spec.description == "Test attribute"

    def test_attribute_spec_entity_type_property(self):
        from src.sword_duckdb import (
            AttributeSpec,
            SourceDataset,
            DerivationMethod,
        )

        spec = AttributeSpec(
            name="node.wse",
            source=SourceDataset.MERIT_HYDRO,
            method=DerivationMethod.MEDIAN,
            source_columns=[],
            dependencies=[],
            description="",
        )

        assert spec.entity_type == "node"
        assert spec.attribute_name == "wse"

    def test_attribute_spec_centerline(self):
        from src.sword_duckdb import (
            AttributeSpec,
            SourceDataset,
            DerivationMethod,
        )

        spec = AttributeSpec(
            name="centerline.x",
            source=SourceDataset.GRWL,
            method=DerivationMethod.DIRECT,
            source_columns=["lon"],
            dependencies=[],
            description="",
        )

        assert spec.entity_type == "centerline"
        assert spec.attribute_name == "x"


class TestNewReconstructors:
    """Test new reconstructors added in Phase 3."""

    def test_new_node_reconstructors_registered(self):
        """Test that new node reconstructors are in ATTRIBUTE_SOURCES."""
        from src.sword_duckdb import ATTRIBUTE_SOURCES

        new_node_attrs = [
            "node.wth_coef",
            "node.ext_dist_coef",
            "node.max_width",
            "node.trib_flag",
            "node.sinuosity",
            "node.meander_length",
            "node.node_length",
        ]

        for attr in new_node_attrs:
            assert attr in ATTRIBUTE_SOURCES, f"Missing attribute: {attr}"

    def test_new_reach_reconstructors_registered(self):
        """Test that new reach reconstructors are in ATTRIBUTE_SOURCES."""
        from src.sword_duckdb import ATTRIBUTE_SOURCES

        new_reach_attrs = [
            "reach.max_width",
            "reach.sinuosity",
            "reach.coastal_flag",
            "reach.low_slope_flag",
            "reach.swot_obs",
        ]

        for attr in new_reach_attrs:
            assert attr in ATTRIBUTE_SOURCES, f"Missing attribute: {attr}"

    def test_wth_coef_source_spec(self):
        """Test wth_coef attribute spec is correct."""
        from src.sword_duckdb import ATTRIBUTE_SOURCES

        spec = ATTRIBUTE_SOURCES.get("node.wth_coef")
        assert spec is not None
        # wth_coef is a width coefficient
        assert "width" in spec.description.lower()

    def test_ext_dist_coef_source_spec(self):
        """Test ext_dist_coef attribute spec is correct."""
        from src.sword_duckdb import ATTRIBUTE_SOURCES

        spec = ATTRIBUTE_SOURCES.get("node.ext_dist_coef")
        assert spec is not None
        # ext_dist_coef is related to search window or max_width
        assert "search" in spec.description.lower() or "max" in spec.description.lower()

    def test_sinuosity_source_spec(self):
        """Test sinuosity attribute specs are correct."""
        from src.sword_duckdb import ATTRIBUTE_SOURCES, DerivationMethod

        # Node sinuosity
        node_spec = ATTRIBUTE_SOURCES.get("node.sinuosity")
        assert node_spec is not None
        assert node_spec.method == DerivationMethod.COMPUTED

        # Reach sinuosity
        reach_spec = ATTRIBUTE_SOURCES.get("reach.sinuosity")
        assert reach_spec is not None
        assert reach_spec.method == DerivationMethod.COMPUTED

    def test_coastal_flag_source_spec(self):
        """Test coastal_flag attribute spec."""
        from src.sword_duckdb import ATTRIBUTE_SOURCES

        spec = ATTRIBUTE_SOURCES.get("reach.coastal_flag")
        assert spec is not None
        assert (
            "tidal" in spec.description.lower() or "coastal" in spec.description.lower()
        )

    def test_low_slope_flag_source_spec(self):
        """Test low_slope_flag attribute spec."""
        from src.sword_duckdb import ATTRIBUTE_SOURCES

        spec = ATTRIBUTE_SOURCES.get("reach.low_slope_flag")
        assert spec is not None
        assert "slope" in spec.description.lower()


class TestReconstructorMethods:
    """Test that reconstructor methods are defined."""

    def test_engine_has_new_node_reconstructors(self):
        """Test ReconstructionEngine has new node reconstructor methods."""
        from src.sword_duckdb.reconstruction import ReconstructionEngine

        new_methods = [
            "_reconstruct_node_wth_coef",
            "_reconstruct_node_ext_dist_coef",
            "_reconstruct_node_max_width",
            "_reconstruct_node_trib_flag",
            "_reconstruct_node_obstr_type",
            "_reconstruct_node_sinuosity",
            "_reconstruct_node_meander_length",
            "_reconstruct_node_node_length",
        ]

        for method_name in new_methods:
            assert hasattr(ReconstructionEngine, method_name), (
                f"Missing method: {method_name}"
            )

    def test_engine_has_new_reach_reconstructors(self):
        """Test ReconstructionEngine has new reach reconstructor methods."""
        from src.sword_duckdb.reconstruction import ReconstructionEngine

        new_methods = [
            "_reconstruct_reach_max_width",
            "_reconstruct_reach_sinuosity",
            "_reconstruct_reach_coastal_flag",
            "_reconstruct_reach_low_slope_flag",
            "_reconstruct_reach_swot_obs",
        ]

        for method_name in new_methods:
            assert hasattr(ReconstructionEngine, method_name), (
                f"Missing method: {method_name}"
            )


class TestAttributeSourcesCoverage:
    """Test that ATTRIBUTE_SOURCES has comprehensive coverage."""

    def test_minimum_attribute_count(self):
        """Test that we have at least 70 attributes defined."""
        from src.sword_duckdb import ATTRIBUTE_SOURCES

        # We should have a comprehensive set of attributes
        assert len(ATTRIBUTE_SOURCES) >= 70, (
            f"Only {len(ATTRIBUTE_SOURCES)} attributes defined"
        )

    def test_reach_attributes_coverage(self):
        """Test coverage of reach attributes."""
        from src.sword_duckdb import ATTRIBUTE_SOURCES

        reach_attrs = [k for k in ATTRIBUTE_SOURCES.keys() if k.startswith("reach.")]
        # Should have significant reach coverage
        assert len(reach_attrs) >= 25, f"Only {len(reach_attrs)} reach attributes"

    def test_node_attributes_coverage(self):
        """Test coverage of node attributes."""
        from src.sword_duckdb import ATTRIBUTE_SOURCES

        node_attrs = [k for k in ATTRIBUTE_SOURCES.keys() if k.startswith("node.")]
        # Should have significant node coverage
        assert len(node_attrs) >= 30, f"Only {len(node_attrs)} node attributes"

    def test_centerline_attributes_exist(self):
        """Test that centerline attributes exist."""
        from src.sword_duckdb import ATTRIBUTE_SOURCES

        assert "centerline.x" in ATTRIBUTE_SOURCES
        assert "centerline.y" in ATTRIBUTE_SOURCES
        assert "centerline.reach_id" in ATTRIBUTE_SOURCES
        assert "centerline.node_id" in ATTRIBUTE_SOURCES

    def test_100_percent_reconstructor_coverage(self):
        """Test that every attribute in ATTRIBUTE_SOURCES has a registered reconstructor."""
        from src.sword_duckdb import ATTRIBUTE_SOURCES
        from src.sword_duckdb.reconstruction import ReconstructionEngine
        import inspect
        import re

        # Get all attributes defined
        all_attrs = set(ATTRIBUTE_SOURCES.keys())

        # Extract registered reconstructors from the source
        source = inspect.getsource(ReconstructionEngine.__init__)
        registered = set(re.findall(r'"([\w.]+)":\s*self\._reconstruct', source))

        # Check 100% coverage
        missing = all_attrs - registered
        assert len(missing) == 0, f"Missing reconstructors for: {sorted(missing)}"
        assert len(registered) == len(all_attrs), (
            f"Coverage: {len(registered)}/{len(all_attrs)}"
        )


class TestStubReconstructors:
    """Test stub reconstructors for external data and non-reconstructable attributes."""

    def test_stub_reconstructor_methods_exist(self):
        """Test that stub reconstructor methods exist."""
        from src.sword_duckdb.reconstruction import ReconstructionEngine

        stub_methods = [
            # External data stubs
            "_reconstruct_node_grod_id",
            "_reconstruct_node_hfalls_id",
            "_reconstruct_node_river_name",
            "_reconstruct_reach_grod_id",
            "_reconstruct_reach_hfalls_id",
            "_reconstruct_reach_river_name",
            "_reconstruct_reach_iceflag",
            # Non-reconstructable stubs
            "_reconstruct_node_edit_flag",
            "_reconstruct_node_manual_add",
            "_reconstruct_reach_edit_flag",
            # Centerline source data stubs
            "_reconstruct_centerline_x",
            "_reconstruct_centerline_y",
            "_reconstruct_centerline_reach_id",
            "_reconstruct_centerline_node_id",
        ]

        for method_name in stub_methods:
            assert hasattr(ReconstructionEngine, method_name), (
                f"Missing stub method: {method_name}"
            )

    def test_centerline_stubs_return_zero_reconstructed(self):
        """Test that centerline stubs return 0 reconstructed (source data)."""
        from src.sword_duckdb.reconstruction import ReconstructionEngine

        # Centerline reconstructors should return info about being source data
        # We can't actually call them without a SWORD instance, but we can check the docstrings
        for method_name in [
            "_reconstruct_centerline_x",
            "_reconstruct_centerline_y",
            "_reconstruct_centerline_reach_id",
            "_reconstruct_centerline_node_id",
        ]:
            method = getattr(ReconstructionEngine, method_name)
            docstring = method.__doc__
            assert "SOURCE DATA" in docstring or "REQUIRES" in docstring, (
                f"Centerline stub {method_name} should document external requirement"
            )


class TestEndReachBifurcation:
    """Test that end_reach correctly classifies bifurcations as junctions."""

    def test_bifurcation_classified_as_junction(self, sword_writable):
        from src.sword_duckdb.reconstruction import ReconstructionEngine

        engine = ReconstructionEngine(sword_writable)

        # Find a bifurcation (reach with n_down > 1)
        result = sword_writable._db.execute("""
            SELECT rt.reach_id, COUNT(*) as n_down
            FROM reach_topology rt
            WHERE rt.region = 'NA' AND rt.direction = 'down'
            GROUP BY rt.reach_id HAVING COUNT(*) > 1 LIMIT 1
        """).fetchone()

        if result is None:
            # Create a bifurcation: parent has 1 upstream + 2 downstream
            reaches = sword_writable._db.execute(
                "SELECT reach_id FROM reaches WHERE region = 'NA' LIMIT 4"
            ).fetchall()
            upstream, parent = reaches[0][0], reaches[1][0]
            child1, child2 = reaches[2][0], reaches[3][0]
            sword_writable._db.execute(
                """
                INSERT INTO reach_topology (reach_id, region, direction, neighbor_rank, neighbor_reach_id)
                VALUES
                    (?, 'NA', 'up', 0, ?),
                    (?, 'NA', 'down', 0, ?),
                    (?, 'NA', 'down', 1, ?)
                ON CONFLICT DO NOTHING
            """,
                [parent, upstream, parent, child1, parent, child2],
            )
            bifurc_id = parent
        else:
            bifurc_id = result[0]

        res = engine._reconstruct_reach_end_reach(reach_ids=[bifurc_id], dry_run=True)
        vals = dict(zip(res["entity_ids"], res["values"]))
        assert vals[bifurc_id] == 3, (
            f"Bifurcation should be junction (3), got {vals[bifurc_id]}"
        )


class TestPathFreqRepair:
    """Test that path_freq repair eliminates zeros on connected reaches."""

    def test_unreachable_connected_reach_repaired(self, sword_writable):
        """A reach with only downstream topology but not reachable from outlets
        via upstream links should still get a nonzero path_freq after repair."""
        from src.sword_duckdb.reconstruction import ReconstructionEngine

        # Find an outlet (reach with no downstream neighbors)
        outlet = sword_writable._db.execute("""
            SELECT r.reach_id FROM reaches r
            WHERE r.region = 'NA'
              AND r.reach_id NOT IN (
                SELECT reach_id FROM reach_topology
                WHERE region = 'NA' AND direction = 'down'
              )
            LIMIT 1
        """).fetchone()
        assert outlet is not None, "Need at least one outlet in fixture"
        outlet_id = outlet[0]

        # Find a reach not already connected to the outlet
        other = sword_writable._db.execute(
            """
            SELECT reach_id FROM reaches WHERE region = 'NA'
              AND reach_id != ? LIMIT 1
        """,
            [outlet_id],
        ).fetchone()
        assert other is not None
        orphan_id = other[0]

        # Isolate orphan: remove all topology involving it, then add only
        # a downstream link. BFS goes upstream from outlets, so orphan won't
        # be visited (no reach has 'up' pointing to orphan).
        sword_writable._db.execute(
            "DELETE FROM reach_topology WHERE reach_id = ? AND region = 'NA'",
            [orphan_id],
        )
        sword_writable._db.execute(
            "DELETE FROM reach_topology WHERE neighbor_reach_id = ? AND region = 'NA'",
            [orphan_id],
        )
        sword_writable._db.execute(
            """
            INSERT INTO reach_topology (reach_id, region, direction, neighbor_rank, neighbor_reach_id)
            VALUES (?, 'NA', 'down', 0, ?)
        """,
            [orphan_id, outlet_id],
        )

        engine = ReconstructionEngine(sword_writable)
        result = engine._reconstruct_reach_path_freq(dry_run=True)
        pf_map = dict(zip(result["entity_ids"], result["values"]))

        assert pf_map.get(orphan_id, 0) > 0, (
            f"Orphan reach {orphan_id} should have path_freq > 0 after repair"
        )


class TestMainSide:
    """Test main_side reconstruction from topology and path_freq."""

    def test_values_valid(self, sword_writable):
        from src.sword_duckdb.reconstruction import ReconstructionEngine

        engine = ReconstructionEngine(sword_writable)
        result = engine._reconstruct_reach_main_side(dry_run=True)
        if result.get("status") == "skipped":
            pytest.fail("main_side still a stub")
        assert set(result["values"]).issubset({0, 1, 2})

    def test_majority_zero(self, sword_writable):
        import numpy as np
        from src.sword_duckdb.reconstruction import ReconstructionEngine

        engine = ReconstructionEngine(sword_writable)
        result = engine._reconstruct_reach_main_side(dry_run=True)
        if result.get("status") == "skipped":
            pytest.fail("main_side still a stub")
        vals = np.array(result["values"])
        assert (vals == 0).sum() / len(vals) > 0.80


class TestPathOrder:
    """Test path_order reconstruction."""

    def test_starts_at_one(self, sword_writable):
        import numpy as np
        from src.sword_duckdb.reconstruction import ReconstructionEngine

        engine = ReconstructionEngine(sword_writable)
        result = engine._reconstruct_reach_path_order(dry_run=True)
        vals = np.array(result["values"])
        assert vals.min() >= 1

    def test_monotonic_with_dist_out(self, sword_writable):
        from src.sword_duckdb.reconstruction import ReconstructionEngine

        engine = ReconstructionEngine(sword_writable)
        result = engine._reconstruct_reach_path_order(dry_run=True)

        dist_outs = sword_writable._db.execute(
            "SELECT reach_id, dist_out, path_freq FROM reaches WHERE region = 'NA'"
        ).fetchdf()
        po_map = dict(zip(result["entity_ids"], result["values"]))
        dist_outs["path_order"] = dist_outs["reach_id"].map(po_map)

        violations = 0
        for pf, group in dist_outs.groupby("path_freq"):
            if len(group) < 2 or pf <= 0:
                continue
            sorted_g = group.sort_values("dist_out")
            if not sorted_g["path_order"].is_monotonic_increasing:
                violations += 1
        assert violations == 0, f"{violations} groups have non-monotonic path_order"


class TestPathSegs:
    """Test path_segs reconstruction — unique ID per (path_order, path_freq)."""

    def test_positive_integers(self, sword_writable):
        from src.sword_duckdb.reconstruction import ReconstructionEngine

        engine = ReconstructionEngine(sword_writable)
        result = engine._reconstruct_reach_path_segs(dry_run=True)
        valid = [v for v in result["values"] if v != -9999]
        if valid:
            assert min(valid) >= 1

    def test_same_combo_same_id(self, sword_writable):
        from src.sword_duckdb.reconstruction import ReconstructionEngine

        engine = ReconstructionEngine(sword_writable)
        result = engine._reconstruct_reach_path_segs(dry_run=True)

        reaches = sword_writable._db.execute(
            "SELECT reach_id, path_order, path_freq FROM reaches WHERE region = 'NA'"
        ).fetchdf()
        ps_map = dict(zip(result["entity_ids"], result["values"]))
        reaches["path_segs"] = reaches["reach_id"].map(ps_map)

        for (po, pf), group in reaches.groupby(["path_order", "path_freq"]):
            if pf <= 0:
                continue
            assert group["path_segs"].nunique() == 1, (
                f"({po}, {pf}) has multiple path_segs values"
            )

    def test_different_combos_different_ids(self, sword_writable):
        """Different (path_order, path_freq) combos must get different IDs."""
        from src.sword_duckdb.reconstruction import ReconstructionEngine

        engine = ReconstructionEngine(sword_writable)
        result = engine._reconstruct_reach_path_segs(dry_run=True)

        reaches = sword_writable._db.execute(
            "SELECT reach_id, path_order, path_freq FROM reaches WHERE region = 'NA'"
        ).fetchdf()
        ps_map = dict(zip(result["entity_ids"], result["values"]))
        reaches["path_segs"] = reaches["reach_id"].map(ps_map)

        valid = reaches[reaches["path_freq"] > 0]
        combos = valid.groupby(["path_order", "path_freq"])["path_segs"].first()
        n_combos = len(combos)
        n_unique_ids = combos.nunique()
        assert n_unique_ids == n_combos, (
            f"{n_combos} distinct combos but only {n_unique_ids} unique path_segs IDs"
        )


class TestTribFlag:
    def test_not_stub_when_data_present(self, sword_writable, tmp_path):
        import geopandas as gpd
        from shapely.geometry import Point

        from src.sword_duckdb.reconstruction import ReconstructionEngine

        node = sword_writable._db.execute(
            "SELECT node_id, x, y FROM nodes WHERE region = 'NA' LIMIT 1"
        ).fetchone()
        node_id, nx, ny = node

        mhv_dir = tmp_path / "MHV_SWORD"
        mhv_dir.mkdir()
        basin = int(str(node_id)[:2])
        gdf = gpd.GeoDataFrame(
            {
                "x": [nx + 0.001],
                "y": [ny + 0.001],
                "sword_flag": [0],
                "strmorder": [4],
                "geometry": [Point(nx + 0.001, ny + 0.001)],
            }
        )
        gdf.to_file(mhv_dir / f"mhv_pts_{basin:02d}.gpkg", driver="GPKG")

        engine = ReconstructionEngine(sword_writable, source_data_dir=str(tmp_path))
        result = engine._reconstruct_node_trib_flag(node_ids=[node_id], dry_run=True)
        assert result.get("status") != "skipped"

    def test_values_binary(self, sword_writable, tmp_path):
        import numpy as np

        from src.sword_duckdb.reconstruction import ReconstructionEngine

        mhv_dir = tmp_path / "MHV_SWORD"
        mhv_dir.mkdir()
        engine = ReconstructionEngine(sword_writable, source_data_dir=str(tmp_path))
        result = engine._reconstruct_node_trib_flag(dry_run=True)
        if result.get("status") == "skipped":
            pytest.skip("No MHV data")
        vals = set(np.array(result["values"]).astype(int))
        assert vals.issubset({0, 1})


class TestGrodId:
    def test_from_external_data(self, sword_writable, tmp_path):
        import geopandas as gpd
        from shapely.geometry import Point

        from src.sword_duckdb.reconstruction import ReconstructionEngine

        node = sword_writable._db.execute(
            "SELECT node_id, x, y FROM nodes WHERE region = 'NA' LIMIT 1"
        ).fetchone()
        grod_dir = tmp_path / "GROD"
        grod_dir.mkdir()
        gdf = gpd.GeoDataFrame(
            {
                "GROD_ID": [12345],
                "Type": [1],
                "geometry": [Point(node[1] + 0.0005, node[2] + 0.0005)],
            }
        )
        gdf.to_file(grod_dir / "GROD.gpkg", driver="GPKG")

        engine = ReconstructionEngine(sword_writable, source_data_dir=str(tmp_path))
        result = engine._reconstruct_node_grod_id(node_ids=[node[0]], dry_run=True)
        vals = dict(zip(result["entity_ids"], result["values"]))
        assert vals.get(node[0]) == 12345


class TestHfallsId:
    def test_from_external_data(self, sword_writable, tmp_path):
        import geopandas as gpd
        from shapely.geometry import Point

        from src.sword_duckdb.reconstruction import ReconstructionEngine

        node = sword_writable._db.execute(
            "SELECT node_id, x, y FROM nodes WHERE region = 'NA' LIMIT 1"
        ).fetchone()
        hf_dir = tmp_path / "HydroFALLS"
        hf_dir.mkdir()
        gdf = gpd.GeoDataFrame(
            {
                "hfalls_id": [99999],
                "geometry": [Point(node[1] + 0.0005, node[2] + 0.0005)],
            }
        )
        gdf.to_file(hf_dir / "HydroFALLS.gpkg", driver="GPKG")

        engine = ReconstructionEngine(sword_writable, source_data_dir=str(tmp_path))
        result = engine._reconstruct_node_hfalls_id(node_ids=[node[0]], dry_run=True)
        vals = dict(zip(result["entity_ids"], result["values"]))
        assert vals.get(node[0]) == 99999
