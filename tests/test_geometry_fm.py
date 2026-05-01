import numpy as np
import pytest

import mikeio
from mikeio.exceptions import OutsideModelDomainError
from mikeio.spatial import GeometryFM2D, GeometryFM3D, GeometryPoint2D


@pytest.fixture
def simple_3d_geom() -> GeometryFM3D:
    #     x     y    z
    nc = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 0.0, -1.0),
        (1.0, 0.0, -1.0),
        (1.0, 1.0, -1.0),
        (0.0, 0.0, -2.0),
        (1.0, 0.0, -2.0),
        (1.0, 1.0, -2.0),
    ]

    el = [(0, 1, 2, 3, 4, 5), (3, 4, 5, 6, 7, 8)]

    g = GeometryFM3D(
        node_coordinates=nc,
        element_table=el,
        projection="LONG/LAT",
        n_layers=2,
        n_sigma=2,
    )

    return g


def test_isel_list_of_indices(simple_3d_geom: GeometryFM3D) -> None:
    g = simple_3d_geom

    g1 = g.isel([0, 1])
    assert isinstance(g1, GeometryFM3D)
    assert g1.element_coordinates[0, 0] == pytest.approx(0.6666666666666666)

    # you can get elements in arbitrary order
    g2 = g.isel([1, 0])
    assert isinstance(g2, GeometryFM3D)
    assert g2.element_coordinates[1, 0] == pytest.approx(0.6666666666666666)


def test_basic() -> None:
    #     x     y    z
    nc = [
        (0.0, 0.0, 0.0),  # 0
        (1.0, 0.0, 0.0),  # 1
        (0.5, 1.0, 0.0),  # 2
    ]

    el = [(0, 1, 2)]

    g = GeometryFM2D(nc, el)
    assert g.n_elements == 1
    assert g.n_nodes == 3
    assert g.is_geo
    assert g.is_tri_only
    assert g.projection == "LONG/LAT"
    assert not g.is_layered
    assert 0 in g.find_index(0.5, 0.5)
    with pytest.raises(ValueError):
        g.find_index(50.0, -50.0)

    assert "nodes: 3" in repr(g)


def test_too_many_elements() -> None:
    #     x     y    z
    nc = [
        (0.0, 0.0, 0.0),  # 0
        (1.0, 0.0, 0.0),  # 1
        (0.5, 1.0, 0.0),  # 2
    ]

    el = [(0, 1, 2, 3)]  # There is no node #3

    with pytest.raises(ValueError) as excinfo:
        GeometryFM2D(nc, el)

    assert "element" in str(excinfo.value).lower()


def test_overset_grid() -> None:
    #     x     y    z
    nc = [
        (0.0, 0.0, 0.0),  # 0
        (1.0, 0.0, 0.0),  # 1
        (0.5, 1.0, 0.0),  # 2
    ]

    el = [(0, 1, 2)]

    proj = "UTM-33"
    g = GeometryFM2D(nc, el, projection=proj)
    grid = g.get_overset_grid(dx=0.5)
    assert grid.nx == 2
    assert grid.ny == 2
    assert grid.projection_string == proj


def test_area() -> None:
    #     x     y    z
    nc = [
        (0.0, 0.0, 0.0),  # 0
        (1.0, 0.0, 0.0),  # 1
        (1.0, 1.0, 0.0),  # 2
        (0.0, 1.0, 0.0),  # 3
    ]

    el = [(0, 1, 2, 3)]

    g = GeometryFM2D(node_coordinates=nc, element_table=el, projection="LONG/LAT")
    assert not g.is_tri_only
    area = g.get_element_area()
    assert len(area) == g.n_elements
    assert area > 0.0


def test_find_index_simple_domain() -> None:
    #     x     y    z
    nc = [
        (0.0, 0.0, 0.0),  # 0
        (1.0, 0.0, 0.0),  # 1
        (1.0, 1.0, 0.0),  # 2
        (0.0, 1.0, 0.0),  # 3
        (0.5, 1.5, 0.0),  # 4
    ]

    el = [(0, 1, 2), (0, 2, 3), (3, 2, 4)]

    g = GeometryFM2D(node_coordinates=nc, element_table=el, projection="LONG/LAT")
    idx = g.find_index(0.5, 0.1)
    assert 0 in idx

    # look for multiple points in the same call
    idx = g.find_index(coords=[(0.5, 0.1), (0.1, 0.5)])

    # TODO checking for subsets can only be done if this is a set
    # assert {0, 1} <= idx
    assert 0 in idx
    assert 1 in idx

    # look for the same points multiple times
    idx = g.find_index(coords=[(0.5, 0.1), (0.5, 0.1)])

    # the current behavior is to return the same index twice
    # but this could be changed to return only unique indices
    assert len(idx) == 2
    assert 0 in idx

    with pytest.raises(OutsideModelDomainError) as ex:
        g.find_index(-0.5, -0.1)

    assert ex.value.x == -0.5
    assert ex.value.y == -0.1
    assert 0 in ex.value.indices


def test_isel_simple_domain() -> None:
    #     x     y    z
    nc = [
        (0.0, 0.0, 0.0),  # 0
        (1.0, 0.0, 0.0),  # 1
        (1.0, 1.0, 0.0),  # 2
        (0.0, 1.0, 0.0),  # 3
        (0.5, 1.5, 0.0),  # 4
    ]

    el = [(0, 1, 2), (0, 2, 3), (3, 2, 4)]

    g = GeometryFM2D(node_coordinates=nc, element_table=el, projection="LONG/LAT")
    gp = g.isel(0)
    assert isinstance(gp, GeometryPoint2D)
    assert gp.projection == g.projection


def test_isel_list_of_indices_simple_domain() -> None:
    #     x     y    z
    nc = [
        (0.0, 0.0, 0.0),  # 0
        (1.0, 0.0, 0.0),  # 1
        (1.0, 1.0, 0.0),  # 2
        (0.0, 1.0, 0.0),  # 3
        (0.5, 1.5, 0.0),  # 4
    ]

    el = [(0, 1, 2), (0, 2, 3), (3, 2, 4)]

    g = GeometryFM2D(node_coordinates=nc, element_table=el, projection="LONG/LAT")
    g1 = g.isel([0, 1])
    assert isinstance(g1, GeometryFM2D)
    assert g1.element_coordinates[0, 0] == pytest.approx(0.6666666666666666)

    g2 = g.isel([1, 0])
    assert isinstance(g2, GeometryFM2D)
    assert g2.element_coordinates[1, 0] == pytest.approx(0.6666666666666666)


def test_plot_mesh() -> None:
    #     x     y    z
    nc = [
        (0.0, 0.0, 0.0),  # 0
        (1.0, 0.0, 0.0),  # 1
        (1.0, 1.0, 0.0),  # 2
        (0.0, 1.0, 0.0),  # 3
    ]

    el = [(0, 1, 2), (0, 2, 3)]

    g = GeometryFM2D(node_coordinates=nc, element_table=el, projection="LONG/LAT")
    assert g.n_elements == 2
    g.plot.mesh()


def test_layered(simple_3d_geom: GeometryFM3D) -> None:
    g = simple_3d_geom

    assert g.n_elements == 2
    assert g.n_layers == 2

    assert len(g.top_elements) == 1

    # find vertical column
    idx = g.find_index(x=0.9, y=0.1)
    assert len(idx) == 2

    # subset
    g2 = g.isel(idx)
    assert isinstance(g2, GeometryFM3D)
    assert g2.n_elements == 2

    assert "elements: 2" in repr(g2)
    assert "layers: 2" in repr(g2)


def test_equality() -> None:
    nc = [
        (0.0, 0.0, 0.0),  # 0
        (1.0, 0.0, 0.0),  # 1
        (1.0, 1.0, 0.0),  # 2
        (0.0, 1.0, 0.0),  # 3
    ]

    el = [(0, 1, 2), (0, 2, 3)]

    g = GeometryFM2D(node_coordinates=nc, element_table=el, projection="LONG/LAT")
    g2 = GeometryFM2D(node_coordinates=nc, element_table=el, projection="LONG/LAT")

    assert g == g2

    g3 = GeometryFM2D(node_coordinates=nc, element_table=el, projection="UTM-33")
    assert g != g3


def test_equality_shifted_coords() -> None:
    nc1 = [
        (0.0, 0.0, 0.0),  # 0
        (1.0, 0.0, 0.0),  # 1
        (1.0, 1.0, 0.0),  # 2
        (0.0, 1.0, 0.0),  # 3
    ]

    el = [(0, 1, 2), (0, 2, 3)]
    g = GeometryFM2D(node_coordinates=nc1, element_table=el, projection="LONG/LAT")

    nc2 = [
        (0.1, 0.0, 0.0),  # 0
        (1.0, 0.0, 0.0),  # 1
        (1.0, 1.0, 0.0),  # 2
        (0.1, 1.0, 0.0),  # 3
    ]

    g2 = GeometryFM2D(node_coordinates=nc2, element_table=el, projection="LONG/LAT")
    assert g != g2


# --- from_2d_geometry tests ---


@pytest.fixture
def triangle_2d() -> GeometryFM2D:
    """Simple 1-triangle 2D mesh with bathymetry at z=-10."""
    nc = [
        (0.0, 0.0, -10.0),
        (1.0, 0.0, -10.0),
        (0.5, 1.0, -10.0),
    ]
    el = [(0, 1, 2)]
    return GeometryFM2D(nc, el, projection="UTM-33")


@pytest.fixture
def two_triangles_2d() -> GeometryFM2D:
    """2-triangle mesh with different bathymetry depths."""
    nc = [
        (0.0, 0.0, -10.0),
        (1.0, 0.0, -8.0),
        (1.0, 1.0, -6.0),
        (0.0, 1.0, -12.0),
    ]
    el = [(0, 1, 2), (0, 2, 3)]
    return GeometryFM2D(nc, el, projection="UTM-33")


class TestFromTwoDGeometrySigma:
    """Tests for GeometryFM3D.from_2d_geometry with pure sigma layers."""

    def test_basic_counts(self, triangle_2d: GeometryFM2D) -> None:
        g3d = GeometryFM3D.from_2d_geometry(triangle_2d, n_sigma=3)
        assert g3d.n_nodes == 3 * (3 + 1)  # 12
        assert g3d.n_elements == 1 * 3  # 3

    def test_element_table_structure(self, triangle_2d: GeometryFM2D) -> None:
        g3d = GeometryFM3D.from_2d_geometry(triangle_2d, n_sigma=3)
        # 3 prism elements, bottom to top
        # 2D nodes [0, 1, 2], npl = 4 (nodes per level)
        # Layer 0 (bottom): [0*4+0, 1*4+0, 2*4+0, 0*4+1, 1*4+1, 2*4+1] = [0,4,8,1,5,9]
        assert len(g3d.element_table) == 3
        e0 = g3d.element_table[0]
        assert len(e0) == 6  # prism = 6 nodes
        np.testing.assert_array_equal(e0, [0, 4, 8, 1, 5, 9])

    def test_z_coordinates_equidistant(self, triangle_2d: GeometryFM2D) -> None:
        g3d = GeometryFM3D.from_2d_geometry(triangle_2d, n_sigma=4)
        # Node 0 column: z = -10, -7.5, -5, -2.5, 0 (equidistant)
        npl = 5  # 4 layers + 1
        z_col0 = g3d.node_coordinates[0:npl, 2]
        expected = [-10.0, -7.5, -5.0, -2.5, 0.0]
        np.testing.assert_array_almost_equal(z_col0, expected)

    def test_z_coordinates_nonequidistant(self, triangle_2d: GeometryFM2D) -> None:
        fractions = [0.5, 0.3, 0.2]  # bottom to top
        g3d = GeometryFM3D.from_2d_geometry(
            triangle_2d, n_sigma=3, layer_fractions=fractions
        )
        # Node 0: z_bot=-10, surface=0, depth=10
        # cumulative: [0, 0.5, 0.8, 1.0]
        # z = -10 + c * 10 → [-10, -5, -2, 0]
        npl = 4
        z_col0 = g3d.node_coordinates[0:npl, 2]
        np.testing.assert_array_almost_equal(z_col0, [-10.0, -5.0, -2.0, 0.0])

    def test_properties(self, triangle_2d: GeometryFM2D) -> None:
        g3d = GeometryFM3D.from_2d_geometry(triangle_2d, n_sigma=3)
        assert g3d.n_layers == 3
        assert g3d.n_sigma_layers == 3
        assert g3d.n_z_layers == 0
        assert g3d.is_layered

    def test_top_and_bottom_elements(self, triangle_2d: GeometryFM2D) -> None:
        g3d = GeometryFM3D.from_2d_geometry(triangle_2d, n_sigma=3)
        np.testing.assert_array_equal(g3d.top_elements, [2])
        np.testing.assert_array_equal(g3d.bottom_elements, [0])

    def test_multi_element(self, two_triangles_2d: GeometryFM2D) -> None:
        g3d = GeometryFM3D.from_2d_geometry(two_triangles_2d, n_sigma=3)
        assert g3d.n_nodes == 4 * (3 + 1)  # 16
        assert g3d.n_elements == 2 * 3  # 6
        assert len(g3d.top_elements) == 2
        assert len(g3d.bottom_elements) == 2

    def test_projection_preserved(self, triangle_2d: GeometryFM2D) -> None:
        g3d = GeometryFM3D.from_2d_geometry(triangle_2d, n_sigma=3)
        assert g3d.projection_string == "UTM-33"

    def test_xy_preserved(self, triangle_2d: GeometryFM2D) -> None:
        g3d = GeometryFM3D.from_2d_geometry(triangle_2d, n_sigma=2)
        # Each 2D node appears at 3 levels (n_sigma+1)
        npl = 3
        # Node 0 column: all should have x=0, y=0
        np.testing.assert_array_equal(g3d.node_coordinates[0:npl, 0], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(g3d.node_coordinates[0:npl, 1], [0.0, 0.0, 0.0])

    def test_codes_replicated(self, triangle_2d: GeometryFM2D) -> None:
        g3d = GeometryFM3D.from_2d_geometry(triangle_2d, n_sigma=3)
        # Default codes are 0 for all nodes; they should be replicated
        assert len(g3d.codes) == g3d.n_nodes

    def test_roundtrip_basin_3d(self) -> None:
        ds = mikeio.read("tests/testdata/basin_3d.dfsu", time=0)
        g_orig = ds.geometry
        assert isinstance(g_orig, GeometryFM3D)
        g2d = g_orig.to_2d_geometry()
        g_new = GeometryFM3D.from_2d_geometry(g2d, n_sigma=g_orig.n_sigma_layers)
        assert g_new.n_elements == g_orig.n_elements
        assert g_new.n_nodes == g_orig.n_nodes
        assert g_new.n_layers == g_orig.n_layers


class TestFromTwoDGeometrySigmaZ:
    """Tests for GeometryFM3D.from_2d_geometry with sigma-z layers."""

    def test_oresund_structure(self) -> None:
        """Verify sigma-z creation with oresund-like parameters."""
        ds = mikeio.read("tests/testdata/oresund_sigma_z.dfsu", time=0)
        g_orig = ds.geometry
        assert isinstance(g_orig, GeometryFM3D)
        g2d = g_orig.to_2d_geometry()
        # oresund: sigma_depth=-8, equidistant 5m z-layers
        g_new = GeometryFM3D.from_2d_geometry(
            g2d, n_sigma=4, sigma_depth=-8.0, z_layer_thickness=5.0
        )
        assert g_new.n_layers == g_orig.n_layers
        assert g_new.n_sigma_layers == g_orig.n_sigma_layers
        assert g_new.n_z_layers == g_orig.n_z_layers
        # Same number of 2D columns
        assert len(g_new.n_layers_per_column) == len(g_orig.n_layers_per_column)
        # Layer counts per column are in valid range
        assert all(4 <= n <= 9 for n in g_new.n_layers_per_column)

    def test_sigma_z_exact_reconstruction(self) -> None:
        """Verify exact reconstruction with a controlled 2-triangle mesh."""
        # sigma_depth=-8, z_layer_thickness=5 → z-levels at -13, -8
        # Node at -15: below -13 → 1 z-layer + partial bottom
        # Node at -5: above sigma depth -8 → 0 z-layers
        # Node at -3: above sigma depth → 0 z-layers
        # Node at -20: below -13 → 2 z-layers + partial bottom
        nc = [
            (0.0, 0.0, -15.0),
            (1.0, 0.0, -5.0),
            (1.0, 1.0, -3.0),
            (0.0, 1.0, -20.0),
        ]
        el = [(0, 1, 2), (0, 2, 3)]
        g2d = GeometryFM2D(nc, el, projection="UTM-33")

        g3d = GeometryFM3D.from_2d_geometry(
            g2d, n_sigma=2, sigma_depth=-8.0, z_layer_thickness=5.0
        )
        assert g3d.n_sigma_layers == 2
        nlpc = g3d.n_layers_per_column
        # Element 0 (nodes at -15, -5, -3): shallowest=-3, above -8 → 0 z-layers → 2 layers
        assert nlpc[0] == 2
        # Element 1 (nodes at -15, -3, -20): shallowest=-3, above -8 → 0 z-layers → 2 layers
        assert nlpc[1] == 2

    def test_sigma_z_variable_thickness(self) -> None:
        """Sigma-z with variable z-layer thicknesses (bottom to top)."""
        nc = [
            (0.0, 0.0, -20.0),
            (1.0, 0.0, -20.0),
            (0.5, 1.0, -20.0),
        ]
        el = [(0, 1, 2)]
        g2d = GeometryFM2D(nc, el, projection="UTM-33")

        # sigma_depth=-5, thicknesses [3, 5, 7] bottom to top
        # z-levels from sigma_depth down: -5, -12, -17, -20
        g3d = GeometryFM3D.from_2d_geometry(
            g2d, n_sigma=2, sigma_depth=-5.0, z_layer_thickness=[3, 5, 7]
        )
        assert g3d.n_sigma_layers == 2
        assert g3d.n_z_layers == 3
        assert g3d.n_layers == 5  # 2 sigma + 3 z

    def test_sigma_z_basic(self, two_triangles_2d: GeometryFM2D) -> None:
        """Sigma-z where some columns lose z-layers due to shallow bathymetry."""
        # Nodes at z=-10, -8, -6, -12
        # sigma_depth=-7, z_layer_thickness=2 → z-levels at -9, -7
        # Element 0 (nodes at -10,-8,-6): shallowest=-6, above sigma → 0 z-layers → 2 sigma
        # Element 1 (nodes at -10,-6,-12): shallowest=-6, above sigma → 0 z-layers → 2 sigma
        g3d = GeometryFM3D.from_2d_geometry(
            two_triangles_2d, n_sigma=2, sigma_depth=-7.0, z_layer_thickness=2.0
        )
        assert g3d.n_sigma_layers == 2

    def test_shallow_mesh_equals_pure_sigma(self) -> None:
        """If all bathymetry is above sigma depth, result should be like pure sigma."""
        nc = [
            (0.0, 0.0, -3.0),
            (1.0, 0.0, -2.0),
            (0.5, 1.0, -4.0),
        ]
        el = [(0, 1, 2)]
        g2d = GeometryFM2D(nc, el, projection="UTM-33")

        # sigma_depth=-6, z_layer_thickness=2 → z-levels at -8, -6
        # All nodes above -6 → no z-layers active → 3 layers only
        g3d = GeometryFM3D.from_2d_geometry(
            g2d, n_sigma=3, sigma_depth=-6.0, z_layer_thickness=2.0
        )
        assert all(n == 3 for n in g3d.n_layers_per_column)


class TestFromTwoDGeometryValidation:
    """Validation tests for from_2d_geometry."""

    def test_n_sigma_zero_raises(self, triangle_2d: GeometryFM2D) -> None:
        with pytest.raises(ValueError):
            GeometryFM3D.from_2d_geometry(triangle_2d, n_sigma=0)

    def test_wrong_fractions_length_raises(self, triangle_2d: GeometryFM2D) -> None:
        with pytest.raises(ValueError):
            GeometryFM3D.from_2d_geometry(
                triangle_2d, n_sigma=3, layer_fractions=[0.5, 0.5]
            )

    def test_fractions_not_summing_to_one_raises(
        self, triangle_2d: GeometryFM2D
    ) -> None:
        with pytest.raises(ValueError):
            GeometryFM3D.from_2d_geometry(
                triangle_2d, n_sigma=3, layer_fractions=[0.5, 0.3, 0.3]
            )

    def test_sigma_depth_without_thickness_raises(
        self, triangle_2d: GeometryFM2D
    ) -> None:
        with pytest.raises(ValueError):
            GeometryFM3D.from_2d_geometry(triangle_2d, n_sigma=3, sigma_depth=-5.0)

    def test_thickness_without_sigma_depth_raises(
        self, triangle_2d: GeometryFM2D
    ) -> None:
        with pytest.raises(ValueError):
            GeometryFM3D.from_2d_geometry(triangle_2d, n_sigma=3, z_layer_thickness=2.0)
