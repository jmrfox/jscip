"""Tests for ParameterSet with vector parameters."""

import numpy as np
import pytest

from jscip import ParameterSet


def test_parameterset_with_scalar_values():
    """Test ParameterSet with traditional scalar values."""
    ps = ParameterSet({"a": 1.0, "b": 2.0, "c": 3.0})
    assert ps["a"] == 1.0
    assert ps["b"] == 2.0
    assert ps["c"] == 3.0


def test_parameterset_with_vector_values():
    """Test ParameterSet can store numpy arrays."""
    ps = ParameterSet({"scalar": 1.0, "vector": np.array([1.0, 2.0, 3.0])})
    assert ps["scalar"] == 1.0
    assert isinstance(ps["vector"], np.ndarray)
    assert np.allclose(ps["vector"], [1.0, 2.0, 3.0])


def test_parameterset_mixed_scalar_vector():
    """Test ParameterSet with mixed scalar and vector parameters."""
    ps = ParameterSet({
        "mass": 1.5,
        "position": np.array([1.0, 2.0, 3.0]),
        "velocity": np.array([0.5, 0.5, 0.5]),
        "time": 2.0,
    })
    
    assert ps["mass"] == 1.5
    assert ps["time"] == 2.0
    assert np.allclose(ps["position"], [1.0, 2.0, 3.0])
    assert np.allclose(ps["velocity"], [0.5, 0.5, 0.5])


def test_parameterset_copy_preserves_arrays():
    """Test that copy() preserves array values."""
    original = ParameterSet({
        "scalar": 5.0,
        "vector": np.array([1.0, 2.0, 3.0])
    })
    
    copy = original.copy()
    
    assert copy["scalar"] == original["scalar"]
    assert np.allclose(copy["vector"], original["vector"])
    
    # Verify it's a true copy (modifying copy doesn't affect original)
    copy["vector"][0] = 999.0
    assert original["vector"][0] == 1.0  # Original unchanged


def test_parameterset_reindex_preserves_arrays():
    """Test that reindex() preserves array types."""
    ps = ParameterSet({
        "a": 1.0,
        "b": np.array([2.0, 3.0]),
        "c": 4.0,
    })
    
    reindexed = ps.reindex(["c", "b", "a"])
    
    assert reindexed["a"] == 1.0
    assert reindexed["c"] == 4.0
    assert isinstance(reindexed["b"], np.ndarray)
    assert np.allclose(reindexed["b"], [2.0, 3.0])


def test_parameterset_satisfies_with_vector_constraint():
    """Test that satisfies() works with constraints on vector parameters."""
    ps = ParameterSet({
        "position": np.array([1.0, 2.0, 3.0]),
        "scalar": 5.0,
    })
    
    # Constraint using vector parameter
    def norm_constraint(p):
        return np.linalg.norm(p["position"]) < 10.0
    
    assert ps.satisfies(norm_constraint)
    
    # Constraint that fails
    def strict_norm(p):
        return np.linalg.norm(p["position"]) < 1.0
    
    assert not ps.satisfies(strict_norm)


def test_parameterset_satisfies_mixed_constraint():
    """Test constraints that use both scalar and vector parameters."""
    ps = ParameterSet({
        "mass": 2.0,
        "velocity": np.array([1.0, 0.0, 0.0]),
    })
    
    # Kinetic energy constraint
    def kinetic_energy_constraint(p):
        v_squared = np.sum(p["velocity"] ** 2)
        ke = 0.5 * p["mass"] * v_squared
        return ke < 10.0
    
    assert ps.satisfies(kinetic_energy_constraint)


def test_parameterset_to_dict_with_vectors():
    """Test that to_dict() works with vector parameters."""
    ps = ParameterSet({
        "a": 1.0,
        "b": np.array([2.0, 3.0, 4.0]),
    })
    
    d = ps.to_dict()
    
    assert d["a"] == 1.0
    assert isinstance(d["b"], np.ndarray)
    assert np.allclose(d["b"], [2.0, 3.0, 4.0])


def test_parameterset_iteration_with_vectors():
    """Test that iteration works with vector parameters."""
    ps = ParameterSet({
        "scalar1": 1.0,
        "vector1": np.array([2.0, 3.0]),
        "scalar2": 4.0,
    })
    
    # Iterate over keys
    keys = list(ps.keys())
    assert keys == ["scalar1", "vector1", "scalar2"]
    
    # Iterate over items
    items = list(ps.items())
    assert items[0] == ("scalar1", 1.0)
    assert items[1][0] == "vector1"
    assert np.allclose(items[1][1], [2.0, 3.0])
    assert items[2] == ("scalar2", 4.0)


def test_parameterset_repr_with_vectors():
    """Test that __repr__ works with vector parameters."""
    ps = ParameterSet({
        "a": 1.0,
        "b": np.array([2.0, 3.0]),
    })
    
    repr_str = repr(ps)
    assert "ParameterSet" in repr_str


def test_parameterset_empty():
    """Test creating an empty ParameterSet."""
    ps = ParameterSet({})
    assert len(ps) == 0


def test_parameterset_vector_different_shapes():
    """Test ParameterSet with vectors of different shapes."""
    ps = ParameterSet({
        "vec2": np.array([1.0, 2.0]),
        "vec3": np.array([3.0, 4.0, 5.0]),
        "vec5": np.array([6.0, 7.0, 8.0, 9.0, 10.0]),
    })
    
    assert len(ps["vec2"]) == 2
    assert len(ps["vec3"]) == 3
    assert len(ps["vec5"]) == 5
