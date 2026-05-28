"""Example demonstrating integrated hypergrid functionality in ParameterBank.

This example shows how to use the new grid_points and grid_scale attributes
with ParameterBank.compute_hypergrid() to generate systematic parameter grids.
"""

from jscip import (
    IndependentScalarParameter,
    ParameterBank,
    DerivedScalarParameter,
)

print("=" * 60)
print("HyperGrid Integration Example")
print("=" * 60)

# Example 1: Basic linear grid
print("\n1. Basic Linear Grid:")
print("-" * 30)

# Create parameters with grid configuration
mass = IndependentScalarParameter(
    value=1.0,
    is_sampled=True,
    range=(0.5, 2.0),
    grid_points=3,  # Linear spacing: [0.5, 1.25, 2.0]
)

velocity = IndependentScalarParameter(
    value=10,
    is_sampled=True,
    range=(5, 15),
    grid_points=2,  # Linear spacing: [5.0, 15.0]
    param_type="float",
)  # explicit type setting

# Fixed parameter (not included in grid)
gravity = IndependentScalarParameter(value=9.81, is_sampled=False)

# Create parameter bank
bank = ParameterBank({"mass": mass, "velocity": velocity, "gravity": gravity})

# Generate hypergrid
grid = bank.compute_hypergrid()
print(f"Generated {len(grid)} grid points")
for i, point in enumerate(grid):
    format_str = (
        f"  Point {i}: mass={point['mass']:.2f}, "
        f"velocity={point['velocity']:.1f}, "
        f"gravity={point['gravity']:.2f}"
    )
    print(format_str)

# Example 2: Mixed linear and logarithmic grids
print("\n2. Mixed Linear and Logarithmic Grids:")
print("-" * 45)

# Linear grid parameter
temp = IndependentScalarParameter(
    value=300.0,
    is_sampled=True,
    range=(200.0, 400.0),
    grid_points=3,  # Linear: [200, 300, 400]
    grid_scale="linear",
)

# Logarithmic grid parameter
pressure = IndependentScalarParameter(
    value=1000,
    is_sampled=True,
    range=(100, 10000),
    grid_points=3,  # Logarithmic: [100, 1000, 10000]
    grid_scale="log",
    param_type="float",
)  # explicit type setting

bank2 = ParameterBank({"temp": temp, "pressure": pressure})
grid2 = bank2.compute_hypergrid()
print(f"Generated {len(grid2)} grid points")
for i, point in enumerate(grid2):
    temp_str = f"temp={point['temp']:.0f}"
    pressure_str = f"pressure={point['pressure']:.0f}"
    print(f"  Point {i}: {temp_str}, {pressure_str}")

# Example 3: Explicit point lists
print("\n3. Explicit Point Lists:")
print("-" * 25)

# Parameter with explicit points
concentration = IndependentScalarParameter(
    value=0.5,
    is_sampled=True,
    range=(0.0, 1.0),
    grid_points=[0.1, 0.5, 0.9],  # Explicit points
    param_type="float",
)

# Parameter with automatic generation
ph = IndependentScalarParameter(
    value=7.0,
    is_sampled=True,
    range=(6.0, 8.0),
    grid_points=2,  # Linear: [6.0, 8.0]
    param_type="float",
)

bank3 = ParameterBank({"concentration": concentration, "ph": ph})
grid3 = bank3.compute_hypergrid()
print(f"Generated {len(grid3)} grid points")
for i, point in enumerate(grid3):
    print(
        f"  Point {i}: concentration={point['concentration']:.1f}, "
        f"ph={point['ph']:.1f}"
    )

# Example 4: Grid with derived parameters
print("\n4. Grid with Derived Parameters:")
print("-" * 35)

# Independent parameters
length = IndependentScalarParameter(
    value=10.0, is_sampled=True, range=(5.0, 15.0), grid_points=3
)

width = IndependentScalarParameter(
    value=5.0, is_sampled=True, range=(2.0, 8.0), grid_points=2
)


# Derived parameter: area = length * width
def compute_area(params) -> float:
    return params["length"] * params["width"]


area = DerivedScalarParameter(compute_area)


# Derived parameter: perimeter = 2 * (length + width)
def compute_perimeter(params) -> float:
    return 2 * (params["length"] + params["width"])


perimeter = DerivedScalarParameter(compute_perimeter)

bank4 = ParameterBank(
    {"length": length, "width": width, "area": area, "perimeter": perimeter}
)

grid4 = bank4.compute_hypergrid()
print(f"Generated {len(grid4)} grid points")
for i, point in enumerate(grid4):
    format_str = (
        f"  Point {i}: length={point['length']:.1f}, "
        f"width={point['width']:.1f}, "
        f"area={point['area']:.1f}, perimeter={point['perimeter']:.1f}"
    )
    print(format_str)

# Example 5: Type support demonstration
print("\n5. Type Support Demonstration:")
print("-" * 35)

print(f"Mass type: {mass.param_type}")
print(f"Velocity type: {velocity.param_type}")
print(f"Temp type: {temp.param_type}")
print(f"Pressure type: {pressure.param_type}")
print(f"Length type: {length.param_type}")
print(f"Width type: {width.param_type}")
print(f"Area type: {area.param_type}")
print(f"Perimeter type: {perimeter.param_type}")

# Show type enforcement in grid points
print("\nGrid point types:")
first_point = grid[0]
print(f"  Mass: {type(first_point['mass'])} " f"(value: {first_point['mass']})")
print(
    f"  Velocity: {type(first_point['velocity'])} "
    f"(value: {first_point['velocity']})"
)
print(
    f"  Gravity: {type(first_point['gravity'])} " f"(value: {first_point['gravity']})"
)

# Auto-detection example
auto_int_param = IndependentScalarParameter(value=42, grid_points=2)
auto_float_param = IndependentScalarParameter(value=3.14, grid_points=2)
auto_bool_param = IndependentScalarParameter(value=True, grid_points=2)

print("\nAuto-detected grid parameter types:")
print(f"  Auto int: {auto_int_param.param_type} (grid: {auto_int_param.grid_points})")
print(
    f"  Auto float: {auto_float_param.param_type} (grid: {auto_float_param.grid_points})"
)
print(
    f"  Auto bool: {auto_bool_param.param_type} (grid: {auto_bool_param.grid_points})"
)

print("\n" + "=" * 60)
print("HyperGrid Integration Example Complete")
print("=" * 60)
