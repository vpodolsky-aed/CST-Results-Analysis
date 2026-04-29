import h5py
import numpy as np

filepath = r"C:\Users\vpodolsky_avalanche\Desktop\CST Local Files\Field dump in HDF5\Geometry1FIT67GHzE-field.h5"
output_path = r"C:\Users\vpodolsky_avalanche\Desktop\CST Local Files\efield_67GHz.vtr"

with h5py.File(filepath, "r") as f:
    x = f["Mesh line x"][:]   # shape (32,)
    y = f["Mesh line y"][:]   # shape (24,)
    z = f["Mesh line z"][:]   # shape (14,)
    
    E = f["E-Field"][:]       # shape (14, 24, 32), compound dtype

# Extract complex components from compound dtype
Ex = E["x"]["re"] + 1j * E["x"]["im"]  # shape (14, 24, 32)
Ey = E["y"]["re"] + 1j * E["y"]["im"]
Ez = E["z"]["re"] + 1j * E["z"]["im"]

# Magnitude (absolute value of complex field)
E_mag = np.sqrt(np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2).astype(np.float32)

# NOTE: CST stores as (nz, ny, nx) - VTK expects (nx, ny, nz) Fortran order
# Transpose so axes align correctly in ParaView
Ex_r = np.real(Ex).astype(np.float32).T  # now (32, 24, 14)
Ey_r = np.real(Ey).astype(np.float32).T
Ez_r = np.real(Ez).astype(np.float32).T
E_mag_t = E_mag.T

nx, ny, nz = len(x), len(y), len(z)

with open(output_path, "w") as out:
    out.write('<?xml version="1.0"?>\n')
    out.write('<VTKFile type="RectilinearGrid" version="0.1" byte_order="LittleEndian">\n')
    out.write(f'  <RectilinearGrid WholeExtent="0 {nx-1} 0 {ny-1} 0 {nz-1}">\n')
    out.write(f'    <Piece Extent="0 {nx-1} 0 {ny-1} 0 {nz-1}">\n')
    out.write('      <PointData>\n')
    
    # E magnitude
    out.write('        <DataArray type="Float32" Name="E_magnitude" format="ascii">\n')
    out.write(' '.join(map(str, E_mag_t.flatten(order='F'))) + '\n')
    out.write('        </DataArray>\n')
    
    # E vector (real part)
    out.write('        <DataArray type="Float32" Name="E_vector" NumberOfComponents="3" format="ascii">\n')
    vec = np.stack([Ex_r, Ey_r, Ez_r], axis=-1).reshape(-1, 3, order='F')
    out.write(' '.join(f'{v[0]} {v[1]} {v[2]}' for v in vec) + '\n')
    out.write('        </DataArray>\n')
    
    out.write('      </PointData>\n')
    out.write('      <Coordinates>\n')
    for label, coords in [("x", x), ("y", y), ("z", z)]:
        out.write(f'        <DataArray type="Float32" Name="{label}" format="ascii">\n')
        out.write(' '.join(map(str, coords.astype(np.float32))) + '\n')
        out.write('        </DataArray>\n')
    out.write('      </Coordinates>\n')
    out.write('    </Piece>\n')
    out.write('  </RectilinearGrid>\n')
    out.write('</VTKFile>\n')

print(f"Done! Written to: {output_path}")
print(f"Grid: {nx} x {ny} x {nz} points")
print(f"|E| range: {E_mag.min():.4f} to {E_mag.max():.4f} V/m")