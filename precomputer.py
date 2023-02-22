import numpy as np
import cmbbest as best
import pandas as pd


p_max = 30
Nks = [10, 50, 200]

for Nk in Nks:
    basis = best.Basis("Legendre", mode_p_max=30, polarisation_on=True, k_grid_size=Nk, use_tetraquad=False)

    i1, i2, i3 = basis.tetrapyd_indices
    k1, k2, k3 = basis.tetrapyd_grid
    weights = basis.tetrapyd_grid_weights
    df = pd.DataFrame({"i1": i1, "i2": i2, "i3": i3, "k1": k1, "k2": k2, "k3": k3,
                            "weight": weights})
    df.to_csv(f"data/uniform_tetrapyd_cython_Nk_{Nk}.csv", float_format="%.18e")

    print(basis.data_path)
    np.save(f"data/mode_bispectra_covarinace_p_{p_max}_Nk_{Nk}.npy", basis.mode_bispectra_covariance)

    '''
    shapes = ["local", "equilateral", "orthogonal"]
    n_s, A_s = basis.parameter_n_scalar, basis.parameter_A_scalar
    print(n_s, A_s)

    #models = [best.Model(shape) for shape in shapes]
    models = [best.Model(shape, parameter_n_scalar=n_s, parameter_A_scalar=A_s) for shape in shapes]

    df = basis.constrain_models(models)

    df.to_csv(f"data/trio_constraints_p_{p_max}_Nk_{Nk}.csv", float_format="%.18e")
    '''
