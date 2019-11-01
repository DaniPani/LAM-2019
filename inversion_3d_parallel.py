import hazel

# WITHOUT RANDOMIZATION
iterator = hazel.Iterator(use_mpi=True)
mod = hazel.Model('configurations/3d.ini', working_mode='inversion', verbose=3, rank=iterator.get_rank())
iterator.use_model(model=mod)
iterator.run_all_pixels()