module GradUtils
( get_init_grads
, get_init_grads_vae
, add_grads
, mult_grads
, grads_momentum
, get_init_adam_optim
, step_vae_adam_optim
) where


import Numeric.LinearAlgebra
import MatNNGradTypes
import MatrixUtils



get_init_grads :: NN -> Grads
get_init_grads (WeightMatList nn) = grads_zeroed
  where mat_list = map (\(Layer x) -> x) nn
        g = GradMatList mat_list
        grads_zeroed = mult_grads 0 g


get_init_grads_vae :: VAE -> (Grads, Grads)
get_init_grads_vae (VAE (front_half, back_half)) = (front_half_zeroed, back_half_zeroed)
  where front_half_zeroed = get_init_grads front_half
        back_half_zeroed = get_init_grads back_half


get_init_adam_optim :: VAE -> Double -> VAEAdamOptim
get_init_adam_optim (VAE (front_half, back_half)) alpha = VAEAdamOptimParams (front_half_nn_optim, back_half_nn_optim, alpha)
  where GradMatList front_half_zeroed = get_init_grads front_half
        GradMatList back_half_zeroed = get_init_grads back_half
        front_half_nn_optim = NNAdamOptimParams $ map (\x -> AdamOptimParams (x, x, 1, beta_1, beta_2)) front_half_zeroed
        back_half_nn_optim = NNAdamOptimParams $ map (\x -> AdamOptimParams (x, x, 1, beta_1, beta_2)) back_half_zeroed
        beta_1 = 0.9
        beta_2 = 0.999



add_grads :: Grads -> Grads -> Grads
add_grads (GradMatList g1) (GradMatList g2) = g3
  where g3 = GradMatList $ zipWith (+) g1 g2

mult_grads :: Double -> Grads -> Grads
mult_grads x (GradMatList g) = g_out
  where g_out = GradMatList $ map (scale x) g


grads_momentum :: (Grads, Grads) -> (Grads, Grads) -> Double -> (Grads, Grads)
grads_momentum (last_vel_front, last_vel_back) (cur_grads_front, cur_grads_back) alpha = (new_vel_front, new_vel_back)
  where new_vel_front = add_grads (mult_grads momentum last_vel_front) (mult_grads (alpha) cur_grads_front)
        new_vel_back = add_grads (mult_grads momentum last_vel_back) (mult_grads (alpha) cur_grads_back)
        momentum = 0.9


step_adam_optim :: AdamOptim -> Matrix R -> (AdamOptim, Matrix R)
step_adam_optim adam_optim g = (new_adam_optim, adam_grads)
  where AdamOptimParams (m, v, t, beta_1, beta_2) = adam_optim
        new_m = (scale beta_1 m) + (scale (1 - beta_1) g)
        new_v = (scale beta_2 v) + (scale (1 - beta_2) (g**2))
        m_hat = scale ((1 - beta_1^t)**(-1.0)) new_m
        v_hat = scale ((1 - beta_2^t)**(-1.0)) new_v
        adam_grads = m_hat*((v_hat**0.5 + 10**(-6.0))**(-1.0))
        new_adam_optim = AdamOptimParams (new_m, new_v, t+1, beta_1, beta_2)


step_nn_adam_optim :: NNAdamOptim -> Grads -> (NNAdamOptim, Grads)
step_nn_adam_optim nn_adam_optim cur_grads = (NNAdamOptimParams new_layer_adam_optims, GradMatList new_layer_grads)
  where (NNAdamOptimParams layer_adam_optims) = nn_adam_optim
        (GradMatList layer_grads) = cur_grads
        (new_layer_adam_optims, new_layer_grads) = unzip $ zipWith step_adam_optim layer_adam_optims layer_grads



step_vae_adam_optim :: VAEAdamOptim -> (Grads, Grads) -> (VAEAdamOptim, (Grads, Grads))
step_vae_adam_optim vae_adam_optim (cur_grads_front, cur_grads_back) = (new_vae_adam_optim, new_grads_tup)
  where VAEAdamOptimParams (front_adam_optim, back_adam_optim, alpha) = vae_adam_optim
        (new_nn_front_optim, new_grads_front) = step_nn_adam_optim front_adam_optim cur_grads_front
        (new_nn_back_optim, new_grads_back) = step_nn_adam_optim back_adam_optim cur_grads_back
        new_vae_adam_optim = VAEAdamOptimParams (new_nn_front_optim, new_nn_back_optim, alpha)
        new_grads_tup = (new_grads_front, new_grads_back)


{-
-}
