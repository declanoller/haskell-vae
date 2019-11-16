module NNUtils
( forward
, forward_vae
, forward_vae_no_sample
, backward_vae
, step_weights
, step_weights_vae
, print_vae
, build_nn
, build_vae
, forward_vae_from_latent_space
, forward_vae_to_latent_space
, forward_vae_no_sample_recon
, kl_div
) where

import Data.Time

import Prelude hiding ((<>))
import Numeric.LinearAlgebra
import MatNNGradTypes
import MatrixUtils

-- Returns both linear and nonlinear outputs, which are both needed at various points.
forward_unit :: (Batch, Layer) -> (Batch, Batch)
forward_unit (Batch x, Layer w) = (Batch y, Batch a)
  where y = (x ||| 1) <> w
        a = nonlinear_fn y

-- Returns list of tuples of (y, a), where y = x <> w, a = sigma(y)
forward :: Batch -> NN -> [(Batch, Batch)]
forward x (WeightMatList w_list) = scanl fold_fn (x, x) w_list
  where fold_fn = \(_, a) w -> forward_unit (a, w)


-- Assumes the VAE layer outputs a list of [mu_i, sigma_i].
-- Adds a small 10**(-6) to prevent it from going too low (common tactic).
forward_vae_unit :: Batch -> IO (Batch, Batch, Matrix R, Matrix R)
forward_vae_unit (Batch g) = do
  let (batch_size, n_g) = size g
      n_sample = div n_g 2

  gauss_sample <- randn batch_size n_sample

  let mu = g ?? (All, Range 0 1 (n_sample - 1))
      sigma = 10**(-6.0) + softplus (g ?? (All, Range n_sample 1 (2*n_sample - 1)))
      sampled_mat = mu + gauss_sample*sigma
  return (Batch sampled_mat, Batch gauss_sample, mu, sigma)

bw_vae_unit :: Batch -> Matrix R -> Matrix R
bw_vae_unit (Batch gauss_sample) sigma = dx_dg
  where dx_dg = konst 1 (size gauss_sample) ||| (gauss_sample*(softplus_bw (softplus_inv (sigma - 10**(-6.0)))))


kl_div :: Matrix R -> Matrix R -> Double -> Double
kl_div mu sigma beta_KL = (sumElements kl)
  where kl = scale beta_KL (-0.5)*(1.0 + log (sigma**2) - mu**2 - sigma**2)
        --n_elements a = fromIntegral $ (fst a)*(snd a) -- /(n_elements $ size kl)

-- This is dKL/dg, which we want to minimize. So we keep the term as it
-- is. The negative that applies to all terms is in the (-beta_KL).
kl_div_bw :: Matrix R -> Matrix R -> Double -> Matrix R
kl_div_bw mu sigma beta_KL = scale (-beta_KL) kl_bw
  where kl_bw = (-mu) ||| ((1/sigma - sigma)*(softplus_bw (softplus_inv (sigma - 10**(-6.0)))))

-- Forward pass of the VAE.
forward_vae :: Batch -> VAE -> IO ([(Batch, Batch)], Batch, Batch, Matrix R, Matrix R, [(Batch, Batch)], Batch, Batch)
forward_vae x (VAE (nn_front, nn_back)) = do

  let outputs_front = forward x nn_front
      y_out_front = fst $ last outputs_front -- using fst so it takes y rather than a

  (input_back, gauss_sample, mu, sigma) <- forward_vae_unit y_out_front

  let outputs_back = forward input_back nn_back
      y_out = fst $ last outputs_back
      y_sig = sigmoid_batch y_out
      out_tup = (outputs_front, input_back, gauss_sample, mu, sigma, outputs_back, y_out, y_sig)

  return out_tup

-- Does a forward pass, but doesn't sample: just uses z = mu, to avoid adding
-- noise when we just want to see the best reconstructions possible.
forward_vae_no_sample :: Batch -> VAE -> ([(Batch, Batch)], Batch, Batch, Matrix R, Matrix R, [(Batch, Batch)], Batch, Batch)
forward_vae_no_sample x (VAE (nn_front, nn_back)) = output_tuple

  where outputs_front = forward x nn_front
        y_out_front = fst $ last outputs_front -- using fst so it takes y rather than a

        Batch g = y_out_front
        (batch_size, n_g) = size g
        n_sample = div n_g 2
        mu = g ?? (All, Range 0 1 (n_sample - 1))
        gauss_sample = Batch (konst 0 (size mu))
        sigma = konst 0 (size mu)
        input_back = Batch mu

        outputs_back = forward input_back nn_back
        y_out = fst $ last outputs_back
        y_sig = sigmoid_batch y_out
        output_tuple = (outputs_front, input_back, gauss_sample, mu, sigma, outputs_back, y_out, y_sig)


forward_vae_no_sample_recon :: VAE -> Batch -> Batch
forward_vae_no_sample_recon vae x = y_sig
  where (_, _, _, _, _, _, y_out, y_sig) = forward_vae_no_sample x vae

forward_vae_from_latent_space :: VAE -> Batch -> Batch
forward_vae_from_latent_space (VAE (nn_front, nn_back)) x = sigmoid_batch $ fst $ last ys_as
  where ys_as = forward x nn_back

forward_vae_to_latent_space :: VAE -> Batch -> Batch
forward_vae_to_latent_space (VAE (nn_front, nn_back)) x = input_back
  where outputs_front = forward x nn_front
        y_out_front = fst $ last outputs_front -- using fst so it takes y rather than a
        Batch g = y_out_front
        (batch_size, n_g) = size g
        n_sample = div n_g 2
        mu = g ?? (All, Range 0 1 (n_sample - 1))
        input_back = Batch mu


-- All grads here are positive, i.e., L should be POSITIVE if you want to MINIMIZE
-- it.
backward_vae :: (Batch, Batch) -> VAE -> ([(Batch, Batch)], Batch, Batch, Matrix R, Matrix R, [(Batch, Batch)], Batch, Batch) -> Double -> (Grads, Grads)
backward_vae ((Batch x), y_target) (VAE (front_half, back_half)) (outputs_front, back_in, gauss_sample, mu, sigma, outputs_back, y_out, y_sig) beta_KL = (GradMatList grads_front, GradMatList grads_back)
  where (WeightMatList front_weights) = front_half

        (y_front, a_front) = unzip outputs_front
        (y_back, a_back) = unzip outputs_back

        Batch y_back_mat = y_out
        dy_sig_dy_back = sigmoid_bw y_back_mat
        Batch dLdy_sig = mse_loss_bw y_target y_sig
        dLdy = dLdy_sig*dy_sig_dy_back

        WeightMatList w_front = front_half
        WeightMatList w_back = back_half

        first_back_w = tr $ drop_last_row $ layer_to_matrix $ head w_back
        new_first_back_w = repmat first_back_w 1 2

        dydw_front = map (tr . right_pad_ones . batch_to_mat) (init a_front)
        dydw_all = map (tr . right_pad_ones . batch_to_mat) $ (init a_front) ++ (init a_back)

        dyda_front = (map (tr . drop_last_row . layer_to_matrix) $ tail w_front)
        dyda_all = dyda_front ++ [new_first_back_w] ++ (map (tr . drop_last_row . layer_to_matrix) $ tail w_back)

        gauss_sample_bridge = bw_vae_unit gauss_sample sigma

        dady_front = (map (nonlinear_fn_bw . batch_to_mat) $ tail $ init y_front)
        dady_all = dady_front ++ [gauss_sample_bridge] ++ (map (nonlinear_fn_bw . batch_to_mat) $ tail $ init y_back)


        dyda_dady_tups = zip dyda_all dady_all
        grad_products = scanr (\(dyda, dady) prev -> (prev<>dyda)*dady) dLdy dyda_dady_tups
        all_grads = zipWith ( <> ) dydw_all grad_products

        dKL_dg = kl_div_bw mu sigma beta_KL

        -- This should have the length of only the grads_front
        dyda_dady_tups_kl = zip dyda_front dady_front
        kl_grad_products = scanr (\(dyda, dady) prev -> (prev<>dyda)*dady) dKL_dg dyda_dady_tups_kl
        kl_grads = zipWith ( <> ) dydw_front kl_grad_products

        (recon_grads_front, grads_back) = splitAt (length front_weights) all_grads
        grads_front = zipWith (+) recon_grads_front kl_grads



step_weights :: NN -> Grads -> Double -> NN
step_weights (WeightMatList w_list) (GradMatList grad_list) alpha = WeightMatList w_list_stepped
  where w_list_stepped = zipWith (\(Layer w) w_grad -> Layer $ (w - (scale alpha w_grad))) w_list grad_list

step_weights_vae :: VAE -> (Grads, Grads) -> Double -> VAE
step_weights_vae (VAE (front_half, back_half)) (grads_front, grads_back) alpha = (VAE (front_stepped, back_stepped))
  where front_stepped = step_weights front_half grads_front alpha
        back_stepped = step_weights back_half grads_back alpha


print_vae :: VAE -> IO ()
print_vae (VAE (nn_1, nn_2)) = do
  putStrLn "\nVAE:"
  let (WeightMatList wlist_1) = nn_1
      sizes = map size $ map layer_to_matrix wlist_1
  print sizes

  let (WeightMatList wlist_2) = nn_2
      sizes = map size $ map layer_to_matrix wlist_2
  print sizes
  return ()



-- Use fan in of each layer to determine std.
-- pytorch uses: uniform(-sqrt(k), sqrt(k)), where k = 1/fan_in
build_nn :: [Int] -> IO NN
build_nn weight_dims = do
  let weight_and_bias_dims = map (+ 1) weight_dims
      input_output_size_tups = zip (init weight_and_bias_dims) (tail weight_dims)
      uniform_bds = map (\x -> (fromIntegral x)**(-0.5)) (init weight_and_bias_dims) -- 1/sqrt(fan_in)

  all_weight_mats <- mapM (\(n_in, n_out) -> (rand n_in n_out)) input_output_size_tups
  putStrLn "\nNN layer sizes:"
  print input_output_size_tups

  let all_weights_scaled = zipWith (\unif_bds w -> 2*unif_bds*w - unif_bds) uniform_bds all_weight_mats
      all_layers = map Layer all_weights_scaled
  return (WeightMatList all_layers)

{-
Pass two lists of ints. The first is the shape of the matrices for the front half,
the second is the shape of matrices for the back half.
-}
build_vae :: [Int] -> IO VAE
build_vae vae_dims = do
  let latent_dim = last vae_dims
      front_half_dims = (init vae_dims) ++ [2*latent_dim]
      back_half_dims = reverse vae_dims

  nn_front <- build_nn front_half_dims
  nn_back <- build_nn back_half_dims
  let vae = VAE (nn_front, nn_back)
  putStrLn "\n\n"
  return vae
