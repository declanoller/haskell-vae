module MatrixUtils
( layer_to_matrix
, batch_to_mat
, batchlist_to_matlist
, nonlinear_fn
, nonlinear_fn_bw
, softplus
, softplus_bw
, softplus_inv
, mse_loss
, mse_loss_bw
, sigmoid
, sigmoid_batch
, sigmoid_bw
, get_y_out_from_vae_output
, get_latent_from_vae_output
, get_mu_sigma_from_vae_output
, right_pad_ones
, drop_last_row
, get_latent_path
, get_grid_around_latent_pt
) where


import Numeric.LinearAlgebra
import MatNNGradTypes


--------- Conversions to/from Batch type

layer_to_matrix :: Layer -> Matrix R
layer_to_matrix (Layer mat) = mat

batch_to_mat :: Batch -> Matrix R
batch_to_mat (Batch b) = b

batchlist_to_matlist :: [Batch] -> [Matrix R]
batchlist_to_matlist batches = map (\(Batch x) -> x) batches


------------ Matrix functions

-- Generic nonlinear function, so if you want to switch out which function to
-- use, all other places call this one, so you just need to switch here.
nonlinear_fn :: Matrix R -> Matrix R
--nonlinear_fn m = mat_relu m
nonlinear_fn m = tanh m

-- corresponding backwards fn.
nonlinear_fn_bw :: Matrix R -> Matrix R
--nonlinear_fn_bw m = mat_relu_bw m
nonlinear_fn_bw m = 1 - (tanh m)**2




mat_relu :: Matrix R -> Matrix R
mat_relu w = cmap (max 0) w


mat_relu_bw :: Matrix R -> Matrix R
mat_relu_bw w = step w


softplus :: Matrix R -> Matrix R
softplus m = log (1 + exp m)


softplus_bw :: Matrix R -> Matrix R
softplus_bw m = (1 + exp (-m))**(-1)


softplus_inv :: Matrix R -> Matrix R
softplus_inv m = log ((exp m ) - 1)


sigmoid :: Matrix R -> Matrix R
sigmoid m = (1 + exp (-m))**(-1)


sigmoid_batch :: Batch -> Batch
sigmoid_batch (Batch m) = Batch $ sigmoid m


sigmoid_bw :: Matrix R -> Matrix R
sigmoid_bw m = (exp (-m))*(1 + exp (-m))**(-2)


right_pad_ones :: Matrix R -> Matrix R
right_pad_ones x = x ||| 1


drop_last_row :: Matrix R -> Matrix R
drop_last_row m = m ?? (DropLast 1, All)


mse_loss :: Batch -> Batch -> Double
mse_loss (Batch y_target) (Batch y) =  (sum $ toList $ flatten $ (y - y_target)**2)/len
  where len = fromIntegral $ (\(a, b) -> a*b) $ size y


mse_loss_bw :: Batch -> Batch -> Batch
mse_loss_bw (Batch y_target) (Batch y) = Batch ((y - y_target)*(2.0/len))
  where len = fromIntegral $ (\(a, b) -> a*b) $ size y




get_y_out_from_vae_output :: ([(Batch, Batch)], Batch, Batch, Matrix R, Matrix R, [(Batch, Batch)], Batch, Batch) -> Batch
get_y_out_from_vae_output vae_outputs = y_sig
  where (_, _, _, _, _, _, y_out, y_sig) = vae_outputs


get_latent_from_vae_output :: ([(Batch, Batch)], Batch, Batch, Matrix R, Matrix R, [(Batch, Batch)], Batch, Batch) -> Batch
get_latent_from_vae_output vae_outputs = latent
  where (_, latent, _, _, _, _, _, _) = vae_outputs


get_mu_sigma_from_vae_output :: ([(Batch, Batch)], Batch, Batch, Matrix R, Matrix R, [(Batch, Batch)], Batch, Batch) -> (Matrix R, Matrix R)
get_mu_sigma_from_vae_output vae_outputs = (mu, sigma)
  where (_, _, _, mu, sigma, _, _, _) = vae_outputs


------------------------- Producing points in the latent space

get_latent_path :: Matrix R -> Matrix R -> Batch
get_latent_path pt1 pt2 = Batch latent_path
  where latent_dim = snd $ size pt1
        diff = pt2 - pt1
        n_steps = 25
        latent_step = diff/(fromIntegral $ n_steps - 1)
        step_mat = repmat latent_step n_steps 1
        m = build (n_steps, latent_dim) (\i j -> i) :: Matrix R

        latent_path = pt1 + m*step_mat

-- Supply a Matrix R of a point in the latent space, as well as the "radius"
-- around it to get latent points from.
get_grid_around_latent_pt :: Matrix R -> Double -> Batch
get_grid_around_latent_pt pt rad = Batch grid
  where latent_dim = snd $ size pt
        n_grid = 15
        n_side = div n_grid 2
        unit_size = rad/(fromIntegral n_side)
        zeros_spacer = replicate (latent_dim - 2) 0.0
        unit_grid = fromLists [[fromIntegral z1, fromIntegral z2] ++ zeros_spacer | z1 <- [(-n_side)..n_side], z2 <- [(-n_side)..n_side]] :: Matrix R
        grid = pt + scale unit_size unit_grid




















--
