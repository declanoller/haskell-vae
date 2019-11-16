module TrainUtils
( train_batch_vae
, train_epoch_vae
, train_n_epochs
) where


import Numeric.LinearAlgebra
import Data.List
import Data.List.Split
import Control.DeepSeq
import Control.Monad
import Data.Time

import MatNNGradTypes
import Utils
import MatrixUtils
import NNUtils
import GradUtils


-- Train for a single batch. Does a forward pass with forward_vae, backward
-- pass with backward_vae, steps weights with step_vae_adam_optim and step_weights_vae.
train_batch_vae :: VAE -> VAEAdamOptim -> TrainInfo -> DataInfo -> Int -> TrainStats -> Double -> IO (VAE, VAEAdamOptim, TrainStats)
train_batch_vae vae vae_optim train_info data_info iter train_stats cur_beta_KL = do

  rand_batch <- get_random_batch (data_dir data_info) (batch_size train_info) (n_tot_samples data_info)
  let x_batch = Batch $ fromLists rand_batch
  --let x_batch = Batch $ fromLists $ replicate (batch_size train_info) $ replicate (n_input data_info) 0

  vae_outputs <- forward_vae x_batch vae

  let y_out = get_y_out_from_vae_output vae_outputs
      (mu, sigma) = get_mu_sigma_from_vae_output vae_outputs
      recon_loss = mse_loss x_batch y_out
      kl_loss = kl_div mu sigma cur_beta_KL
      all_grads = backward_vae (x_batch, x_batch) vae vae_outputs cur_beta_KL
      (new_vae_optim, new_grads) = step_vae_adam_optim vae_optim all_grads
      VAEAdamOptimParams (_, _, alpha) = new_vae_optim
      new_vae = step_weights_vae vae new_grads alpha


  let new_train_stats = TrainStats  { beta_KL = []
                                    , losses_kl = (losses_kl train_stats) ++ [kl_loss]
                                    , losses_recon = (losses_recon train_stats) ++ [recon_loss]
                                    , losses_total = []
                                    }

  return (new_vae, new_vae_optim, new_train_stats)


-- Train for a single epoch. Uses foldM over train_batch_vae.
train_epoch_vae :: VAE -> VAEAdamOptim -> TrainInfo -> DataInfo -> TrainStats -> Int -> IO (VAE, VAEAdamOptim, TrainStats)
train_epoch_vae vae vae_optim train_info data_info train_stats epoch = do

  start_epoch <- getCurrentTime
  putStrLn ("\nEPOCH " ++ (show epoch))
  let n_batches = batches_per_epoch train_info
      cur_beta_KL = get_beta_KL epoch (n_epochs train_info) (beta_KL_max train_info)

  let train_fn = \(cur_vae, cur_optim, cur_train_stats) i -> train_batch_vae (force cur_vae) (force cur_optim) train_info data_info i cur_train_stats cur_beta_KL

  let init_epoch_train_stats = TrainStats { beta_KL = []
                                          , losses_kl = []
                                          , losses_recon = []
                                          , losses_total = []
                                          }

  (new_vae, new_vae_optim, epoch_train_stats) <- foldM train_fn (vae, vae_optim, init_epoch_train_stats) [1..n_batches]

  stop_epoch <- getCurrentTime
  putStrLn ("\tEpoch recon loss = " ++ roundToStr 4 ((sum (losses_recon epoch_train_stats))/(fromIntegral n_batches)))
  putStrLn ("\tEpoch kl loss = " ++ roundToStr 4 ((sum (losses_kl epoch_train_stats))/(fromIntegral n_batches)))
  putStrLn $ "\tEpoch train time = " ++ (show $ diffUTCTime stop_epoch start_epoch)

  -- Periodically save a backup of the current VAE.
  let tot_epochs = n_epochs train_info
  if rem epoch (max 1 (div tot_epochs 20)) == 0
    then do
      putStrLn ("\nEpoch " ++ (show epoch) ++ " / " ++ (show tot_epochs) ++ ", saving VAE...")
      let save_vae_dir = "saved_vaes/current_running_VAE/"
          vae_fname = (save_vae_dir ++ "current_running_VAE")
      putStrLn "Saved!\n"
      save_vae_to_file new_vae vae_fname
    else return ()



  let new_train_stats = TrainStats  { beta_KL = (beta_KL train_stats) ++ [cur_beta_KL]
                                    , losses_kl = (losses_kl train_stats) ++ (losses_kl epoch_train_stats)
                                    , losses_recon = (losses_recon train_stats) ++ (losses_recon epoch_train_stats)
                                    , losses_total = []
                                    }

  return (new_vae, new_vae_optim, new_train_stats)


-- Train for n_epochs. Uses foldM over train_epoch_vae.
-- Note: very important to use (force cur_vae) (force cur_optim) to prevent
-- a large thunk from building up.
train_n_epochs :: VAE -> VAEAdamOptim -> TrainInfo -> DataInfo -> IO (VAE, VAEAdamOptim, TrainStats)
train_n_epochs vae vae_optim train_info data_info = do

  start_train <- getCurrentTime

  let init_train_stats = TrainStats { beta_KL = []
                                    , losses_kl = []
                                    , losses_recon = []
                                    , losses_total = []
                                    }

  let train_fn = \(cur_vae, cur_optim, cur_train_stats) i -> train_epoch_vae (force cur_vae) (force cur_optim) train_info data_info cur_train_stats i

  (new_vae, new_vae_optim, new_train_stats) <- foldM train_fn (vae, vae_optim, init_train_stats) [1..(n_epochs train_info)]

  let losses_tot = zipWith (+) (losses_recon new_train_stats) (losses_kl new_train_stats)

  let final_train_stats = TrainStats  { beta_KL = (beta_KL new_train_stats)
                                      , losses_kl = (losses_kl new_train_stats)
                                      , losses_recon = (losses_recon new_train_stats)
                                      , losses_total = losses_tot
                                      }

  stop_train <- getCurrentTime
  putStrLn $ "\n\nTotal train time = " ++ (show $ diffUTCTime stop_train start_train) ++ "\n"


  return (new_vae, new_vae_optim, final_train_stats)
