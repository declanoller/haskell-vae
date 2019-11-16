import Numeric.LinearAlgebra
import System.Random

import MatNNGradTypes
import Utils
import MatrixUtils
import NNUtils
import GradUtils
import TrainUtils

import System.Directory

-- normally MNIST has 1875 batches of 32 per epoch. Here, we do
-- ~20 epochs of 100 batches of 32.

--there are 7k dogpeople images, so that would be 220 batches of 32
--so an equivalent would be ~2 epochs of 100 batches.

main = do

  let train_info = TrainInfo  { batch_size = 32
                              , batches_per_epoch = 100
                              , n_epochs = 100
                              , lr = 0.001
                              , beta_KL_max = 0.00
                              , beta_KL_method = "constant"
                              }


      data_info_mnist = DataInfo  { n_input = 784
                                  , n_tot_samples = 60000
                                  , prefix = "mnist"
                                  , data_dir = "data/mnist_data/"
                                  }

      data_info_dogpeople = DataInfo  { n_input = 2500
                                      , n_tot_samples = 7000
                                      , prefix = "dogpeople"
                                      , data_dir = "data/dogpeople_data/"
                                      }

      --data_info = data_info_mnist
      data_info = data_info_dogpeople


  ---------------- Filename, setup stuff

  let run_label = (prefix data_info) ++ "_vae_tiny_ex"
      n_hidden = 1000
      n_latent = 2
      vae_name = run_label ++ show n_hidden ++ "hidden_" ++ show n_latent ++ "latent"

      base_save_vae_dir = "saved_vaes/"
      base_output_dir = "output/"

      output_dir = base_output_dir ++ vae_name
      save_vae_dir = (base_save_vae_dir ++ vae_name)
      vae_fname = (save_vae_dir ++ vae_name)

  createDirectoryIfMissing False output_dir
  createDirectoryIfMissing False save_vae_dir


  ------------------------ Build VAE, set up optim, run

  vae <- build_vae [(n_input data_info), n_hidden, n_latent]
  let vae_adam_optim = get_init_adam_optim vae (lr train_info)

  (new_vae, new_vae_optim, new_train_stats) <- train_n_epochs vae vae_adam_optim train_info data_info


  ------------------------- Save losses and VAE to file

  putStrLn "\nSaving losses..."
  save_list_to_file (beta_KL new_train_stats) (output_dir ++ vae_name ++ "_beta_KL.txt")
  save_list_to_file (losses_recon new_train_stats) (output_dir ++ vae_name ++ "_losses_recon.txt")
  save_list_to_file (losses_kl new_train_stats) (output_dir ++ vae_name ++ "_losses_kl.txt")
  save_list_to_file (losses_total new_train_stats) (output_dir ++ vae_name ++ "_losses_total.txt")
  save_vae_to_file new_vae vae_fname
  print "\nDone!\n"

  return ()
