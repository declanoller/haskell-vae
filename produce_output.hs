
import Numeric.LinearAlgebra
import Data.List.Split
import System.Random
import Data.Typeable

import MatNNGradTypes
import Utils
import MatrixUtils
import NNUtils
import GradUtils
import TrainUtils


import System.Directory


main = do

  let train_info = TrainInfo  { batch_size = 32
                              , batches_per_epoch = 100
                              , n_epochs = 200
                              , lr = 0.001
                              , beta_KL_max = 0.00
                              , beta_KL_method = "constant"
                              }


      data_info_mnist = DataInfo  { n_input = 784
                                  , n_tot_samples = 60000
                                  , prefix = "mnist"
                                  , data_dir = "mnist_data/"
                                  }

      data_info_dogpeople = DataInfo  { n_input = 2500
                                      , n_tot_samples = 7000
                                      , prefix = "dogpeople"
                                      , data_dir = "dogpeople_data/"
                                      }

      --data_info = data_info_mnist
      data_info = data_info_dogpeople




  let base_save_vae_dir = "saved_vaes/"
      base_output_dir = "output/"

      --vae_name = "dogpeople_vae_0KL1500hidden_50latent"
      vae_name = "dogpeople_vae_0KL_train_with_friends2000hidden_60latent"
      --vae_name = "current_running_VAE"

      output_dir = base_output_dir ++ vae_name ++ "/"

      save_vae_dir = (base_save_vae_dir ++ vae_name)

      vae_fname = (save_vae_dir ++ "/" ++ vae_name)

  createDirectoryIfMissing False output_dir

  new_vae <- load_vae_from_file vae_fname
  {-
  -}



  putStrLn "\nSaving outputs..."
  {-

  (batch, labels) <- get_random_batch_with_labels (data_dir data_info) 3000 (n_tot_samples data_info)
  let x_batch = Batch $ fromLists batch
      latent_batch = forward_vae_to_latent_space new_vae x_batch
      recon_batch = forward_vae_no_sample_recon new_vae x_batch
      pt1 = (batch_to_mat latent_batch) ?? (Pos $ idxs [0], All)
      pt2 = (batch_to_mat latent_batch) ?? (Pos $ idxs [10], All)

  --pt1_data <- get_contents_as_mat_from_fname ("friends_faces_data/" ++ "face_david.txt")
  --let pt1 = batch_to_mat $ forward_vae_to_latent_space new_vae (Batch pt1_data)

  --let pt2 = (batch_to_mat latent_batch) ?? (Pos $ idxs [10], All)

      latent_path = get_latent_path pt1 pt2
      latent_path_recon = forward_vae_from_latent_space new_vae latent_path

      n_latent = snd $ size pt1

      center_pt = fromLists [replicate n_latent 0]
      latent_mat_dim_1_and_2 = toLists . tr $ (batch_to_mat latent_batch) ?? (All, Pos $ idxs [0, 1])
      rad = (0.25 *) $ maximum $ map (\a -> maximum a - minimum a) latent_mat_dim_1_and_2
      latent_grid = get_grid_around_latent_pt center_pt rad
      latent_grid_recon = forward_vae_from_latent_space new_vae latent_grid


  () <- save_batch_to_file latent_batch (output_dir ++ vae_name ++ "_latent_batch.txt")
  () <- save_batch_to_file x_batch (output_dir ++ vae_name ++ "_input_batch.txt")
  () <- save_batch_to_file recon_batch (output_dir ++ vae_name ++ "_output_batch.txt")
  () <- save_labels_to_file labels (output_dir ++ vae_name ++ "_labels.txt")

  () <- save_batch_to_file latent_path (output_dir ++ vae_name ++ "_latent_path.txt")
  () <- save_batch_to_file latent_path_recon (output_dir ++ vae_name ++ "_latent_path_recon.txt")

  () <- save_batch_to_file latent_grid (output_dir ++ vae_name ++ "_latent_grid.txt")
  () <- save_batch_to_file latent_grid_recon (output_dir ++ vae_name ++ "_latent_grid_recon.txt")

  -}




  let friend_names = ["ben", "bobby", "david", "liz", "max", "phil", "will"]
      friend_fnames = map (\x -> "friends_faces_data/face_" ++ x ++ ".txt") friend_names

  friend_data_lists <- mapM get_contents_from_fname friend_fnames
  let friend_batch = Batch $ fromLists friend_data_lists
      friend_latent_batch = forward_vae_to_latent_space new_vae friend_batch
      friend_recon_batch = forward_vae_no_sample_recon new_vae friend_batch

  () <- save_batch_to_file friend_batch (output_dir ++ vae_name ++ "_friend_input_batch.txt")
  () <- save_batch_to_file friend_latent_batch (output_dir ++ vae_name ++ "_friend_latent_batch.txt")
  () <- save_batch_to_file friend_recon_batch (output_dir ++ vae_name ++ "_friend_output_batch.txt")


  rand_dog_indices <- mapM (\x -> randomRIO (0, 64-1) :: IO Int) [1..(length friend_names)]
  let dog_fnames = map (\x -> "selected_dogs_data/" ++ (show x) ++ ".txt") rand_dog_indices
  dog_data <- mapM get_contents_as_mat_from_fname dog_fnames
  friend_data <- mapM get_contents_as_mat_from_fname friend_fnames

  let dog_data_batches = map Batch dog_data
      friend_data_batches = map Batch friend_data
      dog_latent_batches = batchlist_to_matlist $ map (forward_vae_to_latent_space new_vae) dog_data_batches
      friend_latent_batches = batchlist_to_matlist $ map (forward_vae_to_latent_space new_vae) friend_data_batches

      friend_dog_latent_interps = zipWith get_latent_path friend_latent_batches dog_latent_batches

      friend_dog_data_interps = map (forward_vae_from_latent_space new_vae) friend_dog_latent_interps

      out_friend_fnames = map (\x -> output_dir ++ vae_name ++ "_" ++ x ++ "_interp.txt") friend_names
  mapM (\(batch, fname) -> save_batch_to_file batch fname) $ zip friend_dog_data_interps out_friend_fnames

  {-
  -}


  return ()
