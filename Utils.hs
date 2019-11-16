module Utils
( roundToStr
, print_grads
, save_batch_to_file
, save_labels_to_file
, save_list_to_file
, save_vae_to_file
, load_vae_from_file
, get_random_batch
, get_random_batch_with_labels
, get_beta_KL
, get_contents_as_mat_from_fname
, get_contents_from_fname
) where

import Text.Printf
import Data.Array.IO
import Data.List
import Data.List.Split
import Control.Monad
import Data.Random
import System.Random


import Numeric.LinearAlgebra
import Data.Typeable
import Control.DeepSeq

import System.IO.Unsafe
import System.IO


import MatNNGradTypes


------------------------- Getting batches of data from file

shuffle_list :: [a] -> IO [a]
shuffle_list xs = do
        ar <- newArray n xs
        forM [1..n] $ \i -> do
            j <- randomRIO (i,n)
            vi <- readArray ar i
            vj <- readArray ar j
            writeArray ar j vi
            return vj
  where
    n = length xs
    newArray :: Int -> [a] -> IO (IOArray Int a)
    newArray n xs =  newListArray (1,n) xs


get_contents_from_fname_ind :: String -> Int -> IO [Double]
get_contents_from_fname_ind data_dir fname_ind = do

  --let fname = "mnist_data/" ++ (show fname_ind) ++ ".txt"
  let fname = data_dir ++ (show fname_ind) ++ ".txt"
  let toDouble x = read x / 256
      extractXs  = map toDouble . tail . splitOn ","

  f_h <- openFile fname ReadMode
  line <- hGetLine f_h
  hClose f_h

  let data_list = extractXs line
  return data_list


get_contents_from_fname_ind_with_labels :: String -> Int -> IO ([Double], Int)
get_contents_from_fname_ind_with_labels data_dir fname_ind = do

  --let fname = "mnist_data/" ++ (show fname_ind) ++ ".txt"
  let fname = data_dir ++ (show fname_ind) ++ ".txt"
  let toDouble x = read x / 256
      extractXs  = map toDouble . tail . splitOn ","
      extract_label  = read . head . splitOn ","

  f_h <- openFile fname ReadMode
  line <- hGetLine f_h
  hClose f_h

  let data_list = extractXs line
      label = extract_label line
  return (data_list, label)


get_random_batch :: String -> Int -> Int -> IO [[Double]]
get_random_batch data_dir batch_size n_tot_samples = do
  rand_list <- mapM (\x -> randomRIO (0, n_tot_samples-1)) [1..batch_size]
  rand_batch <- mapM (get_contents_from_fname_ind data_dir) rand_list
  return rand_batch


get_random_batch_with_labels :: String -> Int -> Int -> IO ([[Double]], [Int])
get_random_batch_with_labels data_dir batch_size n_tot_samples = do
  rand_list <- mapM (\x -> randomRIO (0, n_tot_samples-1)) [1..batch_size]
  rand_batch_label_list <- mapM (get_contents_from_fname_ind_with_labels data_dir) rand_list
  let (rand_batch, labels) = unzip rand_batch_label_list
  return (rand_batch, labels)


get_contents_from_fname :: String -> IO [Double]
get_contents_from_fname fname = do
  let toDouble x = read x / 256
      extractXs  = map toDouble . tail . splitOn ","

  f_h <- openFile fname ReadMode
  line <- hGetLine f_h
  hClose f_h

  let data_list = extractXs line
  return data_list


get_contents_as_mat_from_fname :: String -> IO (Matrix R)
get_contents_as_mat_from_fname fname = do
  dat <- get_contents_from_fname fname
  let m = fromLists [dat]
  return m

--------------------- Saving data to file

list_to_csv_string :: [Int] -> String
list_to_csv_string my_list = intercalate "," $ map show my_list


matrix_to_string :: Matrix R -> String
matrix_to_string mat = unlines $ map (intercalate "," . map (roundToStr 3)) $ toLists mat


save_batch_to_file :: Batch -> String -> IO ()
save_batch_to_file (Batch b) fname = do
  writeFile fname $ matrix_to_string b


save_labels_to_file :: [Int] -> String -> IO ()
save_labels_to_file labels fname = do
  writeFile fname $ list_to_csv_string labels


write_matrix_to_file :: Matrix R -> String -> IO ()
write_matrix_to_file m fname = do
  writeFile fname $ matrix_to_string m


read_matrix_from_file :: String -> IO (Matrix R)
read_matrix_from_file fname = do
  let extractXs  = map read . splitOn ","
  num_lists <- map extractXs . lines <$> readFile fname
  let mat = fromLists num_lists
  return mat


save_list_to_file :: [Double] -> String -> IO ()
save_list_to_file l fname = do
  let str_list = intercalate "," $ map (roundToStr 5) l
  writeFile fname $ str_list
  return ()


--------------------- Saving and loading NNs and VAE to and from file

save_vae_to_file :: VAE -> String -> IO ()
save_vae_to_file (VAE (front_half, back_half)) fname_base = do
  putStrLn "\nSaving VAE to file..."
  () <- save_nn_to_file front_half (fname_base ++ "_front_nn")
  () <- save_nn_to_file back_half (fname_base ++ "_back_nn")
  putStrLn "Done!\n"
  return ()


save_nn_to_file :: NN -> String -> IO ()
save_nn_to_file (WeightMatList nn) fname_base = do
  let fnames = [fname_base ++ "_layer_" ++ (show i) ++ ".txt" | i <- [0..(length nn - 1)]]
  writeFile (fname_base ++ "_info.txt") (show $ length nn)
  _ <- mapM (\(a, b) -> save_layer_to_file a b) $ zip nn fnames
      --_ = zipWith save_layer_to_file nn fnames
  return ()


save_layer_to_file :: Layer -> String -> IO ()
save_layer_to_file (Layer m) fname = do
  write_matrix_to_file m fname
  return ()


load_layer_frome_file :: String -> IO Layer
load_layer_frome_file fname = do
  mat <- read_matrix_from_file fname
  putStrLn ("Reading in mat of size " ++ (show $ size mat))
  return (Layer mat)


load_vae_from_file :: String -> IO VAE
load_vae_from_file fname_base = do
  putStrLn "\nLoading VAE from file..."
  nn_front <- load_nn_from_file (fname_base ++ "_front_nn")
  nn_back <- load_nn_from_file (fname_base ++ "_back_nn")
  let vae = VAE (nn_front, nn_back)
  putStrLn "Done!\n"
  return vae


load_nn_from_file :: String -> IO NN
load_nn_from_file fname_base = do

  let info_fname = (fname_base ++ "_info.txt")
  nn_info <- (map read . lines <$> readFile info_fname)
  let n_layers = head nn_info
  let fnames = [fname_base ++ "_layer_" ++ (show i) ++ ".txt" | i <- [0..(n_layers - 1)]]
  layer_list <- mapM load_layer_frome_file fnames
  let nn = WeightMatList layer_list
  return nn


get_beta_KL :: Int -> Int -> Double -> Double
get_beta_KL cur_epoch n_epochs beta_KL_max = beta_KL
  where beta_KL = beta_KL_max -- const version

{-
  | cur_epoch < div n_epochs 2 = 0
  | otherwise = beta_KL_max*(fromIntegral $ cur_epoch - div n_epochs 2)/(fromIntegral $ div n_epochs 2)
  -}




------------------------------ Misc



roundToStr :: (PrintfArg a, Floating a) => Int -> a -> String
roundToStr = printf "%0.*f"


print_grads :: (Grads, Grads) -> IO ()
print_grads (g_1, g_2) = do
  putStrLn "\nGrads:"
  let (GradMatList wlist_1) = g_1
      sizes = map size wlist_1
  print sizes

  let (GradMatList wlist_2) = g_2
      sizes = map size wlist_2
  print sizes
  return ()






--
