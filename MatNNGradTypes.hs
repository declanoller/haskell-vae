{-# LANGUAGE GeneralizedNewtypeDeriving #-}

module MatNNGradTypes
( Layer(..)
, NN(..)
, Grads(..)
, Batch(..)
, VAE(..)
, AdamOptim(..)
, NNAdamOptim(..)
, VAEAdamOptim(..)
, DataInfo(..)
, TrainInfo(..)
, TrainStats(..)
) where


import Numeric.LinearAlgebra
import Data.Typeable
import Control.DeepSeq


newtype Layer = Layer {getLayer :: (Matrix R)} deriving (Show, NFData)
newtype NN = WeightMatList {getNN :: [Layer]} deriving (Show, NFData)
newtype Grads = GradMatList {getGrads :: [Matrix R]} deriving (Show, NFData)
newtype AdamOptim = AdamOptimParams {getAdamOptim :: (Matrix R, Matrix R, Int, Double, Double)} deriving (Show, NFData)
newtype NNAdamOptim = NNAdamOptimParams {getNNAdamOptim :: [AdamOptim]} deriving (Show, NFData)
newtype VAEAdamOptim = VAEAdamOptimParams {getVAEAdamOptim :: (NNAdamOptim, NNAdamOptim, Double)} deriving (Show, NFData)
newtype Batch = Batch {getBatch :: Matrix R} deriving (Show, NFData)
newtype VAE = VAE {getVAE :: (NN, NN)} deriving (Show, NFData)

data DataInfo = DataInfo { n_input :: Int
                         , n_tot_samples :: Int
                         , prefix :: String
                         , data_dir :: String
                         } deriving (Show)



data TrainInfo = TrainInfo {  batch_size :: Int
                            , batches_per_epoch :: Int
                            , n_epochs :: Int
                            , lr :: Double
                            , beta_KL_max :: Double
                            , beta_KL_method :: String
                            } deriving (Show)


data TrainStats = TrainStats  { beta_KL :: [Double]
                              , losses_kl :: [Double]
                              , losses_recon :: [Double]
                              , losses_total :: [Double]
                              } deriving (Show)
