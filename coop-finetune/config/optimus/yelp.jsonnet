local lib = import '../utils.libsonnet';
local data_type = "yelp";
local latent_dim = 512;
local free_bit = 2.0;
# local num_steps = 129325;
# local checkout_step = 130000;
local num_steps = 258650;
local checkout_step = 258660;
local batch_size = 8;
local lr = 1e-6;
local pretrained_optimus = "megagonlabs/optimus-yelp";
local forced_beta_during_finetune = 0.1;

{
    "data_dir": "./data/%s" % data_type,
    "model": lib.Optimus(latent_dim, free_bit),
    "trainer": lib.VAETrainer(num_steps, checkout_step, batch_size, lr, pretrained_optimus, forced_beta_during_finetune)
}
