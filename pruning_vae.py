import os
import torch
import torchvision
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from model import *
from tqdm import *
from dataset import *
from classifier import *
from trainer import *
from models.model_base import ModelBase
from main_prune_non_imagenet import *
import copy
import torch.autograd as autograd


def grasp(net, ratio, train_loader, device,
          num_iters=1, reinit=True):
    eps = 1e-10
    keep_ratio = 1-ratio
    old_net = net
    
    net = copy.deepcopy(net)
    net.zero_grad()
    
    weights = []
    
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            if isinstance(layer, nn.Linear) and reinit:
                nn.init.xavier_normal(layer.weight)
            weights.append(layer.weight)
    
    # targets_one = [] => Generative, no target

    grad_w = None
    for w in weights:
        w.requires_grad_(True)
    
    print_once = False
    for ind in range(1, 3):
        for it in range(num_iters):
            print("(%d): Iterations %d/%d" % (ind, it, num_iters))
            for i, dataitem in tqdm(enumerate(train_loader, 1)):
                _,_,_,_,_,data = dataitem
                print(len(data))
                data.to(device)
                f_mean, f_logvar, f, z_post_mean, z_post_logvar, z, z_prior_mean, z_prior_logvar, recon_x = net.forward(data)
                loss, kld_f, kld_z = loss_fn(data, recon_x, f_mean, f_logvar, z_post_mean, z_post_logvar, z_prior_mean, z_prior_logvar)
                
                grad_w_p = autograd.grad(loss, weights, create_graph=True)
                if grad_w is None:
                    grad_w = list(grad_w_p)
                else:
                    for idx in range(len(grad_w)):
                        grad_w[idx] += grad_w_p[idx]
        
    grads = dict()
    old_modules = list(old_net.modules())
    for idx, layer in enumerate(net.modules()):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            grads[old_modules[idx]] = -layer.weight.data * layer.weight.grad  # -theta_q Hg
    # Gather all scores in a single vector and normalise
    all_scores = torch.cat([torch.flatten(x) for x in grads.values()])
    norm_factor = torch.abs(torch.sum(all_scores)) + eps
    print("** norm factor:", norm_factor)
    all_scores.div_(norm_factor)

    num_params_to_rm = int(len(all_scores) * (1-keep_ratio))
    threshold, _ = torch.topk(all_scores, num_params_to_rm, sorted=True)
    # import pdb; pdb.set_trace()
    acceptable_score = threshold[-1]
    print('** accept: ', acceptable_score)
    keep_masks = dict()
    for m, g in grads.items():
        keep_masks[m] = ((g / norm_factor) <= acceptable_score).float()

    print(torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks.values()])))

    return keep_masks
            
if __name__ == '__main__':
    config = init_config()
    logger, writer = init_logger(config)

    sprite, sprite_test = Sprites('dataset/lpc-dataset/train', 5814), Sprites('dataset/lpc-dataset/test', 522)
    batch_size = 5
    loader = torch.utils.data.DataLoader(sprite, batch_size, shuffle = True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Data Loading')

    vae = DisentangledVAE()
    test_f = torch.rand(1,256, device = device)
    test_f = test_f.unsqueeze(1).expand(1, 8, 256)
    # TODO: Prune the VAE
    mask = None
    mb = ModelBase(None, None, None, model = vae)
    mb.cuda()
    # ====================================== fetch configs ======================================
    ckpt_path = config.checkpoint_dir
    num_iterations = config.iterations
    target_ratio = config.target_ratio
    normalize = config.normalize
    # ====================================== fetch training schemes ======================================
    ratio = 1 - (1 - target_ratio) ** (1.0 / num_iterations)
    learning_rates = str_to_list(config.learning_rate, ',', float)
    weight_decays = str_to_list(config.weight_decay, ',', float)
    training_epochs = str_to_list(config.epoch, ',', int)
    logger.info('Normalize: %s, Total iteration: %d, Target ratio: %.2f, Iter ratio %.4f.' %
                (normalize, num_iterations, target_ratio, ratio))
    logger.info('Basic Settings: ')
    for idx in range(len(learning_rates)):
        logger.info('  %d: LR: %.5f, WD: %.5f, Epochs: %d' % (idx,
                                                              learning_rates[idx],
                                                              weight_decays[idx],
                                                              training_epochs[idx]))
    
    # ====================================== start pruning ======================================
    iteration = 0
    for _ in range(1):
        logger.info('** Target ratio: %.4f, iter ratio: %.4f, iteration: %d/%d.' % (target_ratio,
                                                                                    ratio,
                                                                                    1,
                                                                                    num_iterations))

        mb.model.apply(weights_init)
        print("=> Applying weight initialization(%s)." % config.get('init_method', 'kaiming'))
        print("Iteration of: %d/%d" % (iteration, num_iterations))
        masks = grasp(mb.model, ratio, loader, 'cuda')
        iteration = 0
        print('=> Using GraSP')
        # ========== register mask ==================
        mb.register_mask(masks)
        # ========== save pruned network ============
        logger.info('Saving..')
        state = {
            'net': mb.model,
            'acc': -1,
            'epoch': -1,
            'args': config,
            'mask': mb.masks,
            'ratio': mb.get_ratio_at_each_layer()
        }
        path = os.path.join(ckpt_path, 'prune_%s_%s%s_r%s_it%d.pth.tar' % (config.dataset,
                                                                           config.network,
                                                                           config.depth,
                                                                           config.target_ratio,
                                                                           iteration))
        torch.save(state, path)

        # ========== print pruning details ============
        logger.info('**[%d] Mask and training setting: ' % iteration)
        print_mask_information(mb, logger)
        logger.info('  LR: %.5f, WD: %.5f, Epochs: %d' %
                    (learning_rates[iteration], weight_decays[iteration], training_epochs[iteration]))

        # ========== finetuning =======================
        # train_once(mb=mb,
        #            net=mb.model,
        #            trainloader=trainloader,
        #            testloader=testloader,
        #            writer=writer,
        #            config=config,
        #            ckpt_path=ckpt_path,
        #            learning_rate=learning_rates[iteration],
        #            weight_decay=weight_decays[iteration],
        #            num_epochs=training_epochs[iteration],
        #            iteration=iteration,
        #            logger=logger)
    # TODO: Setup VAE Trainer
        trainer = Trainer(mb.model, sprite, sprite_test, loader, None, test_f, batch_size=25, epochs=1, learning_rate=0.002, device=device)

    # TODO: Compare timing