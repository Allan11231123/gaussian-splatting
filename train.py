#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os, sys
import torch
import cv2
import numpy as np
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim, depth_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, DrivingSceneParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def bise_load_weights(model: torch.nn.Module, weights_path: str):
    ckpt = torch.load(weights_path, map_location="cpu")
    # Some ckpt may be in 'state_dict', some may be flattened
    state = ckpt.get("state_dict", ckpt)
    # get rid of possible 'module.' prefix
    new_state = {}
    for k, v in state.items():
        nk = k.replace("module.", "")
        new_state[nk] = v
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print("[Warn] Missing keys:", missing[:10], "..." if len(missing) > 10 else "")
    if unexpected:
        print("[Warn] Unexpected keys:", unexpected[:10], "..." if len(unexpected) > 10 else "")
@torch.no_grad()
def infer_bisenetv2(bise_model, img_bgr, device="cpu"):
    width, height = img_bgr.shape[1], img_bgr.shape[0]
    new_height = height - height % 32
    img = cv2.resize(img_bgr, (width, new_height), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    x = torch.from_numpy(img).unsqueeze(0).to(device)
    out = bise_model(x)   # expected output: B, C=19, H, W (logits)
    if isinstance(out, (list, tuple)):
        out = out[0]  # some implementations return multi-scale, take main output
    out = F.interpolate(out, size=(img_bgr.shape[0], img_bgr.shape[1]), mode="bilinear", align_corners=False)  # back to original size
    pred = out.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
    # binary sky mask
    sky_mask = (pred == 10).astype(np.uint8)
    return sky_mask

def training(dataset, opt, pipe, drive_params, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    bise_model = None
    if drive_params.use_sky_mask:
        sys.path.append(os.path.abspath(drive_params.mask_model_path))
        try:
            from models import BiSeNetV2
        except Exception as e:
            raise RuntimeError(
                "Failed to load BiSeNetV2, please ensure you've passed the parent directory of the model."
            )
        if not os.path.isfile(drive_params.mask_weights_path):
            raise RuntimeError(f"BiSeNetV2 weights not found at {drive_params.mask_weights_path}")
        bise_model = BiSeNetV2(n_classes=19)
        bise_load_weights(bise_model, drive_params.mask_weights_path)
        bise_model = bise_model.to(lp.data_device)
        bise_model.eval()

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type) # inside the model we create the optimizer
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if drive_params.learn_bg:
        bg_param = torch.nn.Parameter(torch.tensor([
            drive_params.bg_init,
            drive_params.bg_init,
            drive_params.bg_init
        ], device="cuda"))
        optim_bg = torch.optim.Adam([bg_param], lr=0.01)
    else: 
        optim_bg = None
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        # class 'Camera' can be found in scene/cameras.py
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
                
        if drive_params.learn_bg and drive_params.use_sky_mask and bise_model != None:
            if iteration==first_iter:
                print("Using BiSeNetV2 sky segmentation model for background learning.")
            sky_mask = infer_bisenetv2(
                bise_model,
                cv2.cvtColor((viewpoint_cam.original_image.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
                device=dataset.data_device
            )
            sky_mask = torch.from_numpy(sky_mask).float().to(dataset.data_device)
            fg_mask = (1.0 - sky_mask).unsqueeze(0)
            sky_mask3 = sky_mask.unsqueeze(0)
            # broacast to [H,2W,3]
            B = bg_param.view(3,1,1).expand_as(image)
            # Mix-up: non-sky from the foreground, sky from the learnable background B
            _mixed_image = (image*fg_mask + B*sky_mask3)
            # L1 loss calculation
            Ll1 = (torch.abs(image - gt_image) * fg_mask).sum() / (fg_mask.sum()*3.0 + 1e-8)
            # SSIM loss on foreground only
            ssim_fg = ssim(image * fg_mask, gt_image * fg_mask)
            bg_loss = (torch.abs(B-gt_image) * sky_mask3).sum() / (sky_mask3.sum()*3.0 + 1e-8)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_fg) + drive_params.bg_lambda * bg_loss
        else:
            Ll1 = l1_loss(image, gt_image)
            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_value = ssim(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable and False: # Disabled depth regularization with standard depth map
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        # Custom depth regularization with no inverse depth map
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.custom_invdepthmap is not None and viewpoint_cam.depth_reliable:
            if drive_params.depth_loss == "l1":
                if iteration == first_iter:
                    print("Using custom depth map with L1 loss for depth regularization.")
                invDepth = render_pkg["depth"]
                custom_invdepth = viewpoint_cam.custom_invdepthmap.cuda()
                depth_mask = viewpoint_cam.depth_mask.cuda()
                # either 'invDepth' or 'custom_invdepth' may have 0 values, cannot simply take inverse value.
                # convert the 'invDepth' to depth, while not changing the size of tensor
                invDepth[invDepth!=0] = 1.0 / invDepth[invDepth!=0]
                custom_invdepth[custom_invdepth!=0] = 1.0 / custom_invdepth[custom_invdepth!=0]

                Ll1depth_pure = torch.abs((invDepth  - custom_invdepth) * depth_mask).mean()
                Ll1depth_custom = depth_l1_weight(iteration) * Ll1depth_pure 
                loss += Ll1depth_custom
                Ll1depth += Ll1depth_custom.item()
            elif drive_params.depth_loss == "huber":
                if iteration == first_iter:
                    print("Using Huber loss for custom depth regularization.")
                invDepth = render_pkg["depth"]
                custom_invdepth = viewpoint_cam.custom_invdepthmap.cuda()
                depth_mask = viewpoint_cam.depth_mask.cuda()
                # either 'invDepth' or 'custom_invdepth' may have 0 values, cannot simply take inverse value.
                # convert the 'invDepth' to depth, while not changing the size of tensor
                invDepth[invDepth!=0] = 1.0 / invDepth[invDepth!=0]
                custom_invdepth[custom_invdepth!=0] = 1.0 / custom_invdepth[custom_invdepth!=0]

                Ll1depth_pure = F.huber_loss(invDepth * depth_mask, custom_invdepth * depth_mask, reduction='mean')
                Ll1depth_custom = depth_l1_weight(iteration) * Ll1depth_pure 
                loss += Ll1depth_custom
                Ll1depth += Ll1depth_custom.item()
        

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    dp = DrivingSceneParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), dp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
