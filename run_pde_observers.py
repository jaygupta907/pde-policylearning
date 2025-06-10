import wandb
import math
import numpy as np
import torch
import torch.nn as nn
from scipy import io
import torch.nn.functional as F
from timeit import default_timer
from libs.utilities3 import *
from libs.unet_models import *
from libs.models.fno_models import *
from libs.models.rno_models import RNO2dObserver
from libs.models.pino_models import PINObserverFullField, PolicyModel2D
from libs.models.transformer_models import *
from libs.envs.control_env import NSControlEnvMatlab
from libs.envs.ns_control_2d import NSControlEnv2D
from libs.visualization import *
from libs.pde_data_loader import *
from libs.arguments import *
from libs.metrics import *
from tqdm import tqdm
from torch.optim import Adam
from run_control import run_control
import os

torch.manual_seed(0)
np.random.seed(0)


def main(args, sample_data=False, train_shuffle=True):
    if not args.close_wandb:
        wandb.login(key='0d7ca5dd596d99f5ca1593f7aa97421bf844080a')
    if type(args.policy_name) == list:
        policy_list = args.policy_name[:]
        for policy_name in policy_list:  # compare different methods
            args.policy_name = policy_name
            # running another policy
            main(args, sample_data=sample_data, train_shuffle=train_shuffle)
        return
    if args.policy_name in ['unmanipulated', 'gt', 'rand']:
        args.control_only = True
    else:
        args.control_only = False
    if args.control_only:
        run_control(args, observer_model=None, wandb_exist=False)
        return
    args.using_transformer = 'Transformer' in args.model_name
    assert args.model_name in ['UNet', 'RNO2dObserver', 'PINObserverFullField', 'FNO2dObserverOld', 'FNO2dObserver', 'Transformer2D'],  "Model not supported!"
    
    ################################################################
    # create env when using physics-informed learning
    ################################################################
 
    if args.pde_loss_weight > 0:
        print("Initialization env for physics-informed learning ...")
        if args.env_name == 'NSControlEnv2D':
            env_class = NSControlEnv2D
            control_env = env_class(args, detect_plane=args.detect_plane, bc_type=args.bc_type)
        elif args.env_name == 'NSControlEnvMatlab':
            env_class = NSControlEnvMatlab
            control_env = env_class(args)
        else:
            raise RuntimeError("Not supported environment!")
        print("Environment is initialized!")
    
    ################################################################
    # make dataset
    ################################################################
    
    if args.random_split:
        idx = torch.randperm(args.ntrain + args.ntest)
    else:
        idx = torch.arange(args.ntrain + args.ntest)
    training_idx = idx[:args.ntrain]
    testing_idx = idx[-args.ntest:]
    if args.dataset_name == 'SequentialPDEDataset':
        dataset_fn = SequentialPDEDataset
    elif args.dataset_name == "FullFieldNSDataset":
        dataset_fn = FullFieldNSDataset
    else:
        dataset_fn = PDEDataset
    train_dataset = dataset_fn(args, args.DATA_FOLDER, training_idx, args.plane_indexs, args.downsample_rate, args.x_range, args.y_range, use_patch=args.use_patch, full_field=args.full_field)
    test_dataset = dataset_fn(args, args.DATA_FOLDER, testing_idx, args.plane_indexs, args.downsample_rate, args.x_range, args.y_range, use_patch=args.use_patch, full_field=args.full_field)
    if sample_data: 
        p_plane, v_plane = train_dataset[0]
        p_plane, v_plane = p_plane.cuda(), v_plane.cuda()
        v_plane = v_plane.squeeze()
        v_plane_decoded = train_dataset.v_norm.cuda_decode(v_plane)
        np.savetxt('outputs/v_plane_decoded.txt', v_plane_decoded.cpu().numpy())

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=train_shuffle, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    n_steps_per_epoch = math.ceil(len(train_loader.dataset) / args.batch_size)

    ################################################################
    # create observer model
    ################################################################
    
    if args.model_name == 'FNO2dObserverOld':
        observer_model = FNO2dObserverOld(args.modes, args.modes, args.width, use_v_plane=args.use_v_plane).cuda()
    elif args.model_name == 'FNO2dObserver':
        observer_model = FNO2dObserver(args.modes, args.modes, args.width, use_v_plane=args.use_v_plane).cuda()
    elif args.model_name == 'RNO2dObserver':
        observer_model = RNO2dObserver(args.modes, args.modes, args.width, recurrent_index=args.recurrent_index, layer_num=args.layer_num).cuda()
    elif args.model_name == 'PINObserverFullField':
        all_modes = [args.modes, args.modes, args.modes, args.modes]
        observer_model = PINObserverFullField(plane_num=len(args.plane_indexs), modes1=all_modes, modes2=all_modes, modes3=all_modes, fc_dim=128, layers=[64, 64, 64, 64, 64], 
                                     act='gelu', pad_ratio=0.0625, in_dim=1, ).cuda()
    elif args.model_name == 'UNet':
        observer_model = UNet(use_spectral_conv=args.use_spectral_conv).cuda()
    elif args.model_name == 'Transformer2D':
        observer_model = SimpleTransformer(**args.model).cuda()
    else:
        raise NotImplementedError("Model not supported!")

    ################################################################
    # create policy model
    ################################################################
    
    if args.policy_name == 'optimal-observer':
        policy_model = None
    elif args.policy_name in ['gt', 'rand', 'unmanipulated', 'rno', 'fno']:
        policy_model = None
    elif args.policy_name == 'optimal-policy-observer':
        all_modes = [args.modes, args.modes, args.modes, args.modes]
        policy_model = PolicyModel2D(modes1=all_modes, modes2=all_modes, modes3=all_modes, fc_dim=128, layers=[64, 64, 64, 64, 64], 
                                     act='gelu', pad_ratio=0.0625, in_dim=1, ).cuda()
    else:
        raise RuntimeError()

    ################################################################
    # loading pre-trained model if needed
    ################################################################
    
    if args.load_observer:
        observer_model_path = args.observer_model_path
        print(f"Loading pre-trained observer model from {observer_model_path}")
        if os.path.exists(observer_model_path):
            observer_model = torch.load(observer_model_path)
            print("Observer model loaded successfully!")
        else:
            print(f"Warning: Observer model path {observer_model_path} not found. Using newly initialized model.")
    
    if args.load_policy and policy_model is not None:
        policy_model_path = args.policy_model_path
        print(f"Loading pre-trained policy model from {policy_model_path}")
        if os.path.exists(policy_model_path):
            policy_model = torch.load(policy_model_path)
            print("Policy model loaded successfully!")
        else:
            print(f"Warning: Policy model path {policy_model_path} not found. Using newly initialized model.")

    if policy_model is None:
        print(f"Using '{args.policy_name}' without a learned policy model")
    
    ################################################################
    # training and validation
    ################################################################
    
    optimizer = Adam(observer_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    output_path = './outputs/'
    output_path += args.path_name
    output_path += '_observer.mat'
    myloss = LpLoss(size_average=False)

    if not args.close_wandb:
        wandb.init(
            project=args.project_name + "_" + args.path_name,
            name=args.exp_name,
            config={
                "task info": "p-plane-to-v",
                "model_name": args.model_name,
                "file_name": args.path_name,
                "has_prev_press": True,
                "patches": False,
                "permute": True,
                "DATA_FOLDER": args.DATA_FOLDER,
                "ntrain": args.ntrain,
                "ntest": args.ntest,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "epochs": args.epochs,
                "step_size": args.step_size,
                "gamma": args.gamma,
                "modes": args.modes,
                "width": args.width,
                "r": args.downsample_rate,
                "use_v_plane": args.use_v_plane,
                "use_patch": args.use_patch
                })

    best_loss = 10000000000000
    for ep in tqdm(range(args.epochs)):
        observer_model.train()
        t1 = default_timer()
        train_l2, train_num = 0, 0
        if args.dataset_name == "SequentialPDEDataset":
            for step, (p_plane, v_plane) in enumerate(tqdm(train_loader)):
                p_plane, v_plane = p_plane.cuda().float(), v_plane.cuda().float()
                if args.recurrent_model:
                    p_plane = p_plane.reshape(-1, args.model_timestep, args.x_range, args.y_range, 1)
                    v_plane = v_plane.reshape(-1, args.model_timestep, args.x_range, args.y_range, 1)
                    v_plane = v_plane[:, args.recurrent_index, :, :, :]  # select the predict element
                    args.batch_size = v_plane.shape[0]
                elif args.using_transformer:
                    p_plane = p_plane.reshape(-1, args.model_timestep, args.x_range, args.y_range, 1)
                else:
                    p_plane = p_plane.reshape(-1, args.x_range, args.y_range, 1)
                    v_plane = v_plane.reshape(-1, args.x_range, args.y_range, 1)
                train_num += len(v_plane)
                optimizer.zero_grad()
                pred_field_raw = observer_model(p_plane, v_plane)
                pred_field_raw = pred_field_raw.reshape(-1, args.x_range, args.y_range)
                out_decoded = train_dataset.v_norm.cuda_decode(pred_field_raw)
                v_plane = v_plane.squeeze()
                v_plane_decoded = train_dataset.v_norm.cuda_decode(v_plane)
                loss = myloss(out_decoded.view(args.batch_size, -1), v_plane_decoded.view(args.batch_size, -1))
                loss.backward()
                optimizer.step()
                train_l2 += loss.item()
                metrics = {"train/train_loss": loss.item(), 
                        "train/epoch": (step + 1 + (n_steps_per_epoch * ep)) / n_steps_per_epoch}
                if step + 1 < n_steps_per_epoch and not args.close_wandb:
                    # Log train metrics to wandb 
                    wandb.log(metrics)
        elif args.dataset_name == 'FullFieldNSDataset':
            for step, (v_plane, v_field, seq_u, seq_v, seq_w, seq_re, seq_dpdx) in enumerate(tqdm(train_loader)):
                v_plane, v_field, re = v_plane.cuda().float(), v_field.cuda().float(), seq_re.cuda().float()
                seq_u, seq_v, seq_w, seq_dpdx = seq_u.cuda().float(), seq_v.cuda().float(), seq_w.cuda().float(), seq_dpdx.cuda().float()
                v_plane = torch.einsum('btxy -> bxyt', v_plane).unsqueeze(-1)
                train_num += len(v_plane)
                optimizer.zero_grad()
                pred_field_raw = observer_model(v_plane, re)
                pred_field_raw = torch.einsum('bpxzt -> btpxz', pred_field_raw)
                pred_field_decoded = []
                for plane_index in range(len(train_dataset.plane_indexs)):
                    cur_pred = pred_field_raw[:, :, plane_index, :, :]
                    cur_pred = train_dataset.bound_v_norm.cuda_decode(cur_pred)
                    pred_field_decoded.append(cur_pred)
                pred_field_decoded = torch.stack(pred_field_decoded, dim=2)
                # v_field: [bs, feat dim, plane, x, y]
                target_field = []
                for plane_index in range(len(train_dataset.plane_indexs)):
                    target_one_plane = v_field[:, :, plane_index, :, :]
                    target_one_plane = train_dataset.v_field_norm.cuda_decode(target_one_plane)
                    target_field.append(target_one_plane)
                target = torch.stack(target_field, dim=2)
                data_loss = myloss(pred_field_decoded.reshape(args.batch_size, -1), target.reshape(args.batch_size, -1))
                pred_full_field_v = seq_v.clone()  # it's okay to not clone as well
                for idx, plane_index in enumerate(train_dataset.plane_indexs):
                    pred_full_field_v[:, :, :, plane_index, :] = pred_field_decoded[:, :, idx, :, :]
                pde_loss = 0
                if args.pde_loss_weight > 0:
                    for i in range(len(seq_u)):
                        cur_pde_loss = control_env.pde_loss(seq_u[i].squeeze(), seq_v.squeeze(), pred_full_field_v[i].squeeze(), seq_w[i].squeeze(), seq_dpdx[i].squeeze())
                        pde_loss += cur_pde_loss
                loss = data_loss + pde_loss * args.pde_loss_weight
                loss.backward()
                optimizer.step()
                train_l2 += loss.item()
                metrics = {"train/train_loss": loss.item(), 
                        "train/epoch": (step + 1 + (n_steps_per_epoch * ep)) / n_steps_per_epoch}
                if step + 1 < n_steps_per_epoch and not args.close_wandb:
                    # Log train metrics to wandb 
                    wandb.log(metrics)

        observer_model.eval()
        test_l2, test_num = 0.0, 0
        with torch.no_grad():
            if args.dataset_name == "SequentialPDEDataset":
                for p_plane, v_plane in test_loader:
                    p_plane, v_plane = p_plane.cuda().float(), v_plane.cuda().float()
                    if args.recurrent_model:
                        p_plane = p_plane.reshape(-1, args.model_timestep, args.x_range, args.y_range, 1)
                        v_plane = v_plane.reshape(-1, args.model_timestep, args.x_range, args.y_range, 1)
                        v_plane = v_plane[:, args.recurrent_index, :, :, :]
                        args.batch_size = v_plane.shape[0]
                    elif args.using_transformer:
                        p_plane = p_plane.reshape(-1, args.model_timestep, args.x_range, args.y_range, 1)
                    else:
                        p_plane = p_plane.reshape(-1, args.x_range, args.y_range, 1)
                        v_plane = v_plane.reshape(-1, args.x_range, args.y_range, 1)
                    test_num += len(v_plane)
                    out = observer_model(p_plane, v_plane)
                    out = out.reshape(-1, args.x_range, args.y_range)
                    if args.using_transformer:
                        p_plane = p_plane.reshape(-1, args.x_range, args.y_range, 1)
                    elif args.recurrent_model:
                        p_plane = p_plane[:, args.recurrent_index, :, :, :]
                    out_decoded = train_dataset.v_norm.cuda_decode(out)
                    v_plane = v_plane.squeeze()
                    p_plane_decoded = train_dataset.p_norm.cuda_decode(p_plane)
                    v_plane_decoded = train_dataset.v_norm.cuda_decode(v_plane)
                    test_loss = myloss(out_decoded.view(args.batch_size, -1), v_plane_decoded.view(args.batch_size, -1)).item()
                    test_l2 += test_loss
                    test_metrics = {"test/test_loss": test_loss / args.batch_size}
                    if not args.close_wandb:
                        wandb.log(test_metrics)
            elif args.dataset_name == 'FullFieldNSDataset':
                for step, (v_plane, v_field, seq_u, seq_v, seq_w, seq_re, seq_dpdx) in enumerate(tqdm(test_loader)):
                    v_plane, v_field, re = v_plane.cuda().float(), v_field.cuda().float(), seq_re.cuda().float()
                    v_plane = torch.einsum('btxy -> bxyt', v_plane).unsqueeze(-1)
                    test_num += len(v_plane)
                    pred_field_raw = observer_model(v_plane, re)
                    pred_field_raw = torch.einsum('bpxzt -> btpxz', pred_field_raw)
                    pred_field_decoded = []
                    for plane_index in range(len(train_dataset.plane_indexs)):
                        cur_pred = pred_field_raw[:, :, plane_index, :, :]
                        cur_pred = train_dataset.bound_v_norm.cuda_decode(cur_pred)
                        pred_field_decoded.append(cur_pred)
                    pred_field_decoded = torch.stack(pred_field_decoded, dim=2)
                    target_field = []
                    # v_field: [bs, feat dim, plane, x, y]
                    for plane_index in range(len(train_dataset.plane_indexs)):
                        target_one_plane = v_field[:, :, plane_index, :, :]
                        target_one_plane = train_dataset.v_field_norm.cuda_decode(target_one_plane)
                        target_field.append(target_one_plane)
                    target = torch.stack(target_field, dim=2)
                    target = train_dataset.v_field_norm.cuda_decode(v_field)
                    test_loss = myloss(pred_field_decoded.reshape(args.batch_size, -1), target.reshape(args.batch_size, -1)).item()
                    test_l2 += test_loss
                    test_metrics = {"test/test_loss": test_loss / args.batch_size}
                    if not args.close_wandb:
                        wandb.log(test_metrics)

        train_l2 /= train_num
        test_l2 /= test_num
        t2 = default_timer()
        # # save data into disk
        # data = {'gt': target.cpu().numpy(), 'pred': pred_field_decoded.cpu().numpy(),}
        # io.savemat(f'{ep}.mat', data)
        
        if test_l2 < best_loss:
            best_loss = test_l2
            if args.dataset_name == "SequentialPDEDataset":
                dat = {'x': p_plane_decoded.cpu().numpy(), 'pred': out_decoded.cpu().numpy(), 'y': v_plane_decoded.cpu().numpy(),}
                if not args.close_wandb:
                    vis_diagram(dat)
            model_save_p = f"./outputs/{args.path_name}_{args.exp_name}.pth"
            torch.save(observer_model, model_save_p)
            print(f"Best model saved at {model_save_p}!")
        print(f"epoch: {ep}, time passed: {t2-t1}, train loss: {train_l2}, test loss: {test_l2}, best loss: {best_loss}.")
        avg_metrics = {
            "train/avg_train_loss": train_l2,
            "test/avg_test_loss": test_l2,
            "test/best_loss": best_loss
            }
        
        if not args.close_wandb:
            wandb.log(avg_metrics)
    
    ################################################################
    # run control loop to evaluate trained model and exit program
    ################################################################
    
    if args.run_control:
        print("Running control")
        run_control(args, observer_model, policy_model=policy_model, train_dataset=train_dataset, wandb_exist=True)
    if not args.close_wandb and args.dataset_name == "SequentialPDEDataset":
        vis_diagram(dat)
        wandb.finish()


if __name__ == '__main__':
    args = parse_arguments()
    loaded_args = load_arguments_from_yaml(args.train_yaml)
    args = merge_args_with_yaml(args, loaded_args)
    if args.force_close_wandb:
        args.close_wandb = True
    if args.set_re > 0:
        args.Re = args.set_re
    if args.set_epoch > 0:
        args.epochs = args.set_epoch
    main(args)
