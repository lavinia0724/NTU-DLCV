import os
import sys

# Add the gaussian-splatting directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
gaussian_splatting_dir = os.path.join(current_dir, "gaussian-splatting")
sys.path.append(gaussian_splatting_dir)

import torch
from scene import Scene
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def render_set(model_path, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh, output_path):
    render_path = output_path
    # render_path = os.path.join(output_path, "renders")
    makedirs(render_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        try:
            # 修改這裡的 render 調用，移除 use_trained_exp 和 separate_sh 參數
            render_pkg = render(view, gaussians, pipeline, background)
            rendering = render_pkg["render"] if isinstance(render_pkg, dict) else render_pkg
            
            # Get original filename
            original_filename = view.image_name
            
            # Save rendered image to output path
            output_filepath = os.path.join(render_path, original_filename)
            torchvision.utils.save_image(rendering, output_filepath)
            
            # print(f"Successfully rendered and saved view {idx} to {output_filepath}")
            
        except Exception as e:
            print(f"Error rendering view {idx}: {str(e)}")
            import traceback
            traceback.print_exc()

def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool, output_path: str):
    with torch.no_grad():
        try:
            gaussians = GaussianModel(dataset.sh_degree)
            scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

            bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

            # Print camera information
            train_cameras = scene.getTrainCameras()
            test_cameras = scene.getTestCameras()
            
            render_set(dataset.model_path, scene.loaded_iter, test_cameras, 
                        gaussians, pipeline, background, dataset.train_test_exp, False, output_path)
            

        except Exception as e:
            print(f"Error during rendering: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_data_path = sys.argv[1]
    output_path = sys.argv[2]

    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    # Modify argv to include the model path
    original_argv = sys.argv
    setting_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output/setting1")
    sys.argv = [sys.argv[0]] + ["-m", setting_path, "-s", test_data_path]
    args = get_combined_args(parser)
    sys.argv = original_argv

    safe_state(args.quiet)

    print(f"Rendering images from {test_data_path}")
    print(f"Saving outputs to {output_path}")
    print(f"Using settings from {setting_path}")

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), 
                args.skip_train, args.skip_test, output_path)