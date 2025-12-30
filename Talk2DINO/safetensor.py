from safetensors import safe_open
from safetensors.torch import save_file


if __name__ == "__main__":
    safetensor_path = "/home/ilearn/.cache/huggingface/hub/models--lorebianchi98--Talk2DINO-ViTB/snapshots/d120439255ae423ad0a3f4a13896bb74ae48792f/model.safetensors" # 此处替换为需要重新保存的文件位置

    fname, ext = safetensor_path.split("/")[-1].split(".")
    # ext = 'safetensors' # 扩展名
    # fname = 'model' # 文件名

    tensors = dict()
    with safe_open(safetensor_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    save_file(tensors, f"/home/ilearn/.cache/huggingface/hub/models--lorebianchi98--Talk2DINO-ViTB/snapshots/d120439255ae423ad0a3f4a13896bb74ae48792f/{fname}-with-format.{ext}", metadata={"format": "pt"})
