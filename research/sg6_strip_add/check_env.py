"""Read-only environment check; deliberately no einops/timm/CuPy dependency."""
import platform
import sys
import torch,numpy,cv2,scipy

def main():
    versions=dict(python=platform.python_version(),torch=str(torch.__version__),numpy=numpy.__version__,cv2=cv2.__version__,scipy=scipy.__version__,cuda=torch.version.cuda)
    print('Interpreter:',sys.executable,flush=True);print(versions,flush=True)
    expected=dict(python='3.11.8',torch='2.2.2+cu121',numpy='1.26.4',cv2='4.10.0',scipy='1.13.1',cuda='12.1')
    if versions!=expected:raise RuntimeError('Environment differs from fixed SG6; do not upgrade the existing environment')
    if not torch.cuda.is_available() or torch.cuda.get_device_name()!='NVIDIA GeForce RTX 4090':raise RuntimeError('Original RTX4090 CUDA environment required')
    print('Environment PASS. No new dependency installation required.',flush=True)

if __name__=='__main__':main()
