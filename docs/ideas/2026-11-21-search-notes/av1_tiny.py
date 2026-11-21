"""Modern-baseline check for Sperling et al. (1985).

How many kbps does real-time AV1 (libaom, usage=realtime, no lookahead) need for
tiny gray signer video at 15 fps, the conditions the 1985 paper found intelligible?
Prints actual bitrate, PSNR and SSIM against the uncompressed downscaled reference.
"""

import re
import subprocess
from pathlib import Path

import imageio_ffmpeg

HERE = Path(__file__).parent
FF = imageio_ffmpeg.get_ffmpeg_exe()
SRC = HERE / "gsl_alphabet.webm"
CROP = "crop=720:1080:540:0"  # portrait crop around the signer, excludes the letter overlay
FPS = 15


def run(args):
    return subprocess.run([FF, "-hide_banner", *args], capture_output=True, text=True)


def count_frames(y4m: Path) -> int:
    return y4m.read_bytes().count(b"FRAME")


def metric(out: Path, ref: Path, name: str) -> str:
    graph = f"[0:v]format=gray[a];[1:v]format=gray[b];[a][b]{name}"
    r = run(["-i", str(out), "-i", str(ref), "-lavfi", graph, "-f", "null", "-"])
    pat = r"average:([0-9.]+|inf)" if name == "psnr" else r"All:([0-9.]+)"
    m = re.findall(pat, r.stderr)
    return m[-1] if m else "?"


def main():
    for w, h in [(64, 96), (32, 48), (16, 24)]:
        ref = HERE / f"ref_{w}x{h}.y4m"
        run(["-loglevel", "error", "-y", "-i", str(SRC), "-an", "-vf",
             f"{CROP},fps={FPS},scale={w}:{h}:flags=area,format=gray", "-pix_fmt", "gray", str(ref)])
        dur = count_frames(ref) / FPS
        for kbps in (2, 3, 5, 8):
            out = HERE / f"av1_{w}x{h}_{kbps}k.ivf"
            r = run(["-loglevel", "error", "-y", "-i", str(ref), "-c:v", "libaom-av1",
                     "-usage", "realtime", "-cpu-used", "8", "-lag-in-frames", "0",
                     "-b:v", f"{kbps}k", "-g", "150", "-pix_fmt", "gray", str(out)])
            if r.returncode != 0:
                print("encode failed", w, h, kbps, r.stderr[-300:])
                continue
            actual = out.stat().st_size * 8 / dur / 1000
            print(f"{w}x{h} target={kbps}k actual={actual:.2f}kbps "
                  f"psnr={metric(out, ref, 'psnr')}dB ssim={metric(out, ref, 'ssim')} "
                  f"frames={count_frames(ref)}")


if __name__ == "__main__":
    main()
