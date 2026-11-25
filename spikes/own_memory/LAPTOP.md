# Running the 1.7B test

The rules are in `PREREG-1.7B.md`. One command runs everything, in this order: download the data, check speed and memory, score the base model, run the learning-rate pilots, train both methods, and print a summary. Rerunning the same command resumes where it stopped.

## On a rented GPU (RunPod, works from a phone browser)

My estimate is 2 to 5 hours on an RTX 4090. At the roughly $0.34 an hour I found on Nov 22, that's about $1 to $2. About 10 minutes in, the run prints its own projection.

1. **Add credit.** Make an account at runpod.io and add $10 on the Billing page. That covers even the slowest case the rules allow (two 7-hour runs). Don't let the balance reach $0: RunPod then deletes pods that have no network volume, along with their files. RunPod also says prepaid cards should deposit at least $100 at a time.
2. **Deploy a pod.**
   - Pick an **RTX 4090** with the default PyTorch template. Any template with a terminal works.
   - Edit the template and set **Volume Disk to 40 GB**. Files in `/workspace` survive a pod restart; the container disk doesn't.
   - Choose **On-Demand**, not Spot, which can be interrupted.
   - If the page has a CUDA-version filter (under Additional filters), pick 12.6 or newer.
3. **Open a terminal.** On the pod: Connect, then Start Web Terminal, then Open Web Terminal.
4. **Start the test.** Paste this one line:
   ```bash
   cd /workspace && { [ -d Engramscope ] || git clone -b own-memory https://github.com/SebastianAbeSantoso/Engramscope.git; } && bash Engramscope/spikes/own_memory/rent.sh start
   ```
   It first prints the GPU and the PyTorch build it chose, then the progress, about once a minute. **You can close the tab at any time; the test keeps running.** To look again, open a web terminal and run:
   ```bash
   bash /workspace/Engramscope/spikes/own_memory/rent.sh watch
   ```
   About 10 to 15 minutes in, a line starting with `budget:` gives the projected hours. Paste it to me if you want the cost checked.
5. **When it says `Finished`,** run:
   ```bash
   bash /workspace/Engramscope/spikes/own_memory/rent.sh send
   ```
   It prints a code like `5028-hello-private-drink-13`. Paste the code into the chat and keep that terminal open until I confirm I have the files.
6. **Terminate the pod** once I confirm. Stopping it isn't enough, because a stopped pod still pays for its disk.

**If it stops** (an error or a pod restart), paste the step 4 line again. Finished steps are skipped, and training continues from its last checkpoint, at most 15 minutes back. If it stops with an error, send me a screenshot of the last lines from `watch`.

## On the Windows laptop (RTX 4060)

**Needs:**
- the NVIDIA driver (the script checks which CUDA version it supports);
- about 15 GB of free disk space;
- the charger plugged in.

The first run downloads about 6 GB: PyTorch with CUDA, plus the 3.4 GB model.

1. **Get the code.** In PowerShell:
   ```powershell
   git clone https://github.com/SebastianAbeSantoso/Engramscope.git   # skip if you already have it
   cd Engramscope
   git fetch origin
   git checkout own-memory
   git pull
   ```
2. **Run the check** (about 15 minutes, plus the downloads):
   ```powershell
   powershell -ExecutionPolicy Bypass -File spikes\own_memory\laptop.ps1 check
   ```
   It ends with a line like `budget: 20M tokens per method; ... memory run about X h, LoRA run about Y h`. **Paste the last few lines into the chat.**
3. **Start the full run** before bed, in the same window:
   ```powershell
   powershell -ExecutionPolicy Bypass -File spikes\own_memory\laptop.ps1
   ```
   The script stops Windows from sleeping while it runs, but the screen can still turn off. Close games and other GPU apps. It's also worth pausing Windows Update for a day so it doesn't reboot overnight.
4. **If it stops** (an error, a reboot or a closed window): run the step 3 command again. Finished steps are skipped, and training continues from its last checkpoint, at most 15 minutes back.
5. **When it finishes,** it prints a block between `===== paste this into the chat =====` lines. Paste it to me. Optionally, also push the result files:
   ```powershell
   git add spikes/own_memory/results/scale
   git commit -m "1.7B test results"
   git push
   ```

Progress is printed every minute and saved to `%USERPROFILE%\engramscope-ownmem\scale\log.txt`.

## Receiving the results (for the agent, in the Kiro sandbox)

`rent.sh send` uses runpodctl 3.1.1, so receive with the same version. If the sandbox was reset, download `runpodctl-linux-amd64` from the v3.1.1 release and check its sha256 against the one in `rent.sh`. The sandbox's home folder is read-only, so point `HOME` at a writable folder:

```bash
mkdir -p /projects/sandbox/_scratch/recv && cd /projects/sandbox/_scratch/recv
HOME=$PWD /projects/sandbox/_scratch/rpctl/runpodctl-linux-amd64 receive <code>
tar -xzf ownmem-1.7B-results.tar.gz
```

Then copy into `results/scale/`:
- `scale/check.json`, `base.json`, `pilot-choice.json`, `summary.json` and `log.txt`;
- `scale/runs/main-mem/result.json` and `scale/runs/main-lora/result.json`, as `main-mem.json` and `main-lora.json`;
- each `scale/runs/pilot-*/result.json`, as `pilot-*.json`.

Then update `RESULTS.md` and commit.
