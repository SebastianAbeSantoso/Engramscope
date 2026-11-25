# Running the 1.7B test

The rules are in `PREREG-1.7B.md`. One command runs everything, in this order: download the data, check speed and memory, score the base model, run the learning-rate pilots, train both methods, and print a summary. Rerunning the same command resumes where it stopped.

## On a rented GPU (Vast.ai, works from a phone browser)

My estimate is 1 to 4 hours on an A100. At about $0.53 an hour, that's roughly $1 to $2. About 10 to 15 minutes in, the run prints its own projection.

1. **Add credit.** Sign up at cloud.vast.ai and verify your email address; Vast won't rent you a machine before that. Then go to Billing, Add Credit, and add $5 (the minimum). If the balance reaches $0, Vast stops the instance, and its files are deleted later unless you add credit.
2. **Choose the template.** On the Templates page, find Vast's recommended **PyTorch** template and tap its ▶ (play) button. That opens the search page with the template selected.
3. **Set the disk to 50 GB** with the Disk Space slider on the search page, before renting. The default is 10 GB, which is too small, and the size can't be changed later.
4. **Pick a machine.**
   - Filter for **A100**. SXM or PCIe, 40 or 80 GB: all are fine.
   - Keep the rental type on **On-Demand**, not Interruptible. An interruptible machine can be taken away in the middle of the run.
   - On the offer card, check that **Max CUDA** is 12.6 or higher and the download speed is at least 100 Mbps. The run downloads about 8 GB.
   - Tap **Rent**. The price on the card already includes the disk.
5. **Open a terminal.** In the Instances tab, wait until the instance's blue button says **Open**. Loading isn't billed and can take a few minutes. Tap Open, then launch **Jupyter Terminal** from the Instance Portal. If the browser warns about the connection or a certificate, go back, wait a minute and tap Open again. The warning goes away once the secure link (ending in `trycloudflare.com`) is ready.
6. **Start the test.** Paste this one line and press Enter:
   ```bash
   mkdir -p /workspace && cd /workspace && { [ -d Engramscope ] || git clone -b own-memory https://github.com/SebastianAbeSantoso/Engramscope.git; } && git -C Engramscope pull -q --ff-only && bash Engramscope/spikes/own_memory/rent.sh start
   ```
   If the terminal won't take a paste, try the clipboard button on your phone's keyboard. Or launch **Jupyter** from the Instance Portal instead, make a new notebook, paste the same line with a `!` in front, and tap ▶. The `watch` and `send` lines below also work there with a `!` in front.

   It first prints the GPU and the PyTorch build it chose, then the progress, about once a minute. **You can close the tab at any time; the test keeps running.** To look again, open a terminal and run:
   ```bash
   bash /workspace/Engramscope/spikes/own_memory/rent.sh watch
   ```
   About 10 to 15 minutes in, a line starting with `budget:` gives the projected hours for the pilots, the memory run and the LoRA run. If they add up to more than 7 hours, add another $5 of credit. Paste the line to me if you want it checked.
7. **When it says `Finished`,** run:
   ```bash
   bash /workspace/Engramscope/spikes/own_memory/rent.sh send
   ```
   It prints a code like `5028-hello-private-drink-13`. Paste the code into the chat and keep that terminal open until I confirm I have the files. If sending fails, take a screenshot of the block between the `=====` lines near the end of the output instead.
8. **Destroy the instance** (the trash-can button on its card) once I confirm. Stopping it isn't enough, because a stopped instance still pays for its disk, and a running one costs about $0.53 an hour even when the test is done.

**If it stops** (an error or a restart), paste the step 6 line again. Finished steps are skipped, and training continues from its last checkpoint, at most 15 minutes back. If the instance itself was stopped, for example because the credit ran out, add credit and press the ▶ button on its card first. If it stops with an error, send me a screenshot of the last lines from `watch`.

### On RunPod instead

The same steps 6 and 7 work. The differences:
- Add $10 of credit. RunPod says prepaid cards should deposit at least $100. Don't let the balance reach $0: RunPod then deletes pods that have no network volume, along with their files.
- Deploy an **On-Demand** pod (A100 or RTX 4090) with the PyTorch template, and edit the template to set **Volume Disk to 40 GB**. Files in `/workspace` survive a pod restart; the rest of the disk doesn't.
- Open the terminal with Connect, then Start Web Terminal, then Open Web Terminal.
- At the end, **Terminate** the pod. Stopping it isn't enough.

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
