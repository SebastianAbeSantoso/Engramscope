# Running the 1.7B test

The rules are in `PREREG-1.7B.md`. One command runs everything, in this order: download the data, check speed and memory, score the base model, run the learning-rate pilots, train both methods, and print a summary. Rerunning the same command resumes where it stopped.

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

## On a rented GPU (faster)

Any Linux machine with an NVIDIA GPU of 12 GB or more works, for example an RTX 4090 on RunPod or Vast.ai with a PyTorch template. In its terminal:

```bash
git clone https://github.com/SebastianAbeSantoso/Engramscope.git && cd Engramscope && git checkout own-memory
bash spikes/own_memory/rent.sh check
bash spikes/own_memory/rent.sh
```

Copy the printed summary and the `spikes/own_memory/results/scale` folder off the machine before you stop it.
