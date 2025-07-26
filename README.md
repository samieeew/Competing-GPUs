# Competing-GPUs

# CUDA Tic-Tac-Toe / Connect-N

This is a GPU-accelerated implementation of Tic-Tac-Toe and Connect-N using **NVIDIA CUDA**.

- **Player 1 (X)** uses **random move logic** via `cuRAND` on the GPU.
- **Player 2 (O)** uses a **smart strategy** (win/block/center/corner).
- The board size is dynamic (NxN) and is passed as a command-line argument.

---

## 📦 Features

- Supports any board size `N x N` (minimum 3x3).
- Dynamic win condition: 
  - `3` in a row for Tic-Tac-Toe (`N=3`)
  - `4` in a row for Connect-4 (`N>=4`)
- Entire gameplay is handled on the GPU with device-side logic.
- Cross-platform (Linux/macOS/WSL with CUDA).

---

## 🧠 How It Works

| Player | Strategy | Execution                       |
|--------|----------|---------------------------------|
| X      | Random   | `curand` kernel                 |
| O      | Smart    | Win → Block → Center → Fallback |

---

## 🛠️ Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit installed (11.x or 12.x)
- `nvcc` compiler
- Make (GNU Make)

---

## 🚀 Build and Run

### Build
```bash
make build
