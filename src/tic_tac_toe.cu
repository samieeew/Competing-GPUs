#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <time.h>

#define ROWS 3
#define COLS 3

bool checkWinHost(const int *board, int player) {

    for (int row = 0; row < ROWS; row++) {
        if (board[row*COLS] == player &&
            board[row*COLS+1] == player &&
            board[row*COLS+2] == player) {
            return true;
        }
    }

    for (int col = 0; col < COLS; col++) {
        if (board[col] == player &&
            board[col+COLS] == player &&
            board[col+2*COLS] == player) {
            return true;
        }
    }

    if (board[0] == player && board[4] == player && board[8] == player) {
        return true;
    }
    if (board[2] == player && board[4] == player && board[6] == player) {
        return true;
    }

    return false;
}

__device__ bool checkWinDevice(const int *board, int player) {
    for (int row = 0; row < ROWS; row++) {
        if (board[row*COLS] == player &&
            board[row*COLS+1] == player &&
            board[row*COLS+2] == player) {
            return true;
        }
    }
    for (int col = 0; col < COLS; col++) {
        if (board[col] == player &&
            board[col+COLS] == player &&
            board[col+2*COLS] == player) {
            return true;
        }
    }
    if (board[0] == player && board[4] == player && board[8] == player) {
        return true;
    }
    if (board[2] == player && board[4] == player && board[6] == player) {
        return true;
    }
    return false;
}

__global__ void randomMove(int *board, int player, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        curandState_t state;
        curand_init(seed, idx, 0, &state);

        while (true) {
            int pos = curand(&state) % (ROWS * COLS);
            if (board[pos] == 0) {
                board[pos] = player;
                return;
            }
        }
    }
}

__global__ void smartMove(int *board, int player) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        int opponent = 3 - player;

        for (int i = 0; i < ROWS*COLS; i++) {
            if (board[i] == 0) {
                board[i] = player;
                if (checkWinDevice(board, player)) return;
                board[i] = 0;
            }
        }

        for (int i = 0; i < ROWS*COLS; i++) {
            if (board[i] == 0) {
                board[i] = opponent;
                if (checkWinDevice(board, opponent)) {
                    board[i] = player;
                    return;
                }
                board[i] = 0;
            }
        }

        if (board[4] == 0) {
            board[4] = player;
            return;
        }

        int corners[] = {0, 2, 6, 8};
        for (int i = 0; i < 4; i++) {
            if (board[corners[i]] == 0) {
                board[corners[i]] = player;
                return;
            }
        }

        for (int i = 0; i < ROWS*COLS; i++) {
            if (board[i] == 0) {
                board[i] = player;
                return;
            }
        }
    }
}

void printBoard(const int *board) {
    printf("\n");
    printf("    1   2   3\n");
    printf("  +---+---+---+\n");

    for (int row = 0; row < ROWS; row++) {
        printf("%d |", row+1);
        for (int col = 0; col < COLS; col++) {
            int val = board[row*COLS + col];
            printf(" %c |", val == 1 ? 'X' : val == 2 ? 'O' : ' ');
        }
        printf("\n");
        printf("  +---+---+---+\n");
    }
    printf("\n");
}

bool isBoardFull(const int *board) {
    for (int i = 0; i < ROWS*COLS; i++) {
        if (board[i] == 0) return false;
    }
    return true;
}

int main() {
    printf("Tic-Tac-Toe Game (CUDA)\n");
    printf("Player 1: X (Random moves)\n");
    printf("Player 2: O (Smart moves)\n\n");

    int *d_board;
    cudaMalloc(&d_board, ROWS * COLS * sizeof(int));
    cudaMemset(d_board, 0, ROWS * COLS * sizeof(int));

    int *h_board = (int *)malloc(ROWS * COLS * sizeof(int));

    int round = 1;
    int currentPlayer = 1; 

    while (round <= 9) {
        printf("Round %d - Player %d's turn (%c)\n",
               round, currentPlayer, currentPlayer == 1 ? 'X' : 'O');

        if (currentPlayer == 1) {
            randomMove<<<1, 1>>>(d_board, currentPlayer, time(NULL) + round);
        } else {
            smartMove<<<1, 1>>>(d_board, currentPlayer);
        }
        cudaDeviceSynchronize();

        cudaMemcpy(h_board, d_board, ROWS * COLS * sizeof(int), cudaMemcpyDeviceToHost);
        printBoard(h_board);

        if (checkWinHost(h_board, currentPlayer)) {
            printf("Player %d (%c) wins!\n", currentPlayer, currentPlayer == 1 ? 'X' : 'O');
            break;
        }

        if (isBoardFull(h_board)) {
            printf("Game ended in a draw!\n");
            break;
        }

        currentPlayer = 3 - currentPlayer; 
        round++;
    }

    cudaFree(d_board);
    free(h_board);

    printf("Game over.\n");
    return 0;
}
