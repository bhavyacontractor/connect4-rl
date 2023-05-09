import numpy as np
import pygame
import sys
from pygame.locals import *
from helper import *

model = network_linear()
model.load_state_dict(torch.load("./model_final.pt", map_location=torch.device("cpu")))
b = board()


def create_board(rows, cols):
    board = np.zeros((rows, cols))
    return board


def drop_piece(board, row, col, piece):
    board[row][col] = piece


def is_valid_col(board, col):
    return board[5][col] == 0


def get_empty_row(board, col):
    for row in range(ROWS):
        if board[row][col] == 0:
            return row


def print_board(board):
    board = np.flip(board, 0)
    print(board)


def is_win(piece):

    # Horizontal
    for row in range(ROWS):
        for col in range(COLS-3):
            if board[row][col] == piece and board[row][col+1] == piece and board[row][col+2] == piece and board[row][col+3] == piece:
                return True

    # Vertical
    for row in range(ROWS-3):
        for col in range(COLS):
            if board[row][col] == piece and board[row+1][col] == piece and board[row+2][col] == piece and board[row+3][col] == piece:
                return True

    # pos diagonal
    for row in range(ROWS-3):
        for col in range(COLS-3):
            if board[row][col] == piece and board[row+1][col+1] == piece and board[row+2][col+2] == piece and board[row+3][col+3] == piece:
                return True

    # neg diagonal
    for row in range(3, ROWS):
        for col in range(COLS-3):
            if board[row][col] == piece and board[row-1][col+1] == piece and board[row-2][col+2] == piece and board[row-3][col+3] == piece:
                return True


def draw_board():
    for row in range(ROWS):
        for col in range(COLS):
            pygame.draw.rect(screen, BLUE, (col*SQUARE_SIZE, row *
                                            SQUARE_SIZE+SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
            pygame.draw.circle(screen, BLACK, (col*SQUARE_SIZE+SQUARE_SIZE //
                                               2, row*SQUARE_SIZE+SQUARE_SIZE+SQUARE_SIZE//2), RADIUS)

    for row in range(ROWS):
        for col in range(COLS):
            if board[row][col] == 1:
                pygame.draw.circle(screen, RED, (col*SQUARE_SIZE+SQUARE_SIZE //
                                                 2, HEIGHT - (row*SQUARE_SIZE+SQUARE_SIZE//2)), RADIUS)
            elif board[row][col] == 2:
                pygame.draw.circle(screen, YELLOW, (col*SQUARE_SIZE+SQUARE_SIZE //
                                                    2, HEIGHT - (row*SQUARE_SIZE+SQUARE_SIZE//2)), RADIUS)


ROWS = 6
COLS = 7
board = create_board(ROWS, COLS)


turn = 0
game_over = False

pygame.init()
SQUARE_SIZE = 100
WIDTH = COLS * SQUARE_SIZE
HEIGHT = (ROWS+1) * SQUARE_SIZE
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Connect Four')

BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

RADIUS = SQUARE_SIZE//2-5

game_font = pygame.font.SysFont('monospace', 75)


while not game_over:

    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEMOTION:
            x_pos = event.pos[0]
            if turn == 0:
                color = RED
            else:
                color = YELLOW
            pygame.draw.rect(screen, BLACK, (0, 0, WIDTH, SQUARE_SIZE))
            pygame.draw.circle(screen, color, (x_pos, 47), RADIUS)

        if event.type == MOUSEBUTTONDOWN:
            x_pos = event.pos[0]
            col = x_pos//SQUARE_SIZE

            if turn == 0:
                if is_valid_col(board, col):
                    row = get_empty_row(board, col)
                    drop_piece(board, row, col, 1)
                    b.move(1, col)

                    if is_win(1):
                        pygame.draw.rect(
                            screen, BLACK, (0, 0, WIDTH, SQUARE_SIZE))
                        won_text = game_font.render(
                            'You Won !', True, RED)
                        won_rect = won_text.get_rect(
                            center=(WIDTH//2, won_text.get_height()//2))
                        screen.blit(won_text, won_rect)
                        game_over = True

                    turn = 1
            else:
                if is_valid_col(board, col):
                    state = torch.Tensor(b.state)[None, :]
                    values = model(state).detach().numpy()

                    col = np.argmax(np.array(values))
                    while b.state[b.rows - 1][col] != 0:
                        values[col] = -1e9
                        col = np.argmax(np.array(values))
                    print(values)
                    
                    print("Model move: ", col)
                    b.move(2, col)

                    row = get_empty_row(board, col)
                    drop_piece(board, row, col, 2)

                    if is_win(2):
                        pygame.draw.rect(
                            screen, BLACK, (0, 0, WIDTH, SQUARE_SIZE))
                        won_text = game_font.render(
                            'AI Won !', True, YELLOW)
                        won_rect = won_text.get_rect(
                            center=(WIDTH//2, won_text.get_height()//2))
                        screen.blit(won_text, won_rect)
                        game_over = True

                    turn = 0

    draw_board()
    pygame.display.update()

    if game_over:
        pygame.time.delay(3000)