import torch
from .shape_svg.svg_extract_xy import svg_extract_xy
import numpy as np


def circle(num_pts: int) -> torch.Tensor:
    theta = torch.linspace(0, 2 * torch.pi, num_pts)

    # Compute x and y coordinates and concatenate to form a matrix
    X = torch.stack([torch.cos(theta), torch.sin(theta)], dim = 1)
    return X


def square(num_pts: int) -> torch.Tensor:
    # Generate points on the unit circle first and then map them to a square
    theta = torch.linspace(0, 2 * torch.pi, num_pts)

    x, y = torch.cos(theta), torch.sin(theta)
    s = torch.maximum(torch.abs(x), torch.abs(y))
    x_sq, y_sq = x/s, y/s

    X = torch.stack([x_sq, y_sq], dim = 1)
    return X


def square_from_t(t: torch.Tensor) -> torch.Tensor:
    # Generate theta values corresponding to t
    theta = 2 * torch.pi * t.reshape(-1)
    
    x, y = torch.cos(theta), torch.sin(theta)
    s = torch.maximum(torch.abs(x), torch.abs(y))
    x_sq, y_sq = x/s, y/s

    X = torch.stack([x_sq, y_sq], dim = 1)
    return X


def stanford_bunny(num_pts: int) -> torch.Tensor:
    X = svg_extract_xy('stanford_bunny.svg', num_pts = num_pts)
    return X


def airfoil(num_pts: int) -> torch.Tensor:
    X = svg_extract_xy('airfoil.svg', num_pts = num_pts)
    return X


def heart(num_pts: int) -> torch.Tensor:
    X = svg_extract_xy('heart.svg', num_pts = num_pts)
    return X


def hand(num_pts: int) -> torch.Tensor:
    X = svg_extract_xy('hand.svg', num_pts = num_pts)
    return X


def puzzle_piece(num_pts: int) -> torch.Tensor:
    X = svg_extract_xy('puzzle_piece.svg', num_pts = num_pts)
    return X


def airplane(num_pts: int) -> torch.Tensor:
    X = svg_extract_xy('airplane.svg', num_pts = num_pts)
    return X


def snowflake_fractal(num_pts: int) -> torch.Tensor:
    X = svg_extract_xy('snowflake_fractal.svg', num_pts = num_pts)
    return X


def star_fractal(num_pts: int) -> torch.Tensor:
    X = svg_extract_xy('star_fractal.svg', num_pts = num_pts)
    return X


def minkowski_fractal(num_pts: int) -> torch.Tensor:
    X = svg_extract_xy('minkowski_fractal.svg', num_pts = num_pts)
    return X