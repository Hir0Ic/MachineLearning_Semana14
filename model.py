"""Arquitectura de la Red Neuronal Recurrente (RNN) simple Many-to-One en PyTorch.

Implementa la Etapa 09 (sección 4.4 del informe):
- Entrada: tensor 3D (B, L, F) con ventana de L horas y F=8 características.
- Núcleo recurrente: 1 o 2 capas `nn.RNN` con activación `tanh`:
    h_t = tanh(W x_t + U h_{t-1} + b)
- Cabeza de salida lineal sobre el último estado oculto h_t (Many-to-One):
    y_hat(t+1) = V h_t + c
"""
from __future__ import annotations

from typing import Dict, List, Union

import pandas as pd
import torch
import torch.nn as nn

from config import DEFAULT_NUM_LAYERS, FEATURES
from metrics import rnn_param_count


class SimpleRNNForecaster(nn.Module):
    """RNN simple (Elman) de 1 o 2 capas recurrentes con cabeza lineal escalar."""

    def __init__(
        self,
        input_size: int = len(FEATURES),
        hidden_size: int = 32,
        num_layers: int = DEFAULT_NUM_LAYERS,
        dropout: float = 0.0,
    ):
        super().__init__()
        if num_layers not in (1, 2):
            raise ValueError("La arquitectura admite 1 o 2 capas recurrentes según la guía.")
        if hidden_size <= 0 or input_size <= 0:
            raise ValueError("input_size y hidden_size deben ser enteros positivos.")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = float(dropout) if num_layers > 1 else 0.0

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity="tanh",
            batch_first=True,
            dropout=self.dropout_rate,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propaga una ventana (B, L, F) y devuelve el pronóstico escalar (B, 1)."""
        if x.ndim != 3:
            raise ValueError(
                f"Se esperaba un tensor 3D (B, L, F), pero se recibió forma {tuple(x.shape)}."
            )
        # out: (B, L, H), h_n: (num_layers, B, H); h_0 se inicializa en ceros por defecto
        _, h_n = self.rnn(x)
        # Último estado oculto de la última capa recurrente -> (B, H)
        h_last = h_n[-1]
        # Cabeza lineal -> (B, 1)
        return self.head(h_last)


def count_parameters(model: nn.Module) -> int:
    """Cuenta exactamente los parámetros entrenables de un módulo PyTorch."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def parameter_breakdown(
    input_size: int = len(FEATURES),
    hidden_size: int = 32,
    num_layers: int = 1,
) -> List[Dict[str, Union[str, int]]]:
    """Desglosa los tensores de pesos y sesgos de `SimpleRNNForecaster`."""
    rows: List[Dict[str, Union[str, int]]] = []
    for layer_idx in range(num_layers):
        in_dim = input_size if layer_idx == 0 else hidden_size
        rows.extend(
            [
                {
                    "Parámetro": f"rnn.weight_ih_l{layer_idx} (W)",
                    "Forma": f"({hidden_size}, {in_dim})",
                    "Cantidad": hidden_size * in_dim,
                },
                {
                    "Parámetro": f"rnn.weight_hh_l{layer_idx} (U)",
                    "Forma": f"({hidden_size}, {hidden_size})",
                    "Cantidad": hidden_size * hidden_size,
                },
                {
                    "Parámetro": f"rnn.bias_ih_l{layer_idx} (b_ih)",
                    "Forma": f"({hidden_size},)",
                    "Cantidad": hidden_size,
                },
                {
                    "Parámetro": f"rnn.bias_hh_l{layer_idx} (b_hh)",
                    "Forma": f"({hidden_size},)",
                    "Cantidad": hidden_size,
                },
            ]
        )

    rows.extend(
        [
            {
                "Parámetro": "head.weight (V)",
                "Forma": f"(1, {hidden_size})",
                "Cantidad": hidden_size,
            },
            {
                "Parámetro": "head.bias (c)",
                "Forma": "(1,)",
                "Cantidad": 1,
            },
        ]
    )
    return rows


def architecture_table(
    input_size: int = len(FEATURES),
    seq_len: int = 24,
    hidden_size: int = 32,
    num_layers: int = 1,
    batch_size: Union[int, str] = "B",
) -> pd.DataFrame:
    """Genera la tabla concreta de capas (Tabla 7 del informe) con valores reales."""
    p_layer1 = hidden_size * input_size + hidden_size * hidden_size + 2 * hidden_size
    p_layer2 = hidden_size * hidden_size + hidden_size * hidden_size + 2 * hidden_size
    p_head = hidden_size + 1

    rows = [
        {
            "Capa": "Entrada",
            "Tipo": "Ventana temporal",
            "Configuración": f"L = {seq_len} h, F = {input_size} variables",
            "Forma de salida": f"({batch_size}, {seq_len}, {input_size})",
            "Parámetros": 0,
        },
        {
            "Capa": "Recurrente 1",
            "Tipo": "nn.RNN (tanh)",
            "Configuración": f"H = {hidden_size} unidades, batch_first=True",
            "Forma de salida": f"({batch_size}, {seq_len}, {hidden_size}) → h_t: ({batch_size}, {hidden_size})",
            "Parámetros": p_layer1,
        },
    ]

    if num_layers == 2:
        rows.append(
            {
                "Capa": "Recurrente 2",
                "Tipo": "nn.RNN (tanh)",
                "Configuración": f"H = {hidden_size} unidades (entrada H = {hidden_size})",
                "Forma de salida": f"({batch_size}, {seq_len}, {hidden_size}) → h_t: ({batch_size}, {hidden_size})",
                "Parámetros": p_layer2,
            }
        )

    rows.append(
        {
            "Capa": "Salida",
            "Tipo": "nn.Linear (cabeza lineal)",
            "Configuración": f"Entrada H = {hidden_size} → 1 neurona escalar",
            "Forma de salida": f"({batch_size}, 1)",
            "Parámetros": p_head,
        }
    )

    total = rnn_param_count(input_size, hidden_size, num_layers)
    rows.append(
        {
            "Capa": "TOTAL",
            "Tipo": f"RNN Many-to-One ({num_layers} capa{'s' if num_layers > 1 else ''})",
            "Configuración": f"F={input_size}, L={seq_len}, H={hidden_size}",
            "Forma de salida": f"({batch_size}, 1)",
            "Parámetros": total,
        }
    )
    return pd.DataFrame(rows)
