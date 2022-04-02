# Copyright (C) 2022 Matthew Marting
# SPDX-License-Identifier: GPL-3.0-or-later

import csv
from numpy import array
from numpy import column_stack
from numpy import dot
from numpy import eye
from numpy import ones
from numpy import zeros
from qpsolvers import solve_qp

with open("funds/20220330-funds.csv", newline="") as funds_file:
    reader = csv.reader(funds_file)
    funds = {row[0]: array(row[1:]).astype(float).reshape(3, 3) for row in reader}
portfolio_funds = {
    ticker_symbol: fund
    for ticker_symbol, fund in funds.items()
    if ticker_symbol in ("FLCOX", "FXAIX", "FSPGX", "FSMDX", "FSSNX")
}

ti_c_and_sp_us_w = array([08.47, 27.12, 17.43, 09.49, 05.37])
ti_c_and_sp_us_w /= ti_c_and_sp_us_w.sum()


def _portfolio(w):
    return {
        fund_name: 100 * weight
        #
        for fund_name, weight in zip(portfolio_funds.keys(), w)
    }


ti_c_and_sp_us_portfolio = _portfolio(ti_c_and_sp_us_w)


def style_box(portfolio):
    return sum(
        portfolio[ticker_symbol] * portfolio_funds[ticker_symbol]
        #
        for ticker_symbol in portfolio_funds.keys()
    )


ti_c_and_sp_us = style_box(ti_c_and_sp_us_portfolio)


def projt(a, b):
    a = a.flatten()
    b = b.flatten()
    return b / max(bi / ai for ai, bi in zip(a, b))


def projf(a, b):
    a = a.flatten()
    b = b.flatten()
    return array(tuple(min(ai, bi) for ai, bi in zip(a, b)))


def wt(target=None):
    if target is None:
        target = funds["VTSAX"]
    return array(
        tuple(
            projf(fund, fund).sum()
            / (
                projf(fund, fund).sum()
                + sum(
                    projf(fund2, fund).sum()
                    #
                    for ticker_symbol2, fund2 in portfolio_funds.items()
                    if ticker_symbol2 != ticker_symbol
                )
            )
            * projt(target, fund).sum()
            #
            for ticker_symbol, fund in portfolio_funds.items()
        )
    )


def portfolio(k=None, wt=None, target=None):
    if wt is not None and k is None:
        raise ValueError
    if k is None:
        k = 0.0
    if k < 0:
        raise ValueError
    if target is None:
        target = funds["VTSAX"]
    F = column_stack(
        tuple(
            (portfolio_funds[ticker_symbol] / portfolio_funds[ticker_symbol].sum()).flatten().T
            #
            for ticker_symbol in portfolio_funds.keys()
        )
    )
    Ft = (target / target.sum()).flatten().T
    if wt is None:
        wt = ones(F.shape[1]).T
    wt /= wt.sum()
    # minimize norm(F w - Ft) + norm(k (w - wt))
    # subject to ones(F.shape[1]) w = 1
    #            zeros(F.shape[1]) <= w
    # see <https://stackoverflow.com/a/29601451>
    # norm(F w - Ft)^2 + norm(k (w - wt))^2
    # = w' F' F w - 2 w' F' Ft + Ft' Ft + k^2 (w' w - 2 w' wt + wt' wt)
    # = w' (F' F + k^2 I) + w' (-2 F' Ft + -2 k^2 wt)
    # = 1/2 w' (2 (F' F + k^2 I)) w + (-2 (F' Ft + k^2 wt))' w
    w = solve_qp(
        P=2.0 * (dot(F.T, F) + k ** 2 * eye(F.shape[1])),
        q=-2.0 * (dot(F.T, Ft) + k ** 2 * wt),
        A=ones(F.shape[1]),
        b=1.0,
        lb=zeros(F.shape[1]),
    )
    return _portfolio(w)
