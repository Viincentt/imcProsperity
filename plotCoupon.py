import csv
import matplotlib.pyplot as plt
import numpy as np
from math import floor, ceil
import math
from math import log, sqrt, exp, erf
from scipy.optimize import newton


def cdf(x):
    # Constants in the rational approximation
    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429

    # Save the sign of x
    sign = np.sign(x)
    x = np.abs(x) / np.sqrt(2.0)

    # A&S formula 7.1.26
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)

    return 0.5 * (1.0 + sign * y)


def blackScholes_(S, K, T, r, sigma, option="call"):
    """
    S: underlying asset mid price
    K: strike price of the option
    T: time to expiration in years
    r: risk free interest rate
    sigma: volatility
    option: call or put
    """
    d1 = (np.log(max(S, K) / min(S, K)) + (r + (0.5 * (sigma**2))) * T) / (
        sigma * np.sqrt(T)
    )
    # d2 = d1 - (sigma * np.sqrt(T))
    d2 = (math.log(S / K) + (r - (0.5 * (sigma**2))) * T) / (sigma * math.sqrt(T))

    Nd1 = S * cdf(d1)
    Nd2 = K * np.exp(-r * T) * cdf(d2)
    return Nd1 - Nd2


def normal_cdf(x):
    """Cumulative distribution function for the standard normal distribution."""
    return (1 + erf(x / sqrt(2))) / 2


def blackScholes(S, K, T, r, sigma):
    """Calculate BS price of call/put"""
    d1 = (log(S / K) + (r + sigma**2 / 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    price = S * normal_cdf(d1) - K * exp(-r * T) * normal_cdf(d2)
    return int(round(price)), normal_cdf(d1)


def black_scholes_price(S, K, t, r, sigma, option_type="call"):
    d1 = (log(S / K) + (r + 0.5 * sigma * 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    price = S * normal_cdf(d1) - K * np.exp(-r * t) * normal_cdf(d2)
    return price


def implied_volatility(market_price, S, K, t, r, initial_guess=0.2):
    objective_function = (
        lambda sigma: black_scholes_price(S, K, t, r, sigma) - market_price
    )
    return newton(objective_function, initial_guess)


def main():
    coupon = []
    x = []
    mid_price = "mid_price"
    coconut = []
    for i in range(1, 4):
        file = "d4Data/prices_round_4_day_" + str(i) + ".csv"
        with open(file, mode="r") as f:
            tmp = csv.DictReader(f, delimiter=";")
            for row in tmp:
                if row["product"] == "COCONUT_COUPON":
                    coupon.append(float((row[mid_price])))
                else:
                    coconut.append(float(row[mid_price]))
                    # x.append(int(row["timestamp"]) / 100)
    deltaPrice = []
    bsPrice = []
    for i in range(1, len(coconut)):
        coco, coup = coconut[i], coupon[i]
        T = 246 / 252
        rate = 0
        volatility = 0.1609616171503603

        price, delta = blackScholes(coco, 10000, T, rate, volatility)
        divider = 1.8
        if price > coup:
            price -= abs(price - coup) / divider
        else:
            price += abs(price - coup) / divider
        # diff = coco - coconut[i - 1]
        # pred = coupon[i - 1] + delta * diff
        # deltaPrice.append(pred)
        bsPrice.append(price)

    # Plot the grid lines

    plt.plot(coupon, label="Coupon price")
    # plt.plot(deltaPrice, label="Delta")
    plt.plot(bsPrice, label="BlackScholes")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
