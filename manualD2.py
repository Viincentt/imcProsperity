currencies = ["pizza", "wasabi", "snowball", "shells"]
rates = {
    "pizza": [1, 0.48, 1.52, 0.71],
    "wasabi": [2.05, 1, 3.26, 1.56],
    "snowball": [0.64, 0.3, 1, 0.46],
    "shells": [1.41, 0.61, 2.08, 1],
}


def rec(cur: float, l: list[str], size):
    if len(l) == size:
        return cur, l
    res, resL = 0, []
    for currency in currencies:
        tmp, tmpL = rec(
            cur * rates[l[-1]][currencies.index(currency)], l + [currency], size
        )
        if tmp > res and tmpL[-1] == "shells":
            res, resL = tmp, tmpL
    return res, resL


def main():
    print(max(rec(1, ["shells"], i) for i in range(3, 7)))


if __name__ == "__main__":
    main()
