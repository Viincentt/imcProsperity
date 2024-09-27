def profit(low, high):
    lowNumber = low - 900
    lowPerFishProfit = 1000 - low
    lowProfit = lowNumber * lowPerFishProfit

    highNumber = high - 900
    highPerFishProfit = 1000 - high
    highProfit = highNumber * highPerFishProfit
    return lowProfit + highProfit


def main():
    res = 0
    l, h = 0, 0
    for low in range(900, 1000):
        for high in range(low, 1000):

            tmp = profit(low, high)
            if tmp > res:
                l, h = low, high
                res = tmp
            print(low, high, tmp)

    print(l, h, res)


if __name__ == "__main__":
    main()
