from pathlib import Path
import csv
import matplotlib.pyplot as plt


def main():
    ame, star, orch = "AMETHYSTS", "STARFRUIT", "ORCHIDS"
    box, choc, straw = "GIFT_BASKET", "CHOCOLATE", "STRAWBERRIES"
    rose, coupon, coco = "ROSES", "COCONUT_COUPON", "COCONUT"
    symbols = [ame, star, box, choc, straw, rose, coupon, coco]
    # symbols.append(orch)
    valentina, vinnie, vladimir = "Valentina", "Vinnie", "Vladimir"
    vivian, celeste, colin = "Vivian", "Celeste", "Colin"
    carlos, camilla, pablo = "Carlos", "Camilla", "Pablo"
    penelope, percy, petunia = "Penelope", "Percy", "Petunia"
    ruby, remy, rhianna = "Ruby", "Remy", "Rhianna"
    raj, amelia, adam = "Raj", "Amelia", "Adam"
    alina, amir = "Alina", "Amir"
    peeps = [
        valentina,
        vinnie,
        vladimir,
        vivian,
        celeste,
        colin,
        carlos,
        camilla,
        pablo,
        penelope,
        percy,
        petunia,
        ruby,
        remy,
        rhianna,
        raj,
        amelia,
        adam,
        alina,
        amir,
    ]
    buyer, seller, symbol, pos = "buyer", "seller", "symbol", "pos"
    price, quantity, timestamp = "price", "quantity", "timestamp"
    profit = "profit"
    prices = {}
    pnls = {}
    for sym in symbols:
        pnls[sym] = {peep: {profit: [(-1, 0)], pos: 0} for peep in peeps}
        prices[sym] = {}
        prices[sym][buyer] = {peep: [] for peep in peeps}
        prices[sym][seller] = {peep: [] for peep in peeps}
    for round in range(1, 5):
        base = 0
        for day in range(-2, 4):
            file = "d5Data/trades_round_" + str(round) + "_day_" + str(day) + "_wn.csv"
            if not Path(file).exists():
                continue
            with open(file, mode="r") as f:
                lines = csv.DictReader(f, delimiter=";")
                for line in lines:
                    sym = line[symbol]
                    if sym not in symbols:
                        print("ERROR symbol unknown")
                        continue
                    buyerName, sellerName = line[buyer], line[seller]
                    if buyerName not in peeps or sellerName not in peeps:
                        print("ERROR unknown peep", buyerName, sellerName)
                        continue
                    pr, qty = float(line[price]), float(line[quantity])
                    """
                    if buyerName == sellerName:
                        continue
                    """

                    if buyerName in peeps:
                        prices[sym][buyer][buyerName].append(
                            (base + int(line[timestamp]) / 100, pr)
                        )

                        buyerPnl = pnls[sym][buyerName]
                        buyerPnl[profit].append(
                            (
                                base + int(line[timestamp]) / 100,
                                buyerPnl[profit][-1][1] - pr * qty,
                            )
                        )
                        buyerPnl[pos] += qty
                    if sellerName in peeps:
                        prices[sym][seller][sellerName].append(
                            (base + int(line[timestamp]) / 100, pr)
                        )
                        sellerPnl = pnls[sym][sellerName]
                        sellerPnl[profit].append(
                            (
                                base + int(line[timestamp]) / 100,
                                sellerPnl[profit][-1][1] + pr * qty,
                            )
                        )
                        sellerPnl[pos] -= qty
            base += 10000
    for sym in [star]:
        title = sym + " prices"
        print(title)
        tmp_peeps = [remy, ruby, rhianna, amelia]
        for peep, p in prices[sym][buyer].items():
            if peep not in tmp_peeps:
                continue
            if p:
                tmp_peep = peep + "_buy"
                print(tmp_peep, len(p))
                x, y = zip(*p)
                plt.plot(x, y, label=tmp_peep)

        for peep, p in prices[sym][seller].items():
            if peep not in tmp_peeps:
                continue
            if p:
                tmp_peep = peep + "_sell"
                print(tmp_peep, len(p))
                x, y = zip(*p)
                plt.plot(x, y, label=tmp_peep)

        """
        for peep in pnls[sym]:
            pnl = pnls[sym][peep]
            if len(pnl[profit]) > 2:
                print(peep, pnl[pos])

                plt.plot(
                    [p[0] for p in pnl[profit]], [p[1] for p in pnl[profit]], label=peep
                )
        """
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    main()
