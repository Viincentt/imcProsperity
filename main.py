from datamodel import (
    OrderDepth,
    UserId,
    TradingState,
    Order,
    Position,
    Symbol,
    Observation,
    ObservationValue,
    ConversionObservation,
    Product,
    Trade,
)
from typing import List
from math import ceil, floor, sqrt, erf, log, exp
import statistics, jsonpickle
import numpy as np
import math

profit_per_round = {
    "amethysts": [1.17, 1.42, 1.41, 1.64, 1.5],
    "starfruit": [1.55, 1.26, 0.98, 1.65, 1.1],
    "orchid": [0, 77.9, 8.72, 0, 0.3],
    "giftBox": [0, 0, 19.3, 7.9, -7],
    "chocolate": [0, 0, 8.4, 1.9, 0],
    "coupon": [0, 0, 0, -15, 2.4],
}


class Trader:

    def pricingError(self, buyPrice, sellPrice):
        if buyPrice > sellPrice:
            print("ERROR BUY HIGH", buyPrice, "SELL LOW", sellPrice)
            return True
        return False

    def verbosePrint(self, msg, verbose=False):
        if verbose:
            if type(msg) == str:
                print(msg)
            elif type(msg) == OrderDepth:
                self.printOrderDepth(msg)

    def printOrderDepth(self, order_depth: OrderDepth):
        print("sellOrders", order_depth.sell_orders)
        print("buyOrders", order_depth.buy_orders)

    def getWeightedArithmeticMean(self, order_depth: OrderDepth):
        s, total = 0, 0
        for price, nb in order_depth.buy_orders.items():
            s += price * nb
            total += nb
        for price, nb in order_depth.sell_orders.items():
            s += price * -nb
            total += -nb
        if total == 0:
            print("ERROR No order in the book")
            return 0
        return s / total

    def order(self, symbol: Symbol, price, qty):
        return Order(symbol, price, qty)

    def buy(self, symbol, price, qty, verbose=False):
        if verbose:
            print("BUY", str(qty) + "x", price)
        if qty < 0:
            print("ERROR Trying to buy", symbol, "but qty is negative")
        return self.order(symbol, price, qty)

    def sell(self, symbol, price, qty, verbose=False):
        if verbose:
            print("SELL", str(qty) + "x", price)
        if qty > 0:
            print("ERROR Trying to sell", symbol, "but qty is positive")
        return self.order(symbol, price, qty)

    def printObservations(self, obs: Observation, product, verbose=False):
        if not verbose:
            return
        if product in obs.conversionObservations:
            conv: ConversionObservation = obs.conversionObservations[product]
            print(
                "Conv ask",
                conv.askPrice,
                "bid",
                conv.bidPrice,
                "transport",
                conv.transportFees,
                "export",
                conv.exportTariff,
                "import",
                conv.importTariff,
                "sun",
                conv.sunlight,
                "humid",
                conv.humidity,
            )
        plain: ObservationValue = (
            obs.plainValueObservations[product]
            if product in obs.plainValueObservations
            else None
        )
        print("Plain", plain)

    def printState(self, state: TradingState, symbol: Symbol):
        self.printOrderDepth(state.order_depths[symbol])
        self.printObservations(state.observations, symbol)

    def printPosition(self, symbol, position):
        print(symbol[:3], "Position " + ("+" if position > 0 else "") + str(position))

    def trade(
        self,
        symbol: Symbol,
        state: TradingState,
        res: list[Order],
        buyPrice: int,
        sellPrice: int,
        verbose=False,
        limit=20,
    ):
        if self.pricingError(buyPrice, sellPrice):
            return

        order_depth: OrderDepth = state.order_depths[symbol]
        sellOrders = sorted(order_depth.sell_orders.items())
        buyOrders = sorted(order_depth.buy_orders.items(), reverse=True)
        position: Position = state.position[symbol] if symbol in state.position else 0
        buyQty, sellQty = limit - position, -limit - position
        self.printPosition(symbol, position)

        # ---- MARKET TAKE
        self.verbosePrint("Take", verbose)
        # buy
        for ask, askQty in sellOrders:
            if buyQty <= 0 or ask > buyPrice:
                break
            if ask < buyPrice or buyQty > limit:
                # the second condition implies that you cant sell much
                # meaning you cant short much so we are trying to fix
                # that by accepting to buy without profit potentially
                qty = min(-askQty, buyQty)
                buyQty -= qty
                res.append(self.buy(symbol, ask, qty, verbose))

        # sell
        for bid, bidQty in buyOrders:
            if sellQty >= 0 or bid < sellPrice:
                break
            if bid > sellPrice or -sellQty > limit:
                qty = max(-bidQty, sellQty)
                sellQty -= qty
                res.append(self.sell(symbol, bid, qty, verbose))

        # ---- MARKET MAKE
        self.verbosePrint("Make", verbose)
        # buy
        softLimit = 0.5 * limit
        price = max(buyOrders)[0]
        if buyQty > softLimit:
            price += 1
        if buyQty > limit:
            price += 1
        price = min(price, buyPrice - (-sellQty > softLimit))
        res.append(self.buy(symbol, price, buyQty, verbose))

        # sell
        price = min(sellOrders)[0]
        if -sellQty > softLimit:
            price -= 1
        if -sellQty > limit:
            price -= 1
        price = max(price, sellPrice + (buyQty > softLimit))
        res.append(self.sell(symbol, price, sellQty, verbose))

    def starfruitStrategy(self, state: TradingState, verbose=False):
        res, symbol = [], "STARFRUIT"

        order_depth: OrderDepth = state.order_depths[symbol]
        fair_price = self.getWeightedArithmeticMean(order_depth)
        buyPrice, sellPrice = floor(fair_price), ceil(fair_price)

        self.trade(symbol, state, res, buyPrice, sellPrice, verbose)
        return res

    def amethystsStrategy(self, state: TradingState, verbose=False):
        res, symbol, fair_price = [], "AMETHYSTS", 10000
        self.trade(symbol, state, res, fair_price, fair_price, verbose)
        return res

    def orchidTrade(
        self,
        symbol: Symbol,
        state: TradingState,
        res: list[Order],
        buyPrice: int,
        sellPrice: int,
        verbose=False,
        limit=100,
    ):
        if self.pricingError(buyPrice, sellPrice):
            return

        order_depth: OrderDepth = state.order_depths[symbol]
        sellOrders = sorted(order_depth.sell_orders.items())
        buyOrders = sorted(order_depth.buy_orders.items(), reverse=True)
        buyQty, sellQty = limit, -limit

        # ---- MARKET TAKE
        self.verbosePrint("Take", verbose)
        # buy
        for ask, askQty in sellOrders:
            if buyQty <= 0 or ask > buyPrice:
                break
            qty = min(-askQty, buyQty)
            buyQty -= qty
            res.append(self.buy(symbol, ask, qty, verbose))

        # sell
        for bid, bidQty in buyOrders:
            if sellQty >= 0 or bid < sellPrice:
                break
            qty = max(-bidQty, sellQty)
            sellQty -= qty
            res.append(self.sell(symbol, bid, qty, verbose))

        # ---- MARKET MAKE
        # TODO sometimes about 50% are willing to sell at price + 2
        # so we should catch them at +2 instead of +1
        # try to find out when they are willing to buy at +2
        # bots on the market have way more capital/position than us
        # they will buy/sell everything that they deem fair
        self.verbosePrint("Make", verbose)

        # buy
        res.append(self.buy(symbol, buyPrice, buyQty, verbose))

        # sell
        """
        50 39   sellPrice
        75 263  + 1
        87 227  + 2
        93 27   + 3
        """
        res.append(self.sell(symbol, sellPrice, -(-sellQty // 2), verbose))
        res.append(self.sell(symbol, sellPrice + 1, -(-sellQty // 2), verbose))
        # res.append(self.sell(symbol, sellPrice + 1, sellQty, verbose))

    def orchidStrategy(self, state: TradingState, verbose=False):
        res, symbol, limit = [], "ORCHIDS", 100
        order_depth: OrderDepth = state.order_depths[symbol]
        conversion = -state.position[symbol] if symbol in state.position else 0
        self.printPosition(symbol, -conversion)

        fair_price = self.getWeightedArithmeticMean(order_depth)
        buyPrice, sellPrice = floor(fair_price + 0.1), ceil(fair_price)
        """
        self.verbosePrint(
            "Main buy {:.1f} sell {:.1f}".format(buyPrice, sellPrice),
            verbose,
        )
        """

        obs = state.observations.conversionObservations[symbol]
        imp, exp, transp = obs.importTariff, obs.exportTariff, obs.transportFees
        # our price if we buy/sell in south
        southBuy = obs.askPrice + transp + imp
        # ^^^^ don't need to + 0.1 because if we buy from south it means
        # we were short position and we can only go back to 0
        southSell = obs.bidPrice - transp - exp
        """
        self.verbosePrint(
            "South buy {:.1f} sell {:.1f}".format(southBuy, southSell),
            verbose,
        )
        """

        buyPrice = floor(min(buyPrice, southBuy))
        sellPrice = ceil(max(sellPrice, southSell))
        if southBuy + 1 < sellPrice:
            # if interesting to buy in south then sell on main just above south buy price
            sellPrice = ceil(southBuy)
            self.orchidTrade(symbol, state, res, buyPrice, sellPrice, verbose, limit)
        if southSell > buyPrice + 1:
            # if interesting to sell in south then buy on main just below south sell price
            buyPrice = floor(southSell)
            self.orchidTrade(symbol, state, res, buyPrice, sellPrice, verbose, limit)
        self.verbosePrint(
            "Buy Price " + str(buyPrice) + " Sell Price " + str(sellPrice), verbose
        )

        self.verbosePrint(order_depth, verbose)

        return res, conversion

    def loveStrategy(self, state: TradingState, verbose=False):
        """
        NOTE
        - gift box price is always pricier than individual components tgt
        - it seems that only the gift basket is worth trading.
        the individual components dont show much profit.
        """
        box, strawberry = "GIFT_BASKET", "STRAWBERRIES"
        chocolate, rose, love = "CHOCOLATE", "ROSES", "love"
        buyQty, sellQty = "buyQty", "sellQty"
        symbol = [box, chocolate, strawberry, rose]
        limit = {box: 60, chocolate: 250, strawberry: 350, rose: 60}
        res, orderDepth, price, rawData = {}, {}, {}, state.traderData
        qty, pos, data = {}, {}, jsonpickle.loads(rawData) if rawData else {}
        for sym in symbol:
            res[sym] = []
            orderDepth[sym] = state.order_depths[sym]
            price[sym] = self.getWeightedArithmeticMean(orderDepth[sym])
            pos[sym] = state.position[sym] if sym in state.position else 0
            qty[sym] = {buyQty: limit[sym] - pos[sym], sellQty: -limit[sym] - pos[sym]}

        # self.printPosition(box, pos[box])
        # self.printPosition(chocolate, pos[chocolate])
        self.printPosition(rose, pos[rose])

        nData = 500
        arb = 4 * price[chocolate] + 6 * price[strawberry] + price[rose]
        boxDiff = price[box] - arb
        boxData: list[float] = data[box][-nData:] if box in data else []
        boxData.append(boxDiff)
        if len(boxData) < 2:
            return res, {box: boxData}

        # ====== GIFTBOX AND CHOCOLATE

        boxStd, boxMean = np.std(boxData), statistics.mean(boxData)
        envTrade, envClose = 0.8, 0.15
        boxTrustIndex, boxDist = len(boxData) / nData, abs(boxDiff - boxMean)
        boxBuy, boxSell = floor(price[box]), ceil(price[box])
        # this value 7 is empirical
        edge, qtyDiv = floor(boxDist / 7), 3
        if boxTrustIndex * boxDist > envTrade * boxStd:
            # print("Deviating")
            # the more we deviate the more we allow less favorable prices
            # cuz we know we are going to go back to normal so we really want to take position
            boxBuy, boxSell = boxBuy + edge, boxSell - edge
            if boxDiff > boxMean:
                # print("Diverging")
                boxQty = -(-qty[box][sellQty] // qtyDiv)
                res[box].append(self.sell(box, boxSell, boxQty, verbose))
                for comp in [chocolate, rose]:
                    compQty = qty[comp][buyQty] // qtyDiv
                    compBuy = floor(price[comp]) + edge
                    res[comp].append(self.buy(comp, compBuy, compQty, verbose))
            elif boxDiff < boxMean:
                # print("Converging")
                boxQty = qty[box][buyQty] // qtyDiv
                res[box].append(self.buy(box, boxBuy, boxQty, verbose))
                for comp in [chocolate, rose]:
                    compQty = qty[comp][sellQty] // qtyDiv
                    compSell = ceil(price[comp]) - edge
                    res[comp].append(self.sell(comp, compSell, compQty, verbose))
        elif boxDist < envClose * boxStd * boxTrustIndex:
            # print("Closing")
            # doesnt necesarily need the closing since the diverging and converging compensate each other
            # this is just to be safe in case the dataset ends in an deviation
            boxPos = pos[box]
            if boxPos > 0:
                res[box].append(self.sell(box, boxSell, -boxPos, verbose))
            else:
                res[box].append(self.buy(box, boxBuy, -boxPos, verbose))

        return res, {box: boxData}

    """
    ===== TO COMPUTE the volatility for black scholes

    def black_scholes_price(S, K, t, r, sigma, option_type='call'):
        d1 = (np.log(S / K) + (r + 0.5 * sigma*2) t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)
        price = S * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)
        return price

    def implied_volatility(market_price, S, K, t, r, initial_guess=0.2):
        objective_function = lambda sigma: black_scholes_price(S, K, t, r, sigma) - market_price
        return newton(objective_function, initial_guess)
    """

    def normal_cdf(self, x):
        """Cumulative distribution function for the standard normal distribution."""
        return (1 + erf(x / sqrt(2))) / 2

    def blackScholes(self, S, K, T, r, sigma):
        """
        S: underlying asset mid price
        K: strike price of the option
        T: time to expiration in years
        r: risk free interest rate
        sigma: volatility
        """
        d1 = (log(S / K) + (r + sigma**2 / 2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        price = S * self.normal_cdf(d1) - K * exp(-r * T) * self.normal_cdf(d2)
        return int(round(price)), self.normal_cdf(d1)

    def coconutStrategy(self, state: TradingState, verbose=False):
        coconut, coupon = "COCONUT", "COCONUT_COUPON"
        buyQty, sellQty = "buyQty", "sellQty"
        res = {coconut: [], coupon: []}
        limit = {coconut: 300, coupon: 600}
        qty, pos, orderDepth, price = {}, {}, {}, {}
        for sym in [coconut, coupon]:
            orderDepth[sym] = state.order_depths[sym]
            price[sym] = self.getWeightedArithmeticMean(orderDepth[sym])
            pos[sym] = state.position[sym] if sym in state.position else 0
            qty[sym] = {buyQty: limit[sym] - pos[sym], sellQty: -limit[sym] - pos[sym]}

        nData, data = 500, state.traderData
        data = jsonpickle.loads(data) if data else {}
        cocoPrices = data[coconut][-nData:] if coconut in data else []
        cocoPrices.append(price[coconut])
        previousCoupon = data[coupon] if coupon in data else None

        if len(cocoPrices) < 3:
            return res, {coconut: cocoPrices, coupon: price[coupon]}

        returns = [
            np.log(cocoPrices[i] / cocoPrices[i - 1]) for i in range(1, len(cocoPrices))
        ]
        volatility = np.std(returns) * np.sqrt(252)
        volatility, rate, T = 0.1609616171503603, 0, 246 / 252
        bsPrice, delta = self.blackScholes(price[coconut], 10000, T, rate, volatility)

        diff = cocoPrices[-1] - cocoPrices[-2]
        deltaCouponPrice = previousCoupon + delta * diff
        predPrice = bsPrice
        dist = abs(predPrice - price[coupon])
        priceEdge = 1.5
        buyPrice, sellPrice = floor(predPrice + dist / priceEdge), ceil(
            predPrice - dist / priceEdge
        )
        qtyDiv = 3

        if dist > 5:
            if buyPrice > price[coupon]:
                # buy coupon
                res[coupon].append(
                    self.buy(coupon, buyPrice, qty[coupon][buyQty] // qtyDiv, verbose)
                )
            elif sellPrice < price[coupon]:
                # sell coupon
                res[coupon].append(
                    self.sell(
                        coupon, sellPrice, qty[coupon][sellQty] // qtyDiv, verbose
                    )
                )
        """
        if pos[coupon] > 0:
            res[coconut].append(
                self.sell(
                    coconut,
                    ceil(price[coconut]),
                    max(qty[coconut][sellQty], -floor(delta * pos[coupon])),
                    verbose,
                )
            )
        elif pos[coupon] < 0:
            res[coconut].append(
                self.buy(
                    coconut,
                    floor(price[coconut]),
                    min(qty[coconut][buyQty], floor(delta * -pos[coupon])),
                    verbose,
                )
            )
        """

        return res, {}  # {coconut: cocoPrices, coupon: price[coupon]}

    def printTrades(self, trades: List[Trade]):
        for tr in trades:
            print(
                tr.symbol,
                tr.quantity,
                "x",
                tr.price,
                "buyer",
                tr.buyer,
                "seller",
                tr.seller,
            )

    def strategies(self, state: TradingState):
        res, conversion, traderData = {}, 0, {}
        """
        for symbol in state.own_trades:
            self.printTrades(state.own_trades[symbol])
        for symbol in state.market_trades:
            self.printTrades(state.market_trades[symbol])
        """

        for symbol in state.listings:
            match symbol:
                case "ORCHIDS":
                    # continue
                    res[symbol], conversion = self.orchidStrategy(state, verbose=False)
                case "STARFRUIT":
                    # continue
                    res[symbol] = self.starfruitStrategy(state, verbose=False)
                case "AMETHYSTS":
                    # continue
                    res[symbol] = self.amethystsStrategy(state, verbose=False)
                case "GIFT_BASKET" | "CHOCOLATE" | "STRAWBERRIES" | "ROSES":
                    # continue
                    if symbol == "GIFT_BASKET":
                        # just to call once
                        loveRes, loveData = self.loveStrategy(state, verbose=False)
                        res.update(loveRes)
                        traderData.update(loveData)
                case "COCONUT_COUPON" | "COCONUT":
                    # continue
                    if symbol == "COCONUT":
                        coconutRes, cocoData = self.coconutStrategy(
                            state, verbose=False
                        )
                        res.update(coconutRes)
                        traderData.update(cocoData)
                case _:
                    print("Unknown Product")
        return res, conversion, jsonpickle.dumps(traderData)

    """
    TODO 
    start with products we aren't trading
    copy trade top trader of each product
    try to get worse traders trade ie if remy is really bad; and he is selling 10 stocks try to buy these stocks
    """

    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        return self.strategies(state)

    def marketMaker_(
        self,
        symbol: Symbol,
        state: TradingState,
        res: list[Order],
        fairPrices: tuple[int, int],  # fair buy price, fair selling price
        buyPrices: tuple[int, int],
        sellPrices: tuple[int, int],
        limit=20,
        verbose=False,
    ):
        order_depth: OrderDepth = state.order_depths[symbol]
        position: Position = state.position[symbol] if symbol in state.position else 0
        buyQty, sellQty = limit - position, -limit - position
        marketMakeQtyCoef = 1.5
        # the lower this coef the more qty for closer prices to fair price

        print(symbol[:3], "Position", position)
        minBuy, maxBuy = buyPrices[0], buyPrices[1]
        minSell, maxSell = sellPrices[0], sellPrices[1]

        # ---- MARKET TAKE
        softMarketLimit = int(0.35 * limit)
        hardMarketLimit = int(0.8 * limit)
        if abs(position) >= softMarketLimit:
            self.verbosePrint("Market take", verbose)
            if position > 0:
                # sell
                for bid, bidQty in sorted(order_depth.buy_orders.items(), reverse=True):
                    if sellQty > limit:
                        break
                    if bid >= fairPrices[1]:
                        self.verbosePrint("Market take SELL", verbose)
                        qty = max(-bidQty, sellQty)
                        res.append(self.sell(symbol, bid, qty, verbose))
                        sellQty -= qty
                # this yields worse results but I still think it is a good idea...
                # if we REALLY need to fix our position and we couldn't find any acceptable
                # order in the book then we just send an order to try fixing the position.
                # maybe try with less qty?
                if sellQty == -limit - position and position >= hardMarketLimit:
                    self.verbosePrint("Hard limit sell", verbose)
                    qty = softMarketLimit - position
                    sellQty -= qty
                    res.append(self.sell(symbol, minSell, qty, verbose))
            else:
                # buy
                for ask, askQty in sorted(order_depth.sell_orders.items()):
                    if buyQty < limit:
                        break
                    if ask <= fairPrices[0]:
                        self.verbosePrint("Market take BUY", verbose)
                        qty = min(-askQty, buyQty)
                        res.append(self.buy(symbol, ask, qty, verbose))
                        buyQty -= qty
                if buyQty == limit - position and -position >= hardMarketLimit:
                    self.verbosePrint("Hard limit buy", verbose)
                    qty = -position - softMarketLimit
                    buyQty -= qty
                    res.append(self.buy(symbol, maxBuy, qty, verbose))

        # MARKET MAKE

        if verbose:
            self.printOrderDepth(order_depth)
            print("BuyQty", buyQty, "MinBuy", minBuy, "MaxBuy", maxBuy)
            print("SellQty", sellQty, "MinSell", minSell, "MaxSell", maxSell)
