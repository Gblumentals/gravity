import asyncio
from typing import Dict, Any, Tuple

from binance import AsyncClient, BinanceSocketManager
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
import json
import config

debug = config.debug


async def depth_matrix(
    bsm: BinanceSocketManager,
    pair: str,
    filter: Dict[str, Any],
    calibration_window_seconds: float = +5 * 60,
) -> Dict[int, int]:
    """
    Populate depth matrix, storing observed depth cents from the mid price
    at which limit orders are hit and occurrences.
    Args:
        bsm: BinanceSocketManager
        pair: ticker symbol
        filter: price filter metadata from the exchange
        calibration_window_seconds: timeframe in seconds for calibrating
    Returns:
        Returns dictionary of depth cents and counts of orders
    """

    mid_price = 0
    order_depths = dict()
    run_start = time.time()
    ticker_stream = f"{pair.lower()}@bookTicker"
    trade_stream = f"{pair.lower()}@aggTrade"

    async with bsm.multiplex_socket([trade_stream, ticker_stream]) as stream:
        while time.time() - run_start <= calibration_window_seconds:
            res = await stream.recv()
            stream_name = res["stream"]

            # Ticker update
            if stream_name == ticker_stream:
                ticker = res["data"]
                mid_price = (float(ticker["a"]) + float(ticker["b"])) / 2

            # New trade update
            if stream_name == trade_stream:
                if mid_price > 0:
                    trade = res["data"]

                    # trade price distance from mid
                    distance = float(trade["p"]) - mid_price
                    depth_ticks = int(distance / filter["tickSize"])

                    # populate dictionary of order depths
                    if depth_ticks in order_depths.keys():
                        order_depths[depth_ticks] += 1
                    else:
                        order_depths[depth_ticks] = 1

        return order_depths


async def quote(
    bsm: BinanceSocketManager,
    pair: str,
    l_lim: float,
    u_lim: float,
    quote_window_seconds: int = 10,
    duration: float = 10 * 60,
) -> Dict[str, Any]:
    """
    Prints quotes every 10 seconds and checks if our orders are filled.
    Runs until keyboard interruption

    Args:
        bsm: BinanceSocketManager
        pair: ticker symbol
        l_lim: distance between our bid and best bid
        u_lim: distance between our ask and best ask
        quote_window_seconds: interval for printing quotes
        duration: how long to run the bot for
    Returns:
        Disctionary with registered best bid, ask, our quotes and filled trades
    """

    current_time = time.time()
    my_ask = 0
    my_bid = 0
    filled_bids = 0
    filled_asks = 0
    bid_is_filled = False
    ask_is_filled = False
    end_time = time.time() + duration

    ticker_stream = f"{pair.lower()}@bookTicker"
    trade_stream = f"{pair.lower()}@aggTrade"
    quotes = {
        "time": [],
        "best_bid": [],
        "best_ask": [],
        "my_bid": [],
        "my_ask": [],
        "filled_asks": [],
        "filled_bids": [],
    }

    # Can also test quotes with trade stream
    async with bsm.multiplex_socket([ticker_stream, trade_stream]) as stream:
        while time.time() <= end_time:
            res = await stream.recv()
            stream_name = res["stream"]

            if stream_name == ticker_stream:
                ticker = res["data"]

                best_bid = float(ticker["b"])
                best_ask = float(ticker["a"])

                # If time to quote
                if time.time() >= current_time + quote_window_seconds:
                    current_time = time.time()

                    my_bid = best_bid - abs(l_lim)
                    my_ask = best_ask + abs(u_lim)

                    quotes["time"].append(current_time)
                    quotes["best_ask"].append(best_ask)
                    quotes["best_bid"].append(best_bid)
                    quotes["my_ask"].append(my_ask)
                    quotes["my_bid"].append(my_bid)

                    if bid_is_filled:
                        quotes["filled_bids"].append(1)
                    else:
                        quotes["filled_bids"].append(0)

                    if ask_is_filled:
                        quotes["filled_asks"].append(1)
                    else:
                        quotes["filled_asks"].append(0)

                    bid_is_filled = False
                    ask_is_filled = False

                    print("best bid and ask", best_bid, best_ask)
                    print("my bid and ask", my_bid, my_ask)

                # check if order gets filled between new order placement
                if my_ask > 0 and my_bid > 0:
                    if best_bid < my_bid and not bid_is_filled:
                        filled_bids += 1
                        bid_is_filled = True

                    if best_ask > my_ask and not ask_is_filled:
                        filled_asks += 1
                        ask_is_filled = True

        return quotes


async def get_price_filter(client: AsyncClient, pair: str):
    """
    Gets meta data from binance about the pair.
    Gets price filter data.

    Args:
        client: Async Client
        pair: exchange pair symbol

    Returns:
        Relevant tick size constrictions for the pair
    """

    # Symbol metadata
    info = await client.get_symbol_info(pair)
    # Get price filters
    filters = info["filters"]
    for f in filters:
        if f["filterType"] == "PRICE_FILTER":
            price_filter = f
            price_filter["tickSize"] = float(f["tickSize"])

    return price_filter


async def plot_depths(depth_cents: np.ndarray, occurrences):
    """
    This function runs when debug=True.
    Plot Gaussian curve
    Args:
        depth_cents: distances from mid price for market orders
        occurrences: count of occurrences
    """
    fig, ax = plt.subplots(1, 1)
    ax.scatter(depth_cents, occurrences)
    ax.set_xlabel("depth_ticks")
    ax.set_ylabel("occurrences")
    # Calculate the mean and std of noticed order depths
    mu = np.mean(depth_cents)
    sigma = np.std(depth_cents)
    # simulate space
    u = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 100)
    # normalize data
    ax2 = ax.twinx()
    ax2.plot(u, stats.norm.pdf(u, mu, sigma), color="crimson")
    ax2.set_ylabel("normal curve")
    plt.show()


async def main():

    pair = "BTCUSDT"

    client = await AsyncClient.create()
    bsm = BinanceSocketManager(client)

    price_filter = await get_price_filter(client, pair)

    # Read exchange websocket for ticker and trade data
    order_depths = await depth_matrix(bsm, pair, price_filter, config.calibration_window)

    print(order_depths)

    # Convert populated dict to numpy array
    depth_cents = np.array(list(order_depths.keys())).astype(int)
    occurrences = np.array(list(order_depths.values())).astype(int)

    if debug:
        await plot_depths(depth_cents, occurrences)

    bid_spread = math.floor(np.quantile(depth_cents, 0.25)) * price_filter["tickSize"]
    ask_spread = math.ceil(np.quantile(depth_cents, 0.75)) * price_filter["tickSize"]

    # connect to websocket again to print quotes on events
    quotes = await quote(bsm, pair, bid_spread, ask_spread, config.quote_interval, config.quote_duration)

    for name, d in zip(["order-depths", "quotes"], [order_depths, quotes]):
        with open(f"{name}.json", "w") as fp:
            json.dump(d, fp)

    # Graceful exit
    await client.close_connection()


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
