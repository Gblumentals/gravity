# Probability estimation
The algorithm works by connecting to binance, registering market orders and the distances from the mid price 
at which they occurred for a set period of time (calibration window). These distances are measured in cents from the 
mid price (currency specific). Then it calibrates itself by finding the quantile Q3 and Q1 at which these market orders happened. 
The solution expects the posted limit orders to be at the edge of this range with a 50% 
chance of the orders getting triggered. The algorithm simulates orders with a type of paper trading. Posted orders occur 
every 10 seconds, the ticker updates from the exchange happen in milliseconds, at each of these events we check if the 
best bid and/or best ask are at higher or lower level than our orders and consider an order filled if that is so. 
Orders get posted/printed for a specified period after calibration. After the run is done results are stored in **_quotes.json_**
and **_order-depths.json_**

## Environment and Dependencies
To import and install dependencies using Anaconda: </br>

```conda env create --name gravity --file=environment.yml```

## Running the bot
`python bot.py` </br>
</br>
The bot starts quoting after the calibration window is done. Will not print anything in between.

### Config
_**config.py**_ has properties to control the duration and debug paramaters of the algorithm 
* _quote_interval_, how many seconds in between quotes, 
* _calibration_window_, how long to build depth matrix for,
* _quote_duration_, how long to run the bot,
* _debug_, if `True` then plots debugging info between calibration and quoting;

## Testing
Testing is done by analyzing the fill ratio of bids and asks during a run. We visualize the calibration input and 
behaviour of quotes and filled orders in a jupyter notebook.

`jupyter notebook report.ipynb` </br>

Examples with previous data in HTML format:
* [Mid sample size](/https://github.com/Gblumentals/gravity/blob/main/report%20(1).html)
* [Small sample size](/report.html)

### Old Data
For the simplicity of the reader old data from a previous run is kept, so the jupyter notebook can run before running
the bot.

### Calibration Input
X is the depth at which the orders arrive and Y is the count of occurences.
![Depth Matrix](/order-depths.png "Depth Matrix")

### Results plot
Scatter markers represent filled bids and asks.
![Quotes and filled orders](/quotes_and_orders.PNG "Quotes and filled orders")

### Debugging
If `debug=True` in bot.py then between the calibration window and the quoting window, the depth matrix with a Gaussian 
bell curve will be plotted in a line and scatter chart.

![Gaussian Bell Curve](/gaussian.png "Gaussian Bell Curve")


### Approach and Notes

My approach is based on finding the .75 and .25 quantile of depths at which market orders arrive and using them to make the 
respective bid and ask spread.
These depths could also be modelled in a Poisson process or a probability distribution for other approaches of the 
estimation. Some code  inside the bot.py has another version that I was working on where the depth matrix is modelled 
as a Gaussian.