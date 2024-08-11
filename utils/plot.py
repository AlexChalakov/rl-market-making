import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16
plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # font size of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # font size of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # font size of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # font size of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend font size
plt.rc('figure', titlesize=BIGGER_SIZE)  # font size of the figure title


def plot_observation_space(observation: np.ndarray,
                           labels: str,
                           save_filename: str or None = None) -> None:
    """
    Represent all the observation spaces seen by the agent as one image.
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    im = ax.imshow(observation,
                   interpolation='none',
                   cmap=cm.get_cmap('seismic'),
                   origin='lower',
                   aspect='auto',
                   vmax=observation.max(),
                   vmin=observation.min())
    plt.xticks(range(len(labels)), labels, rotation='vertical')
    plt.tight_layout()

    if save_filename is None:
        plt.show()
    else:
        plt.savefig(f"{save_filename}_OBS.png")
        plt.close(fig)


class Visualize(object):

    def __init__(self,
                 columns: list or None,
                 store_historical_observations: bool = True):
        """
        Helper class to store episode performance.

        :param columns: Column names (or labels) for rending data
        :param store_historical_observations: if TRUE, store observation
            space for rendering as an image at the end of an episode
        """
        self._data = list()
        self._columns = columns

        # Observation space for rendering
        self._store_historical_observations = store_historical_observations
        self._historical_observations = list()
        self.observation_labels = None

    def add_observation(self, obs: np.ndarray) -> None:
        """
        Append current time step of observation to list for rendering
        observation space at the end of an episode.

        :param obs: Current time step observation from the environment
        """
        if self._store_historical_observations:
            self._historical_observations.append(obs)

    def add(self, *args):
        """
        Add time step to visualizer.

        :param args: midpoint, buy trades, sell trades
        :return:
        """
        self._data.append(args)

    def to_df(self) -> pd.DataFrame:
        """
        Get episode history of prices and agent transactions in the form of a DataFrame.

        :return: DataFrame with episode history of prices and agent transactions
        """
        return pd.DataFrame(data=self._data, columns=self._columns)

    def reset(self) -> None:
        """
        Reset data for new episode.
        """
        self._data.clear()
        self._historical_observations.clear()
        
    # Plot this entire history of an episode including:
    # 1) Midpoint prices with trade executions
    # 2) Inventory count at every step
    # 3) Realized PnL at every step
    def plot_episode_history(self, history: pd.DataFrame or None = None,
                             save_filename: str or None = None) -> None:
        if isinstance(history, pd.DataFrame):
            data = history  # data from past episode
        else:
            data = self.to_df()

        midpoints = data['midpoint'].values
        long_fills = data.loc[data['buys'] > 0., 'buys'].index.values
        short_fills = data.loc[data['sells'] > 0., 'sells'].index.values
        inventory = data['inventory'].values
        pnl = data['realized_pnl'].values

        heights = [6, 2, 2]
        widths = [14]
        gs_kw = dict(width_ratios=widths, height_ratios=heights)
        fig, axs = plt.subplots(nrows=len(heights), ncols=len(widths),
                                sharex=True,
                                figsize=(widths[0], int(sum(heights))),
                                gridspec_kw=gs_kw)

        axs[0].plot(midpoints, label='midpoints', color='blue', alpha=0.6)
        axs[0].set_ylabel('Midpoint Price (USD)', color='black')

        # Redundant labeling for all computer compatibility
        axs[0].set_facecolor("w")
        axs[0].tick_params(axis='x', colors='black')
        axs[0].tick_params(axis='y', colors='black')
        axs[0].spines['top'].set_visible(True)
        axs[0].spines['right'].set_visible(True)
        axs[0].spines['bottom'].set_visible(True)
        axs[0].spines['left'].set_visible(True)
        axs[0].spines['top'].set_color("black")
        axs[0].spines['right'].set_color("black")
        axs[0].spines['bottom'].set_color("black")
        axs[0].spines['left'].set_color("black")
        axs[0].grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

        axs[0].scatter(x=long_fills, y=midpoints[long_fills], label='buys', alpha=0.7,
                       color='green', marker="^")

        axs[0].scatter(x=short_fills, y=midpoints[short_fills], label='sells', alpha=0.7,
                       color='red', marker="v")

        axs[1].plot(inventory, label='inventory', color='orange')
        axs[1].axhline(0., color='grey')
        axs[1].set_ylabel('Inventory Count', color='black')
        axs[1].set_facecolor("w")
        axs[1].tick_params(axis='x', colors='black')
        axs[1].tick_params(axis='y', colors='black')
        axs[1].spines['top'].set_visible(True)
        axs[1].spines['right'].set_visible(True)
        axs[1].spines['bottom'].set_visible(True)
        axs[1].spines['left'].set_visible(True)
        axs[1].spines['top'].set_color("black")
        axs[1].spines['right'].set_color("black")
        axs[1].spines['bottom'].set_color("black")
        axs[1].spines['left'].set_color("black")
        axs[1].grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

        axs[2].plot(pnl, label='Realized PnL', color='purple')
        axs[2].axhline(0., color='grey')
        axs[2].set_ylabel("PnL (%)", color='black')
        axs[2].set_xlabel('Number of steps (1 second each step)', color='black')
        # Redundant labeling for all computer compatibility
        axs[2].set_facecolor("w")
        axs[2].tick_params(axis='x', colors='black')
        axs[2].tick_params(axis='y', colors='black')
        axs[2].spines['top'].set_visible(True)
        axs[2].spines['right'].set_visible(True)
        axs[2].spines['bottom'].set_visible(True)
        axs[2].spines['left'].set_visible(True)
        axs[2].spines['top'].set_color("black")
        axs[2].spines['right'].set_color("black")
        axs[2].spines['bottom'].set_color("black")
        axs[2].spines['left'].set_color("black")
        axs[2].grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
        plt.tight_layout()

        if save_filename is None:
            plt.show()
        else:
            plt.savefig(f"{save_filename}.png")
            plt.close(fig)

    # Represent all the observation spaces seen by the agent as one image.
    def plot_obs(self, save_filename: str or None = None) -> None:
       
        observations = np.asarray(self._historical_observations, dtype=np.float32)
        plot_observation_space(observation=observations,
                               labels=self.observation_labels,
                               save_filename=save_filename)
        
import matplotlib.pyplot as plt
import numpy as np


class TradingGraph:
    """
    A stock trading visualization using matplotlib
    made to render OpenAI gym environments
    """
    plt.style.use('dark_background')

    def __init__(self, sym=None):
        # attributes for rendering
        self.sym = sym
        self.line1 = []
        self.screen_size = 1000
        self.y_vec = None
        self.x_vec = np.linspace(0, self.screen_size * 10,
                                 self.screen_size + 1)[0:-1]

    def reset_render_data(self, y_vec):
        self.y_vec = y_vec
        self.line1 = []

    def render(self, midpoint=100., mode='human'):
        if mode == 'human':
            self.line1 = self.live_plotter(self.x_vec,
                                           self.y_vec,
                                           self.line1,
                                           identifier=self.sym)
            self.y_vec = np.append(self.y_vec[1:], midpoint)

    @staticmethod
    def live_plotter(x_vec, y1_data, line1, identifier='Add Symbol Name',
                     pause_time=0.00001):
        if not line1:
            # this is the call to matplotlib that allows dynamic plotting
            plt.ion()
            fig = plt.figure(figsize=(20, 12))
            ax = fig.add_subplot(111)
            # create a variable for the line so we can later update it
            line1, = ax.plot(x_vec, y1_data, '-', label='midpoint', alpha=0.8)
            # update plot label/title
            plt.ylabel('Price')
            plt.legend()
            plt.title('Title: {}'.format(identifier))
            plt.show(block=False)

        # after the figure, axis, and line are created, we only need to update the
        # y-data
        line1.set_ydata(y1_data)

        # adjust limits if new data goes beyond bounds
        if np.min(y1_data) <= line1.axes.get_ylim()[0] or \
                np.max(y1_data) >= line1.axes.get_ylim()[1]:
            plt.ylim(np.min(y1_data), np.max(y1_data))

        # this pauses the data so the figure/axis can catch up
        # - the amount of pause can be altered above
        plt.pause(pause_time)

        # return line so we can update it again in the next iteration
        return line1

    @staticmethod
    def close():
        plt.close()