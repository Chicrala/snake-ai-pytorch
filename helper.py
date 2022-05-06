import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)

'''
def moreplots(scores, mean_scores, rewards, mean_rewards):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(scores)
    ax0_b = ax[0].twinx()
    ax0_b.plot(mean_scores)
    ax[0].set_ylabel('Score')
    ax0_b.set_ylabel('Mean Score')

    #ax[0].set_ylim(ymin=0)
    #ax1_b.set_ylim(ymin=0)
    ax[0].text(len(scores) - 1, scores[-1], str(scores[-1]))
    ax0_b.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))

    ax[1].plot(rewards)
    ax1_b = ax[1].twinx()
    ax1_b.plot(mean_rewards)
    ax[1].set_ylabel('Rewards')
    ax1_b.set_ylabel('Mean Rewards')

    #ax[1].set_ylim(ymin=0)
    #ax1_b.set_ylim(ymin=0)
    ax[1].text(len(rewards) - 1, rewards[-1], str(rewards[-1]))
    ax1_b.text(len(mean_rewards) - 1, mean_rewards[-1], str(mean_rewards[-1]))

    #ax.hold(True)
    #ax[0].hold(True)
    #ax[0]_b.hold(True)
    #ax[1].hold(True)
    #ax[1]_b.hold(True)

    plt.show(block=False)
    plt.draw()
    plt.pause(.1)
'''