import matplotlib.pyplot as plt
import numpy as np

def displayReconstruction(reconstruction):

    plt.imshow(reconstruction.reshape(28,28));
    plt.gca().set_xticklabels(''); plt.gca().set_yticklabels('');

def plotFilterBank(weights):
    numUnits = len(weights)

    plt.figure(figsize=(24,24))
    plt.subplot(int(np.sqrt(numUnits)),int(np.sqrt(numUnits)),1)

    for unitIdx in range(numUnits):
        plt.subplot(int(np.sqrt(numUnits)),int(np.sqrt(numUnits)),unitIdx+1)
        plt.imshow(weights[unitIdx],interpolation='bicubic');
        plt.gca().set_xticklabels(''); plt.gca().set_yticklabels('');

    plt.suptitle('Filter Bank',fontsize=72, fontweight='bold');

def plotRandomUnitActivations(activations,):
    """ plot the activation histograms of 10 randomly chosen units"""
    numImages, numUnits = activations.shape
    numToDisplay = 10

    unitsToDisplay = np.random.choice(range(numUnits), size = numToDisplay)
    fig, axes = plt.subplots(nrows=2,ncols=numToDisplay//2,
                             sharex=True,sharey=False,
                            figsize=(12,6))

    plt.suptitle("Activation Histograms of " + str(numToDisplay) + " Random Units Over " + str(numImages) + " Images",
                 fontsize='xx-large', fontweight='bold')
    xMax = np.max(activations)
    for idx,unit in enumerate(unitsToDisplay):
        eps = 0.05
        axes[idx%2,idx%5].hist(activations[:,unit],
                               bins=np.arange(0,xMax+eps,eps),
                               normed=True,log=False)

def plotSparsityStatistics(activations):
    """ plots lifetime sparsity (i.e. fraction of non-zero responses to images, aggregated over units)
        and population sparsity (i.e. fraction of non-zero units, aggregated over images)"""
    numImages, numUnits = activations.shape
    plt.figure(figsize=(12,6))

    plt.subplot(121)

    population_sparsities = np.mean(np.equal(activations,0.0),axis=1)

    plt.hist(population_sparsities,normed=True,
            bins=np.arange(0,1.1,0.1));
    plt.title("Population Sparsity",
                 fontsize='x-large', fontweight='bold');

    plt.subplot(122)

    lifetime_sparsities = np.mean(np.equal(activations,0.0),axis=0)
    plt.hist(lifetime_sparsities,normed=True,
             bins=np.arange(0,1.1,0.1));
    plt.title("Lifetime Sparsities ",
                 fontsize='x-large', fontweight='bold');

    plt.suptitle("Sparsity Statistics Over " + str(numImages) + " Images",
                 fontsize='xx-large', fontweight='bold');

def plotUnitActivations(activations):
    """plots a histogram of all unit activations"""
    plt.figure();
    plt.hist(np.ravel(activations),normed=True,log=False,);
    plt.suptitle('Averaged Hidden Unit Activations',fontsize='xx-large', fontweight='bold');
