import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import sys
import tensorflow as tf

from IPython import embed

def copy_files(outdir):
    """Copies files to the outdir to store complete script with each trial"""
    codedir = outdir+"/code"
    if not os.path.exists(codedir):
        os.makedirs(codedir)

    code = []
    exclude = set(['logs', 'paper_experiments', 'experiments'])
    for root, dirs, files in os.walk(".", topdown=True):
        dirs[:] = [d for d in dirs if d not in exclude]
        for f in files:
            if not f.endswith('.py'):
                continue
            code += [(root,f)]
    for r, f in code:
        if not os.path.exists(codedir+'/'+r):
            os.makedirs(codedir+'/'+r)
        shutil.copy2(r+'/'+f, codedir+'/'+r+'/'+f)


def scatter_points(points, directory, iteration, flow_length, reference_points=None, X_LIMS=(-7,7), Y_LIMS=(-7,7)):

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    ax.scatter(points[:, 0], points[:, 1], alpha=0.7, s=25)
    if reference_points is not None:
        ax.scatter(reference_points[:, 0], reference_points[:, 1], alpha=0.7, s=10, c='r')
    ax.set_xlim(*X_LIMS)
    ax.set_ylim(*Y_LIMS)
    ax.set_title(
        "Flow length: {}\n Samples on iteration #{}"
        .format(flow_length, iteration)
    )

    fig.savefig(os.path.join(directory, "flow_result_{}.png".format(iteration)))
    plt.close()


def plot_density_from_samples(samples, directory, iteration, flow_length, X_LIMS=(-7,7), Y_LIMS=(-7,7)):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import kde

    # create data
    x = samples[:,0]
    y = samples[:,1]

    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    nbins=300
    k = kde.gaussian_kde([x,y], bw_method=.1)
    x1 = np.linspace(*X_LIMS, 300)
    x2 = np.linspace(*Y_LIMS, 300)
    xi, yi = np.meshgrid(x1, x2)
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    # Make the plot
    fig = plt.figure(figsize=(7, 7))
    plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap='hot')
    plt.colorbar()
    plt.title(
        "Flow length: {}\n Samples on iteration #{}"
        .format(flow_length, iteration)
    )
    plt.axis('equal')
    # plt.axis('tight')
    fig.savefig(os.path.join(directory, "flow_result_{}.png".format(iteration)))
    plt.close()


def plot_density(density, directory, X_LIMS=(-7,7), Y_LIMS=(-7,7)):

    x1 = np.linspace(*X_LIMS, 300)
    x2 = np.linspace(*Y_LIMS, 300)
    x1, x2 = np.meshgrid(x1, x2)
    shape = x1.shape
    x1 = x1.ravel()
    x2 = x2.ravel()

    z = np.c_[x1, x2]
    
    density_values = density(z)
    if type(density_values) == tf.Tensor:
        density_values = tf.Session().run(density_values)
    density_values = density_values.reshape(shape)

    fig = plt.figure(figsize=(7, 7))
    plt.imshow(density_values, extent=(*X_LIMS, *Y_LIMS), cmap="hot")
    plt.title("True density")
    plt.colorbar()
    fig.savefig(os.path.join(directory, "density.png"))
    plt.close()


def detJacHeatmap(domain, flowFcn, displaceFlag=False):
    """
     This function (numerically) computes the determinant of the jacobian of the deformation field.
     Inputs:
          domain : it is a list of tuples, the same way as it is used linspace function
          flowFcn :  flow function 
          displaceFlag : the input is not flow but displacement (dx,dy) (default : False)
    """
    import numpy as np
    import numdifftools as ndt

    x = np.linspace(*domain[0])
    y = np.linspace(*domain[1])
    XX, YY = np.meshgrid(x, y)
    coords = np.column_stack([XX.flatten(), YY.flatten()])


    d = coords.shape[1]
    Jfun = ndt.Jacobian(flowFcn)

    output = np.zeros(coords.shape[0],)
    for idx,c in enumerate(coords):
        jac = Jfun(c)
        if displaceFlag:
            jac = np.eye(d) + jac

        output[idx] =  np.linalg.det(jac.reshape(d,d))
        
    ratio_negative = sum(output<0)/(0.0 + len(output))

    print("Percentage of Det(J) < 0 : ", 100*ratio_negative )    
    output = output.reshape(XX.shape)

    return output, ratio_negative


def deformationGrid(flowFcn, domainRange, gridSize, numPoints, directory=None):
    """
    flowFcn : flow function
    gridSize : size of the grid, it is a list
    domainRange: it is a list of tuples specifying the range of the domain
    numPoints : density of points in each line
    """
    ptsList = []
    mvPtsList = []
    
    X = np.linspace(domainRange[0][0],domainRange[0][1],numPoints)
    Y = np.linspace(domainRange[1][0],domainRange[1][1],gridSize[1])

    for y in Y:
        p = np.zeros( (len(X),2) )
        p[:,0] = X
        p[:,1] = y
        mp = flowFcn(p)
        ptsList.append(p)
        mvPtsList.append(mp)


    X = np.linspace(domainRange[0][0],domainRange[0][1],gridSize[0])
    Y = np.linspace(domainRange[1][0],domainRange[1][1],numPoints)

    for x in X:
        p = np.zeros( (len(Y),2) )
        p[:,1] = Y
        p[:,0] = x
        mp = flowFcn(p) 
        ptsList.append(p)
        mvPtsList.append(mp)

    plt.figure()    
    plt.axis('equal')
    for pts in ptsList:
        plt.plot(pts[:,0],pts[:,1],'black')
    plt.title("Original Grid")
    if not(directory==None):
        plt.savefig(os.path.join(directory, "OriginalGrid.png"))
        plt.close()
    else:
        plt.show()

    plt.figure()    
    plt.axis('equal')
    for mp in mvPtsList:
        plt.plot(mp[:,0],mp[:,1],'black')
    plt.title("Deformed Grid")

    if directory:
        plt.savefig(os.path.join(directory, "DeformationGrid.png"))
        plt.close()
    else:
        plt.show()


def displacementField(flowFcn, domainRange, visualizeRange, directory=None, title=None):
    """
    flowFcn : flow function
    domainRange: it is a list of tuples specifying the range of the domain (the same as linspace)
    visualizeRange : list of tuples specifying the range of visualization
    """
    XX,YY = np.meshgrid(np.linspace(*domainRange[0]), 
                        np.linspace(*domainRange[1]) )
    N1 = len(XX.ravel())
    gridPoints = np.zeros((N1,2))
    gridPoints[:,0] = XX.ravel()
    gridPoints[:,1] = YY.ravel()

    # make a displacement field
    displField = flowFcn(gridPoints) - gridPoints

    plt.figure()
    ax = plt.axes()
    ax.set_title('Disaplacement Field - $\Delta \phi_1 $')
    ax.set_ylim(*visualizeRange[0])
    ax.set_xlim(*visualizeRange[1])
    for l in range(0,N1):
        if displField[l, 0] == displField[l, 1] == 0.0:
            displField[l, 0] = 0.0001 # avoid undefined arrow

        ax.arrow(gridPoints[l, 0], gridPoints[l, 1],
                 displField[l, 0], displField[l, 1],
                 head_width=0.2, head_length=0.2,length_includes_head=True,
                 fc="grey", ec='k')
    
    if directory:
        if title:
            plt.savefig(os.path.join(directory, title + ".png"))    
        else:
            plt.savefig(os.path.join(directory, "displacementField.png"))
        plt.close()
    else:
        plt.show()
