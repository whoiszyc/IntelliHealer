.. _install:

*************************
Installation
*************************

IntelliHealer can be installed in Python 3.7. We can start with a new environment as follows:

.. code-block:: bash

    $ conda create -n envname python=3.7
    $ conda activate envname

Dependency
===========
Before install ``IntelliHealer``, please install the following dependencies:

.. code-block:: bash

    $ conda install numpy=1.19.5
    $ conda install pandas=0.24.2
    $ conda install scipy=1.6.0
    $ conda install gym=1.6.0
    $ conda install -c conda-forge stable-baselines3
    $ conda install -c conda-forge pyomo==5.6.1
    $ conda install -c conda-forge tensorflow=1.14
    $ conda install pytorch=1.7.1
    $ conda install networkx
    $ pip install keras==2.3.0

Then, we need to install the cplex solver 1280 (must be 1280 version).

- Go to the IBM academic initiatives, login and download CPLEX studio, and install.

- Go to local directory of CPLEX, browse into the Python directory inside. The directory generally looks like:

.. code-block:: bash

    $ /Applications/CPLEX_Studio128/cplex/python/3.6/x86-64_osx/

- Install the Cplex 1280 python interface:

.. code-block:: bash

    $ python setup.py install




Performance Packages
====================
After successfully install all dependencies, navigate to the root directory of ``IntelliHealer`` and install it using the following command

.. code-block:: bash

    $ python setup.py install
