# CHARA array 
This project made up 10% of my part II physics course - it was my first time using data science techniques which were motivated from first principles. 

This can be used to find the positions of the telescopes at different points in time. 

Basic knowledge of NumPy, Pandas and Matplotlib is assumed. 

![image](https://user-images.githubusercontent.com/64110421/173235328-0011d2b5-f39e-45a7-926c-c3b8e66a7ea2.png)

## What is the problem 
The angular resolution of a telescope is the smallest angle between objects that can be visibly resolved and is proportional to the telescope diameter. Although, when using two telescopes together, the angular resolution is set by the distance between the telescopes and not the sizes of the individual telescopes. This means two telescopes that are distance $D$ apart have the same resolving capability (but not light collecting capability) as a larger continuous telescope of diameter $D$. To get the telescope joining benefits, the light in both branches must have travelled the same distance down to the tenths of wavelength to see an image (due to finite coherence lengths, this is explainined clearly in any optics textbook). 

The first telescope to take advantage of this effect was the Michelson stellar interferometer in 1920 when it was set up on Mount Wilson, California to image the star Betelgeuse. Today the Centre for High Angular Resolution Astronomy (CHARA) array sits on Mount Wilson. CHARA consists of six 1m telescopes arranged in a Y-shape with baseline distance distances between telescopes ranging from 34m to 330m. It is possible to use measurements from the CHARA array to find the positions of the telescopes relative to each other and find whether there has been significant movement of the telescopes between different dates.


### The physics of this problem is unimportant to understanding the analysis. All you need to know is that there is an equation describing the path difference between two different telescopes. When measurements are taken the path difference vanishes

$$\text{optical path difference}= \hat{\pmb{S}} \cdot (\pmb{r}_1 - \pmb{r}_2) + d_1 - d_2$$

Where $d_i = \text{CART}_i + \text{POP}_i$ is the path difference due to the projection of the sum vector and the optical path length equaliser

So when the optical path difference is zero we get the equation...

$$\text{CART}_2 - \text{CART}_1 = \hat{\pmb{S}} \cdot (\pmb{r}_1 - \pmb{r}_2) + \text{POP}_1 - \text{POP}_2$$

## Linear regression from first principles 
As there is an equation for each measurement this can be turned into a matrix. A (pseudo)inverse can be found to invert this and find the model parameters e.g. the positions of the telescopes.



## Project useful to learn... 
- Pandas 
- NumPy arrays and vectorisation
- Mathematics behind linear regression (design matrices and Moore-Penrose pseudoinverse)
- $\LaTeX$




Use data from CHARA telescope array [1] to find telescope positions relative to each other using least-squares regression via Moore-Penrose pseudoinverse.
Analysis over time to conclude whether significant movement has occured (e.g. due to plate tectonics).


[1] Ten Brummelaar, T. A., et al. "First results from the CHARA Array. II. A description of the instrument." The Astrophysical Journal 628.1 (2005): 453.

### Sources 
- Original paper
- CHARA website
- Numerical recipes 
- My final write up
